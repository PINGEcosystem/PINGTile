'''
Shared mapper orchestration workflow for PING-based mappers.
'''

#########
# Imports
import json
import os
import shutil
import time
import zipfile
from glob import glob
from os import cpu_count

import pandas as pd
import requests

from pingtile.mosaic2tile import doMosaic2tile
from pingtile.utils import avg_npz_files, map_npzs, maps2Shp, mosaic_maps


#=======================================================================
def resolve_thread_count(threadCnt: float) -> int:
    '''
    Resolve requested thread count into a safe integer worker count.
    '''
    if threadCnt == 0:
        threadCnt = cpu_count()
    elif threadCnt < 0:
        threadCnt = cpu_count() + threadCnt
        if threadCnt < 1:
            threadCnt = 1
    elif threadCnt < 1:
        threadCnt = int(cpu_count() * threadCnt)
        if threadCnt % 2 == 1:
            threadCnt -= 1
            if threadCnt < 1:
                threadCnt = 1
    else:
        threadCnt = int(threadCnt)

    if threadCnt > cpu_count():
        threadCnt = cpu_count()
        print(
            "\nWARNING: Specified more process threads than available, "
            "using {} threads instead.".format(threadCnt)
        )

    return threadCnt


#=======================================================================
def download_model_if_needed(modelDir: str, releases_repo: str):
    '''
    Download and extract model artifacts if local model directory is missing/empty.
    '''
    if os.path.exists(modelDir) and len(os.listdir(modelDir)) > 0:
        return

    os.makedirs(modelDir, exist_ok=True)

    seg_model = os.path.basename(modelDir)
    url = f'https://github.com/PINGEcosystem/{releases_repo}/releases/download/models/{seg_model}.zip'
    print(f'\n\nDownloading segmentation models (v1.0): {url}')

    filename = modelDir + '.zip'
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        if not zipfile.is_zipfile(filename):
            with open(filename, 'rb') as fh:
                head = fh.read(512).decode('utf-8', errors='replace')
            os.remove(filename)
            raise RuntimeError(f"Downloaded file is not a zip file. Server response starts with: {head!r}")

        with zipfile.ZipFile(filename, 'r') as z_fp:
            z_fp.extractall(modelDir)

        os.remove(filename)
        print('Model download and extraction success!')

    except requests.RequestException as e:
        if os.path.exists(filename):
            os.remove(filename)
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e
    except zipfile.BadZipFile as e:
        if os.path.exists(filename):
            os.remove(filename)
        raise RuntimeError('Downloaded model archive is corrupted or not a zip file.') from e


#=======================================================================
def resolve_model_config(modelDir: str):
    '''
    Resolve model config file from common package layouts.
    '''
    config_candidates = [
        os.path.join(modelDir, 'config.json'),
    ]
    config_candidates += sorted(glob(os.path.join(modelDir, 'config', '*.json')))
    config_candidates = [c for c in config_candidates if os.path.exists(c)]

    if not config_candidates:
        raise FileNotFoundError(f'No config file found in model directory: {modelDir}')

    configFile = config_candidates[0]

    with open(configFile) as f:
        config = json.load(f)

    return configFile, config


#=======================================================================
def resolve_target_size(config: dict):
    '''
    Resolve target image size from both legacy and transformers config layouts.
    '''
    if 'image_size' in config:
        return (int(config['image_size']), int(config['image_size']))

    if 'TARGET_SIZE' in config:
        return (int(config['TARGET_SIZE'][0]), int(config['TARGET_SIZE'][1]))

    return (224, 224)


#=======================================================================
def default_predict_tiles(imagesDF: pd.DataFrame,
                          modelDir: str,
                          out_npz: str,
                          predBatchSize: int,
                          threadCnt: int):
    '''
    Select backend automatically and run tile prediction.
    '''
    has_torch_model = os.path.exists(os.path.join(modelDir, 'model.safetensors'))
    has_tf_model = bool(glob(os.path.join(modelDir, 'weights', '*.h5'))) or bool(glob(os.path.join(modelDir, '*.h5')))

    if has_torch_model:
        from pingseg.seg_torch import seg_torch_folder

        return seg_torch_folder(
            imgDF=imagesDF,
            modelDir=modelDir,
            out_dir=out_npz,
            batch_size=predBatchSize,
            threadCnt=threadCnt,
        )

    if has_tf_model:
        from pingseg.seg_gym import seg_gym_folder

        return seg_gym_folder(
            imgDF=imagesDF,
            modelDir=modelDir,
            out_dir=out_npz,
            batch_size=predBatchSize,
            threadCnt=threadCnt,
        )

    raise FileNotFoundError(
        f"Could not determine model backend for {modelDir}. "
        'Expected model.safetensors (Transformers) or one/more .h5 files (Segmentation Gym).'
    )


#=======================================================================
def run_mapper_workflow(
    *,
    mapper_name: str,
    releases_repo: str,
    inDir: str,
    outDirTop: str,
    modelDir: str,
    projName: str,
    mapRast: bool,
    mapShp: bool,
    epsg: int,
    windowSize_m: tuple,
    window_stride: int,
    minArea_percent: float,
    threadCnt: float,
    mosaicFileType: str,
    predBatchSize: int,
    deleteIntData: bool = True,
    minPatchSize: float = 3,
    smoothShp: bool = False,
    smoothTol_m: float = 0.5,
    print_usage=None,
    predict_tiles=None,
    debug: bool = False,
    list_mosaics: bool = False,
):
    '''
    Shared end-to-end mapper execution workflow.
    '''

    outDir = os.path.join(outDirTop, projName)

    if os.path.exists(outDir) and not debug:
        shutil.rmtree(outDir)

    os.makedirs(outDir, exist_ok=True)

    download_model_if_needed(modelDir=modelDir, releases_repo=releases_repo)

    threadCnt = resolve_thread_count(threadCnt)

    to_delete = {}

    outSonDir = os.path.join(outDir, 'images')
    os.makedirs(outSonDir, exist_ok=True)

    mosaics = glob(os.path.join(inDir, '**', f'*{mosaicFileType}'), recursive=True)
    if len(mosaics) == 0:
        raise FileNotFoundError(f'No mosaic files matching *{mosaicFileType} found under {inDir}')

    if list_mosaics:
        print('\nFound {} mosaic files.'.format(len(mosaics)))
        for mosaic in mosaics:
            print(mosaic)

    configFile, config = resolve_model_config(modelDir)
    target_size = resolve_target_size(config)

    print('\n\nTiling Mosaics...\n\n')
    start_time = time.time()

    imagesAll = []
    for mosaic in mosaics:
        r = doMosaic2tile(
            inFile=mosaic,
            outDir=outSonDir,
            windowSize=windowSize_m,
            windowStride_m=window_stride,
            outName=projName,
            epsg_out=epsg,
            threadCnt=threadCnt,
            target_size=target_size,
            minArea_percent=minArea_percent,
        )
        if list_mosaics:
            tile_cnt = len(r) if r is not None else 0
            print(f"Tiles accepted from {os.path.basename(mosaic)}: {tile_cnt}")
        imagesAll.append(r)

    imagesDF = pd.concat(imagesAll, axis=0, ignore_index=True)

    outDF = os.path.join(outDir, f'{projName}_{windowSize_m[0]}_{windowSize_m[1]}_tiles.csv')
    if not deleteIntData:
        imagesDF.to_csv(outDF, index=False)

    if deleteIntData:
        to_delete['outSonDir'] = [outSonDir]

    print('Image Tiles Generated: {}'.format(len(imagesDF)))
    if len(imagesDF) == 0:
        print('[WARNING] No tiles met inclusion criteria. Check raster nodata/value scale and consider lowering minArea_percent.')

    print('\nDone!')
    print('Time (s):', round(time.time() - start_time, ndigits=1))
    if print_usage:
        print_usage()

    if not deleteIntData:
        imagesDF = pd.read_csv(outDF)

    print('\n\nPredicting substrate from sonar tiles...\n\n')
    start_time = time.time()

    out_npz = os.path.join(outDir, 'preds_npz')

    if predict_tiles is None:
        predict_tiles = default_predict_tiles

    imagesDF = predict_tiles(
        imagesDF=imagesDF,
        modelDir=modelDir,
        out_npz=out_npz,
        predBatchSize=predBatchSize,
        threadCnt=threadCnt,
    )

    outDF = os.path.join(outDir, f'{projName}_{windowSize_m[0]}_{windowSize_m[1]}_tileseg.csv')
    if not deleteIntData:
        imagesDF.to_csv(outDF, index=False)

    if deleteIntData:
        to_delete['out_npz'] = [out_npz]
        shutil.rmtree(outSonDir, ignore_errors=True)

    print('\nPrediction Complete!')
    print('Time (s):', round(time.time() - start_time, ndigits=1))
    if print_usage:
        print_usage()

    if not deleteIntData:
        imagesDF = pd.read_csv(outDF)

    print('\n\nAveraging overlapping substrate predictions...\n\n')
    start_time = time.time()

    out_avg_npz = os.path.join(outDir, 'preds_avg_npz')
    os.makedirs(out_avg_npz, exist_ok=True)

    gdf = avg_npz_files(
        df=imagesDF,
        in_dir=out_npz,
        out_dir=out_avg_npz,
        outName=projName,
        windowSize_m=windowSize_m,
        stride=windowSize_m[0],
        epsg=epsg,
        threadCnt=threadCnt,
    )

    outDF = os.path.join(outDir, f'{projName}_{windowSize_m[0]}_{windowSize_m[1]}_avgnpz.csv')
    if not deleteIntData:
        gdf.to_csv(outDF, index=False)

    if deleteIntData:
        to_delete['out_avg_npz'] = [out_avg_npz]
        shutil.rmtree(out_npz, ignore_errors=True)

    print('\nDone!')
    print('Time (s):', round(time.time() - start_time, ndigits=1))
    if print_usage:
        print_usage()

    if not deleteIntData:
        gdf = pd.read_csv(outDF)

    print('\n\nMapping predicted substrate...\n\n')
    start_time = time.time()

    out_maps = os.path.join(outDir, 'preds_mapped')
    os.makedirs(out_maps, exist_ok=True)

    minPatchSize_tile = minPatchSize / 4.0

    gdf = map_npzs(
        df=gdf,
        in_dir=out_avg_npz,
        out_dir=out_maps,
        outName=projName,
        minPatchSize=minPatchSize_tile,
        windowSize_m=windowSize_m,
        epsg=epsg,
        threadCnt=threadCnt,
    )

    if deleteIntData:
        shutil.rmtree(out_avg_npz, ignore_errors=True)

    print('\nDone!')
    print('Time (s):', round(time.time() - start_time, ndigits=1))
    if print_usage:
        print_usage()

    outDF = os.path.join(outDir, f'{projName}_{windowSize_m[0]}_{windowSize_m[1]}_mapped_npzs.csv')
    if not deleteIntData:
        gdf.to_csv(outDF, index=False)

    if deleteIntData:
        to_delete['out_maps'] = [out_maps]

    if mapRast:
        print('\n\nExport map as raster mosaic...\n\n')
        start_time = time.time()

        map_files = glob(os.path.join(out_maps, '*.tif'))
        out_mosaic = os.path.join(outDir, 'mosaic')
        os.makedirs(out_mosaic, exist_ok=True)

        mosaic_maps(map_files, out_mosaic, projName)

        print('\nDone!')
        print('Time (s):', round(time.time() - start_time, ndigits=1))
        if print_usage:
            print_usage()

    if mapShp:
        print('\n\nExport map as shapefile...\n\n')
        start_time = time.time()

        map_files = glob(os.path.join(out_maps, '*.tif'))
        out_shp = os.path.join(outDir, 'map_shp')
        os.makedirs(out_shp, exist_ok=True)

        maps2Shp(map_files, out_shp, projName, configFile, minPatchSize, [1], smoothShp, smoothTol_m)

        print('\nDone!')
        print('Time (s):', round(time.time() - start_time, ndigits=1))
        if print_usage:
            print_usage()

    if deleteIntData:
        print('\n\nDeleting intermediate data...\n\n')
        start_time = time.time()

        for name, paths in to_delete.items():
            for path in paths:
                print(f'Deleting intermediate data: {name} -> {path}')
                shutil.rmtree(path, ignore_errors=True)

        print('\nDone!')
        print('Time (s):', round(time.time() - start_time, ndigits=1))
        if print_usage:
            print_usage()

    print(f'\n\n{mapper_name} run complete.\n')
