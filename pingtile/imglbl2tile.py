
'''
Copyright (c) 2025- Cameron S. Bodine
'''

#########
# Imports

import os, sys
from collections import Counter
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm
import rasterio as rio

# # Debug
# from utils import reproject_raster_keep_bands, reproject_raster_gray, getMovingWindow_rast, doMovWin, reproject_shp, doMovWin_imgshp, getMaskFootprint

from pingtile.utils import reproject_raster_gray, reproject_raster_keep_bands, getMovingWindow_rast, doMovWin, reproject_shp, doMovWin_imgshp, getMaskFootprint


#=======================================================================
def doImgLbl2tile(inFileSonar: str,
                  inFileMask: str,
                  outDir: str,
                  outName: str,
                  epsg_out: int,
                  classCrossWalk: dict,
                  windowSize: tuple,
                  windowStride_m: float,
                  classFieldName: str='',
                  minArea_percent: float=0.5,
                  target_size: tuple=(512,512),
                  threadCnt: int=4,
                  doPlot: bool=False,
                  allowNoMapTiles: bool=False,
                  grayscale: bool=False
                  ):
    
    '''
    Generate tiles from input sonar image and label mask.
    '''

    # Check src_path band count
    with rio.open(inFileSonar) as src:
        bandCnt = src.count
    
    # Limit band count to 3 if greater than 3
    if bandCnt > 3:
        bandCnt = 3

    # Reproject raster to epsg_out (if necessary)
    if bandCnt >= 3:
        mosaic_reproj, _ = reproject_raster_keep_bands(src_path=inFileSonar, dst_dir=outDir, dst_crs=epsg_out)
    else:
        mosaic_reproj, _ = reproject_raster_gray(src_path=inFileSonar, dst_path=outDir, dst_crs=epsg_out)

    # # Reproject raster to epsg_out (if necessary)
    # mosaic_reproj = reproject_raster(src_path=inFileSonar, dst_path=outDir, dst_crs=epsg_out)

    # Check if mask ends with .shp
    if inFileMask.lower().endswith('.shp'):
        mask_reproj = reproject_shp(src_path=inFileMask, dst_crs=epsg_out)
    else:
        mask_reproj, _ = reproject_raster_gray(src_path=inFileMask, dst_path=outDir, dst_crs=epsg_out)

    # Get the moving window
    movWin = getMovingWindow_rast(sonRast=mosaic_reproj, windowSize=windowSize, windowStride_m=windowStride_m)

    ########################
    # Optimize moving window 
    # ## by subsetting to only those that intersect the mask_reproj

    maskFootprint = getMaskFootprint(sonPath=mosaic_reproj)

    if maskFootprint is not None:
        # Filter windows that intersect the actual data footprint
        # Use .apply() to properly handle the geometry comparison
        movWin = movWin[movWin.geometry.intersects(maskFootprint)].reset_index(drop=True)

    # # Save moving window to file for debugging
    # outFile = os.path.join(outDir, 'movWin.shp')
    # os.makedirs(outDir, exist_ok=True)
    # movWin.to_file(outFile, driver='ESRI Shapefile')

    ##################
    # Do moving window
    total_win = len(movWin)

    # # First on mosaic_reproj
    # outDir_sonar = os.path.join(outDir, 'images')
    # if not os.path.exists(outDir_sonar):
    #     os.makedirs(outDir_sonar)
    
    # _ = Parallel(n_jobs=threadCnt)(delayed(doMovWin)(i, movWin.iloc[i], mosaic_reproj, target_size, outDir_sonar, outName, minArea, windowSize) for i in range(total_win))

    # os.remove(mosaic_reproj)

    # # Then on mask_reproj
    # outDir_mask = os.path.join(outDir, 'masks')
    # if not os.path.exists(outDir_mask):
    #     os.makedirs(outDir_mask)
    
    # if mask_reproj.lower().endswith('.shp'):

    #     _ = Parallel(n_jobs=threadCnt)(delayed(doMovWin_shp)(i, movWin.iloc[i], mask_reproj, target_size, outDir_mask, outName, classFieldName, minArea, windowSize, classCrossWalk) for i in range(total_win))

    # else:
    #     _ = Parallel(n_jobs=threadCnt)(delayed(doMovWin)(i, movWin.iloc[i], mask_reproj, target_size, outDir_mask, outName, minArea_percent, windowSize) for i in range(total_win))

    # os.remove(mask_reproj)

    outSonDir = os.path.join(outDir, 'images')
    outMaskDir = os.path.join(outDir, 'labels')
    outPltDir = os.path.join(outDir, 'plots')

    if mask_reproj.lower().endswith('.shp'):
        results = []
        recycle_size = 250
        with tqdm(total=total_win, desc='Processing windows') as progress:
            for start in range(0, total_win, recycle_size):
                stop = min(start + recycle_size, total_win)
                with Parallel(
                    n_jobs=threadCnt,
                    batch_size='auto',
                    pre_dispatch='2 * n_jobs',
                ) as parallel:
                    batch_results = parallel(
                        delayed(doMovWin_imgshp)(
                            i=i,
                            movWin=movWin.geometry.iloc[i],
                            mosaic=mosaic_reproj,
                            shp=mask_reproj,
                            target_size=target_size,
                            outSonDir=outSonDir,
                            outMaskDir=outMaskDir,
                            outPltDir=outPltDir,
                            outName=outName,
                            classFieldName=classFieldName,
                            minArea_percent=minArea_percent,
                            windowSize=windowSize,
                            classCrossWalk=classCrossWalk,
                            doPlot=doPlot,
                            allowNoMapTiles=allowNoMapTiles,
                            grayscale=grayscale,
                        )
                        for i in range(start, stop)
                    )
                get_reusable_executor().shutdown(wait=True, kill_workers=True)
                results.extend(batch_results)
                progress.update(stop - start)

        status_counter = Counter()
        exported = []
        for result in results:
            if not isinstance(result, dict):
                status_counter['skipped_unknown'] += 1
                continue

            if result.get('status') == 'skipped':
                status_counter[f"skipped_{result.get('reason', 'unknown')}"] += 1
                continue

            tile_kind = result.get('tile_export_kind', 'classified')
            status_counter[f"exported_{tile_kind}"] += 1
            exported.append(result)

        print(
            "[Tile summary] "
            + ", ".join(f"{k}={v}" for k, v in sorted(status_counter.items()))
        )

        return exported

    return []