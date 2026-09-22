
'''
Copyright (c) 2025-2026 Cameron S. Bodine
'''

#########
# Imports

import os, sys
import re
from joblib import Parallel, delayed, cpu_count
from PIL import Image
import numpy as np
import pandas as pd

# Debug
# Add current directory to path for testing
sys.path.append(os.path.dirname(__file__))
# from imglbl2tile import doImgLbl2tile
# from utils import mask_to_coco_json

# For Package
from pingtile.imglbl2tile import doImgLbl2tile
from pingtile.utils import build_case_insensitive_basename_lookup, mask_to_coco_json

import rasterio as rio
import json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

if __name__ == "__main__":
    ############
    # Parameters

    # Map can be specified as a directory containing all map files, or a single map file to use for all mosaics.
    map = r"D:\redbo_science\projects\USGS-CERC_2025\00_carp_group_targets\mask2shp"

    # Sonar Directory can be specified as a directory containing all sonar files, or a single sonar file to process (if map is a single file).
    sonarDir = r"D:\redbo_science\projects\USGS-CERC_2025\00_carp_group_targets\Mosaics"

    outDirTop = r'D:\redbo_science\projects\USGS-CERC_2025\01_img-lbl_v2'
    outName = 'carp_tiles'

    classCrossWalk = {
        'background': 0,
        'none': 1,          # No carp
        'carp school': 2,   # Carp school
    }

    # classCrossWalk = {
    #     'background': 0,
    #     'fines': 1,
    #     'sand': 2,
    #     'gravelf': 3,
    #     'gravelc': 4,
    #     'boulder': 5,
    #     'bedrock': 6,
    #     'mask': 255
    # }

    # classCrossWalk = {
    #     '0':0,
    #     'U':1,
    #     'G':2,
    #     'B_C':3,
    #     'B':4
    # }

    windowSize_m = [
                    # (12,12),
                    # (18,18),
                    (24,24),
                    # (36,36),
                    ]

    windowStride = 12
    classFieldName = 'class'
    minArea_percent = 0.5
    target_size = (512, 512) #(1024, 1024)
    threadCnt = 0.75
    epsg_out = 32616
    doPlot = True
    lbl2COCO = True
    allowNoMapTiles = False  # Set True to also export sonar-covered tiles that have no map overlap.
    grayscale = True  # Set True to convert multi-band sonar images to one grayscale band.
    strict_mask_validation = True  # Abort export if unexpected mask class values are detected.

    if not os.path.exists(outDirTop):
        os.makedirs(outDirTop)


    ###############################################
    # Specify multithreaded processing thread count
    if threadCnt==0: # Use all threads
        threadCnt=cpu_count()
    elif threadCnt<0: # Use all threads except threadCnt; i.e., (cpu_count + (-threadCnt))
        threadCnt=cpu_count()+threadCnt
        if threadCnt<0: # Make sure not negative
            threadCnt=1
    elif threadCnt<1: # Use proportion of available threads
        threadCnt = int(cpu_count()*threadCnt)
        # Make even number
        if threadCnt % 2 == 1:
            threadCnt -= 1
    else: # Use specified threadCnt if positive
        pass

    if threadCnt>cpu_count(): # If more than total avail. threads, make cpu_count()
        threadCnt=cpu_count();
        print("\nWARNING: Specified more process threads then available, \nusing {} threads instead.".format(threadCnt))

    print("\nUsing {} threads for processing.\n".format(threadCnt))


    # Find sonar files from either a directory or one explicitly selected mosaic.
    if os.path.isfile(sonarDir):
        if not sonarDir.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"Sonar file must be a TIFF raster: {sonarDir}")
        sonarFiles = [sonarDir]
    elif os.path.isdir(sonarDir):
        sonarFiles = []
        for root, dirs, files in os.walk(sonarDir):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    sonarFiles.append(os.path.join(root, file))
    else:
        raise FileNotFoundError(f"Sonar path does not exist: {sonarDir}")

    print(f"Found {len(sonarFiles)} sonar files for processing.")
    if len(sonarFiles) == 0:
        raise FileNotFoundError(f"No sonar files found under: {sonarDir}")


    # Resolve map input: either one map file for all mosaics, or directory of per-mosaic maps.
    map_is_dir = os.path.isdir(map)
    single_map_file: str | None = None
    map_lookup: dict[str, str] = {}

    if map_is_dir:
        print(f"Map input mode: directory pairing from {map}")
        map_exts = ('.shp', '.tif', '.tiff')
        map_files = []
        for root, dirs, files in os.walk(map):
            for file in files:
                if file.lower().endswith(map_exts):
                    map_files.append(os.path.join(root, file))

        if len(map_files) == 0:
            raise FileNotFoundError(f"No map files (*.shp, *.tif, *.tiff) found under: {map}")

        map_lookup = build_case_insensitive_basename_lookup(map_files)

        print(f"Found {len(map_lookup)} map lookup keys for pairing: {sorted(map_lookup.keys())}")
        if len(map_lookup) != len(map_files):
            print(
                "WARNING: Found duplicate map basenames and kept first match for some files."
            )
    else:
        if not os.path.exists(map):
            raise FileNotFoundError(f"Map path does not exist: {map}")
        print(f"Map input mode: single map file applied to all mosaics: {map}")
        single_map_file = map

    _pingmapper_suffix_re = re.compile(r'(_rect_wcr|_wcr)?_mosaic(_\d+)?$', re.IGNORECASE)


    def _audit_mask_class_values(mask_path: str, valid_values: set[int]):
        """Warn or fail if unexpected non-zero mask values appear."""
        try:
            data = np.asarray(Image.open(mask_path))
        except Exception:
            return

        unique_values = {int(v) for v in np.unique(data).tolist() if int(v) != 0}
        invalid_values = sorted(v for v in unique_values if v not in valid_values)
        if invalid_values:
            msg = (
                f"{os.path.basename(mask_path)} contains unexpected mask values {invalid_values}; "
                "these values will be skipped during COCO export."
            )
            if strict_mask_validation:
                raise ValueError(msg)
            print(f"WARNING: {msg}")


    def _build_tile_coco_record(file_name: str, outSonDir: str, outMaskDir: str, categories_info: dict):
        """Build one tile's image metadata and local annotations.

        Annotation IDs are intentionally local and are reassigned globally later.
        """
        image_path = os.path.join(outSonDir, f"{file_name}.png")
        mask_path = os.path.join(outMaskDir, f"{file_name}.png")
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            return None

        valid_values = set(int(v) for v in categories_info.keys())
        _audit_mask_class_values(mask_path, valid_values)

        with Image.open(image_path) as img:
            image_stub = {
                "id": 1,
                "file_name": os.path.basename(image_path),
                "width": img.width,
                "height": img.height,
            }

        local_annotations, _ = mask_to_coco_json(mask_path, image_stub, categories_info, 1)
        return {
            "file_name": file_name,
            "image_stub": image_stub,
            "annotations": local_annotations,
        }


    def export_mosaic_coco(tile_csv: str, outSonDir: str, outMaskDir: str, outJsonDir: str, classCrossWalk: dict, mosaic_label: str):
        '''Build and save a COCO json for the tiles listed in one mosaic's tile-info CSV.'''
        tiles_df = pd.read_csv(tile_csv)
        if tiles_df.empty or 'file_name' not in tiles_df.columns:
            print(f"No tile file names found in {tile_csv}; skipping COCO export.")
            return

        # categories_info passed to mask_to_coco_json should map id -> name
        categories_info = {v: str(k) for k, v in classCrossWalk.items()}
        # COCO categories (exclude background id 0 if present)
        categories = [{"id": v, "name": str(k)} for k, v in classCrossWalk.items() if v != 0]

        coco = {
            "info": {"description": mosaic_label},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }

        file_names = tiles_df['file_name'].tolist()
        worker_count = max(1, min(int(threadCnt), len(file_names)))

        tile_records = Parallel(n_jobs=worker_count, prefer="threads")(
            delayed(_build_tile_coco_record)(fn, outSonDir, outMaskDir, categories_info)
            for fn in tqdm(file_names, desc=f"COCO export ({mosaic_label})", unit="tile")
        )

        annotation_id = 1
        image_id = 1

        for tile_record in tile_records:
            if tile_record is None:
                continue

            image_info = dict(tile_record["image_stub"])
            image_info["id"] = image_id
            coco["images"].append(image_info)

            for ann in tile_record["annotations"]:
                ann["id"] = annotation_id
                ann["image_id"] = image_id
                coco["annotations"].append(ann)
                annotation_id += 1

            image_id += 1

        out_json = os.path.join(outJsonDir, f"{mosaic_label}_annotations.coco.json")
        with open(out_json, "w") as f:
            json.dump(coco, f, indent=2)

        print(
            f"COCO JSON saved to {out_json} with {len(coco['images'])} images "
            f"and {len(coco['annotations'])} annotations using {worker_count} worker(s)."
        )


    for windowSize in windowSize_m:

        # windowStride_m = windowStride*windowSize[0]
        windowStride_m = windowStride
        # minArea = minArea_percent * windowSize[0]*windowSize[1]

        dirName = f"{windowSize[0]}_{windowSize[0]}"
        outDir = os.path.join(outDirTop, dirName)
        outSonDir = os.path.join(outDir, 'images')
        outMaskDir = os.path.join(outDir,'labels')
        pltDir = os.path.join(outDir,'plots')

        outJsonDir = os.path.join(outDir, 'json')

        if not os.path.exists(outSonDir):
            os.makedirs(outSonDir)
            os.makedirs(outMaskDir)
            os.makedirs(pltDir)

        if lbl2COCO and not os.path.exists(outJsonDir):
            os.makedirs(outJsonDir)

        processed_cnt = 0
        skipped_cnt = 0
        print(f"\nStarting tiling for window size {windowSize}...\n")

        for sonarFile in sonarFiles:

            sonar_path = str(sonarFile)
            sonar_base = os.path.splitext(os.path.basename(sonar_path))[0]
            map_file: str | None

            if map_is_dir:
                normalized_sonar = re.sub(r'(_rect_wcr|_wcr)(?=_mosaic(?:_\d+)?$)', '', sonar_base, flags=re.IGNORECASE).lower()
                candidate_keys = [normalized_sonar]

                # Preserve indexed mosaic names (e.g. mosaic_0 vs mosaic_12) because they
                # are distinct files and should never be collapsed to the same lookup key.
                base_without_ext = normalized_sonar
                for suffix in ('_reproj', '_map', '_shp', '_tif', '_tiff', '_polygon', '_polygons'):
                    if base_without_ext.endswith(suffix):
                        base_without_ext = base_without_ext[:-len(suffix)]
                        break
                candidate_keys.append(base_without_ext)

                if '_mosaic_' not in normalized_sonar:
                    # Trailing _N index fallback (e.g. transect_1 -> transect).
                    candidate_keys.append(re.sub(r'_\d+$', '', normalized_sonar))
                    # Date-token fallback for YYYYMMDD/YYYYDDD mismatches between file sets.
                    candidate_keys.append(re.sub(r'_\d{6,8}(?=_|$)', '', normalized_sonar))

                map_file = None
                for key in dict.fromkeys(candidate_keys):  # preserve order, skip dupes
                    if key in map_lookup:
                        map_file = map_lookup[key]
                        break
            else:
                map_file = single_map_file

            if map_file is None:
                print(f"Skipping {os.path.basename(sonar_path)}: no valid map path resolved.")
                skipped_cnt += 1
                continue

            print(
                f"\nProcessing sonar={os.path.basename(sonar_path)} "
                f"with map={os.path.basename(map_file)} "
                f"windowSize={windowSize} windowStride_m={windowStride_m}...\n"
            )

            mosaic_records = doImgLbl2tile(inFileSonar=sonarFile,
                          inFileMask=map_file,
                          outDir=outDir,
                          outName=outName,
                          epsg_out=epsg_out,
                          classCrossWalk=classCrossWalk,
                          windowSize=windowSize,
                          windowStride_m=windowStride_m,
                          classFieldName=classFieldName,
                          minArea_percent=minArea_percent,
                          target_size=target_size,
                          threadCnt=int(threadCnt),
                          doPlot=doPlot,
                          allowNoMapTiles=allowNoMapTiles,
                          grayscale=grayscale
                          )

            processed_cnt += 1

            # Write this mosaic's tile-info CSV and export its COCO json right away,
            # instead of waiting until every mosaic has been processed.
            if mosaic_records:
                mosaic_tile_csv = os.path.join(outDir, f"{outName}_{sonar_base}_tile_info.csv")
                pd.DataFrame(mosaic_records).to_csv(mosaic_tile_csv, index=False)
                print(f"Tile info CSV saved to {mosaic_tile_csv} with {len(mosaic_records)} tiles.")

                if lbl2COCO:
                    print(f"\nConverting {sonar_base} tiles to COCO format for windowSize: {windowSize}...\n")
                    export_mosaic_coco(mosaic_tile_csv, outSonDir, outMaskDir, outJsonDir, classCrossWalk, sonar_base)
            else:
                print(f"No tiles exported for {os.path.basename(sonar_path)}; skipping tile CSV/COCO export.")

        print(
            f"Completed window size {windowSize}: processed={processed_cnt}, "
            f"skipped_missing_map={skipped_cnt}."
        )

    print("\nWorkflow complete.")


        
        

        