
'''
Copyright (c) 2025 Cameron S. Bodine
'''

#########
# Imports

import os, sys
import re
from joblib import Parallel, delayed, cpu_count
from PIL import Image

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

############
# Parameters

# Map can be specified as a directory containing all map files, or a single map file to use for all mosaics.
map = r"Z:\UDEL\Projects\SAV_DESG_CBIG\data\20260811_LukeMerrit_Maps\Polygons SHP Files\Germany"

# Sonar Directory can be specified as a directory containing all sonar files, or a single sonar file to process (if map is a single file).
sonarDir = r"Z:\UDEL\Projects\SAV_DESG_CBIG\data\20260811_LukeMerrit_Maps\Training Mosaics\Germany"

outDirTop = r'Z:\UDEL\Projects\SAV_DESG_CBIG\data\20260811_LukeMerrit_Maps\pingtiles\Germany'
outName = 'sav_tiles'

classCrossWalk = {
    'background': 0,
    'p': 1,             # SAV present
    'n': 2,             # SAV absent
    'u': 3,             # SAV unknown
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
                ]

windowStride = 12
classFieldName = 'SAV'
minArea_percent = 0.5
target_size = (512, 512) #(1024, 1024)
threadCnt = 0.75
epsg_out = 32618
doPlot = True
lbl2COCO = True
allowNoMapTiles = False  # Set True to also export sonar-covered tiles that have no map overlap.

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


# Find all sonar files
sonarFiles = []
for root, dirs, files in os.walk(sonarDir):
    for file in files:
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            sonarFiles.append(os.path.join(root, file))

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

for windowSize in windowSize_m:

    # windowStride_m = windowStride*windowSize[0]
    windowStride_m = windowStride
    # minArea = minArea_percent * windowSize[0]*windowSize[1]

    dirName = f"{windowSize[0]}_{windowSize[0]}"
    outDir = os.path.join(outDirTop, dirName)
    outSonDir = os.path.join(outDir, 'images')
    outMaskDir = os.path.join(outDir,'labels')
    pltDir = os.path.join(outDir,'plots')

    if not os.path.exists(outSonDir):
        os.makedirs(outSonDir)
        os.makedirs(outMaskDir)
        os.makedirs(pltDir)

    processed_cnt = 0
    skipped_cnt = 0

    print(f"\nStarting tiling for window size {windowSize}...\n")

    for sonarFile in sonarFiles:

        sonar_path = str(sonarFile)
        sonar_base = os.path.splitext(os.path.basename(sonar_path))[0]
        map_file: str | None

        if map_is_dir:
            normalized_sonar = _pingmapper_suffix_re.sub('', sonar_base).lower()
            candidate_keys = [normalized_sonar]
            for suffix in ('_reproj', '_map', '_shp', '_tif', '_tiff', '_polygon', '_polygons'):
                if normalized_sonar.endswith(suffix):
                    candidate_keys.append(normalized_sonar[:-len(suffix)])
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

        doImgLbl2tile(inFileSonar=sonarFile,
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
                      allowNoMapTiles=allowNoMapTiles
                      )

        processed_cnt += 1

    print(
        f"Completed window size {windowSize}: processed={processed_cnt}, "
        f"skipped_missing_map={skipped_cnt}."
    )

# Convert masks to COCO format
if lbl2COCO:
    

    for windowSize in windowSize_m:

        dirName = f"{windowSize[0]}_{windowSize[0]}"
        outDir = os.path.join(outDirTop, dirName)
        outSonDir = os.path.join(outDir, 'images')
        outMaskDir = os.path.join(outDir,'labels')
        pltDir = os.path.join(outDir,'plots')
        outJsonDir = os.path.join(outDir,'json')

        if not os.path.exists(outJsonDir):
            os.makedirs(outJsonDir)

        print(f"\nConverting to COCO format for windowSize: {windowSize}...\n")

        # Get the mask files
        maskFiles = []
        for root, dirs, files in os.walk(outMaskDir):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                    maskFiles.append(os.path.join(root, file))

        print(f"Found {len(maskFiles)} mask tiles for COCO conversion in {outMaskDir}.")
        if len(maskFiles) == 0:
            print("Skipping COCO conversion for this window size because no mask tiles were found.")
            continue

        # maskFiles=maskFiles[:10] # Debug limit to 10 files

        # Build categories list / lookup from classCrossWalk
        # categories_info passed to mask_to_coco_json should map id -> name
        categories_info = {v: str(k) for k, v in classCrossWalk.items()}
        # COCO categories (exclude background id 0 if present)
        categories = [{"id": v, "name": str(k)} for k, v in classCrossWalk.items() if v != 0]

        coco = {
            "info": {"description": outName or ""},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }

        annotation_id = 1
        image_id = 1

        for mask_path in maskFiles:
            base = os.path.splitext(os.path.basename(mask_path))[0]
            # Find corresponding image
            image_path = os.path.join(outSonDir, base + '.png')
            if not os.path.exists(image_path):
                continue
            
            # Add image entry
            img = Image.open(image_path)
            image_info = {
                "id": image_id,
                "file_name": os.path.basename(image_path),
                "width": img.width,
                "height": img.height
            }
            coco["images"].append(image_info)
            
            # Convert mask to COCO annotations
            annotations, annotation_id = mask_to_coco_json(mask_path, image_info, categories_info, annotation_id)
            coco["annotations"].extend(annotations)
            annotation_id += len(annotations)
            image_id += 1

        out_json = os.path.join(outJsonDir, f"_annotations.coco.json")
        with open(out_json, "w") as f:
            json.dump(coco, f, indent=2)
        
        print(f"COCO JSON saved to {out_json} with {len(coco['images'])} images and {len(coco['annotations'])} annotations.")

print("\nWorkflow complete.")


        
        

        