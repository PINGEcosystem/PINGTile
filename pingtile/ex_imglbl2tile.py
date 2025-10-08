
'''
Copyright (c) 2025 Cameron S. Bodine
'''

#########
# Imports

import os, sys
from joblib import Parallel, delayed, cpu_count

# Debug
from imglbl2tile import doImgLbl2tile
from utils import mask_to_coco_json

# # For Package
# from pingtile.imglbl2tile import doImgLbl2tile
# from pingtile.utils import mask_to_coco_json

import rasterio as rio
import json

############
# Parameters

map = r'Z:\tmp\pingtile_test\map\Model_Training_Substrate_Polygons_Export.shp'
sonarDir = r'Z:\tmp\pingtile_test\mosaic'

outDirTop = r'Z:\tmp\pingtile_test'
outName = 'Hudson'

classCrossWalk = {
    '0':0,
    'U':1,
    'G':2,
    'B_C':3,
    'B':4
}

windowSize_m = [
                (12,12),
                (18,18),
                (24,24),
                ]

windowStride = 3
classFieldName = 'Substrate_'
minArea_percent = 0.5
target_size = (512, 512) #(1024, 1024)
threadCnt = 0.75
epsg_out = 32616
doPlot = True
lbl2COCO = True

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


# for windowSize in windowSize_m:

#     # windowStride_m = windowStride*windowSize[0]
#     windowStride_m = windowStride
#     # minArea = minArea_percent * windowSize[0]*windowSize[1]

#     dirName = f"{windowSize[0]}_{windowSize[0]}"
#     outDir = os.path.join(outDirTop, dirName)
#     outSonDir = os.path.join(outDir, 'images')
#     outMaskDir = os.path.join(outDir,'labels')
#     pltDir = os.path.join(outDir,'plots')

#     if not os.path.exists(outSonDir):
#         os.makedirs(outSonDir)
#         os.makedirs(outMaskDir)
#         os.makedirs(pltDir)

#     for sonarFile in sonarFiles:

#         print(f"\nProcessing {os.path.basename(sonarFile)} with windowSize: {windowSize} and windowStride_m: {windowStride_m}...\n")

#         doImgLbl2tile(inFileSonar=sonarFile,
#                       inFileMask=map,
#                       outDir=outDir,
#                       outName=outName,
#                       epsg_out=epsg_out,
#                       classCrossWalk=classCrossWalk,
#                       windowSize=windowSize,
#                       windowStride_m=windowStride_m,
#                       classFieldName=classFieldName,
#                       minArea_percent=minArea_percent,
#                       target_size=target_size,
#                       threadCnt=threadCnt,
#                       doPlot=doPlot
#                       )

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

        maskFiles=maskFiles[:10] # Debug limit to 10 files

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

            # try to find corresponding image filename in images folder (same base name)
            matched_image = None
            for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                candidate = os.path.join(outSonDir, base + ext)
                if os.path.exists(candidate):
                    matched_image = os.path.basename(candidate)
                    break
            if matched_image is None:
                # fallback to mask basename (acceptable as file_name in COCO)
                matched_image = os.path.basename(mask_path)

            # read mask to get width/height
            try:
                with rio.open(mask_path) as src:
                    width, height = src.width, src.height
            except Exception as e:
                print(f"Skipping {mask_path}: cannot read ({e})")
                continue

            image_info = {
                "id": image_id,
                "file_name": matched_image,
                "width": width,
                "height": height
            }

            # mask_to_coco_json should return (annotations_list, next_annotation_id)
            anns, annotation_id = mask_to_coco_json(mask_path, image_info, categories_info, annotation_id)

            if anns:
                coco["images"].append(image_info)
                coco["annotations"].extend(anns)
                image_id += 1

        out_json = os.path.join(outJsonDir, f"_annotations.coco.json")
        with open(out_json, "w") as f:
            json.dump(coco, f)

        
        

        