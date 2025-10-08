
'''
Copyright (c) 2025 Cameron S. Bodine
'''

#########
# Imports

import os, sys
from joblib import Parallel, delayed, cpu_count

# # Debug
# from imglbl2tile import doImgLbl2tile

# For Package
from pingtile.imglbl2tile import doImgLbl2tile

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

    for sonarFile in sonarFiles:

        print(f"\nProcessing {os.path.basename(sonarFile)} with windowSize: {windowSize} and windowStride_m: {windowStride_m}...\n")

        doImgLbl2tile(inFileSonar=sonarFile,
                      inFileMask=map,
                      outDir=outDir,
                      outName=outName,
                      epsg_out=epsg_out,
                      classCrossWalk=classCrossWalk,
                      windowSize=windowSize,
                      windowStride_m=windowStride_m,
                      classFieldName=classFieldName,
                      minArea_percent=minArea_percent,
                      target_size=target_size,
                      threadCnt=threadCnt,
                      doPlot=doPlot
                      )