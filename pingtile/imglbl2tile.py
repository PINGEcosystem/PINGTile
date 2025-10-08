
'''
Copyright (c) 2025 Cameron S. Bodine
'''

#########
# Imports

import os, sys
from joblib import Parallel, delayed
from tqdm import tqdm

# # Debug
# from utils import reproject_raster, getMovingWindow_rast, doMovWin, reproject_shp, doMovWin_imgshp

from pingtile.utils import reproject_raster, getMovingWindow_rast, doMovWin, reproject_shp, doMovWin_imgshp


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
                  doPlot: bool=False
                  ):
    
    '''
    Generate tiles from input sonar image and label mask.
    '''

    # Reproject raster to epsg_out (if necessary)
    sonar_reproj = reproject_raster(src_path=inFileSonar, dst_path=outDir, dst_crs=epsg_out)

    # Check if mask ends with .shp
    if inFileMask.lower().endswith('.shp'):
        mask_reproj = reproject_shp(src_path=inFileMask, dst_crs=epsg_out)
    else:
        mask_reproj = reproject_raster(src_path=inFileMask, dst_path=outDir, dst_crs=epsg_out)

    # Get the moving window
    movWin = getMovingWindow_rast(sonRast=sonar_reproj, windowSize=windowSize, windowStride_m=windowStride_m)

    # # save to file
    # outFile = os.path.join(outDir, 'movWin.shp')
    # movWin.to_file(outFile, driver='ESRI Shapefile')

    # print(movWin)

    ##################
    # Do moving window
    total_win = len(movWin)

    # # First on sonar_reproj
    # outDir_sonar = os.path.join(outDir, 'images')
    # if not os.path.exists(outDir_sonar):
    #     os.makedirs(outDir_sonar)
    
    # _ = Parallel(n_jobs=threadCnt)(delayed(doMovWin)(i, movWin.iloc[i], sonar_reproj, target_size, outDir_sonar, outName, minArea, windowSize) for i in range(total_win))

    # os.remove(sonar_reproj)

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
        _ = Parallel(n_jobs=threadCnt)(delayed(doMovWin_imgshp)(i=i, movWin=movWin.iloc[i], mosaic=sonar_reproj, shp=mask_reproj, target_size=target_size, outSonDir=outSonDir, outMaskDir=outMaskDir, outPltDir=outPltDir, outName=outName, classFieldName=classFieldName, minArea_percent=minArea_percent, windowSize=windowSize, classCrossWalk=classCrossWalk, doPlot=doPlot) for i in tqdm(range(total_win)))

    return