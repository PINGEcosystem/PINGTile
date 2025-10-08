
'''
Copyright (c) 2025 Cameron S. Bodine
'''

#########
# Imports

import sys, os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

from pingtile.utils import reproject_raster, getMovingWindow_rast, doMovWin

#=======================================================================
def doMosaic2tile(inFile: str,
                  outDir: str,
                  windowSize: tuple,
                  windowStride_m: int,
                  outName: str='',
                  epsg_out: int=4326,
                  threadCnt: int=1,
                  target_size: list=[512,512],
                  minArea_percent: float=0.5
                  ):

    '''
    Generate tiles from input mosaic.
    '''

    # Reproject raster to epsg_out (if necessary)
    mosaic_reproj = reproject_raster(src_path=inFile, dst_path=outDir, dst_crs=epsg_out)
        
    # Get the moving window
    movWin = getMovingWindow_rast(sonRast=mosaic_reproj, windowSize=windowSize, windowStride_m=windowStride_m)

    # Do moving window
    total_win = len(movWin)
    r = Parallel(n_jobs=threadCnt)(delayed(doMovWin)(i, movWin.iloc[i], mosaic_reproj, target_size, outDir, outName, minArea_percent, windowSize) for i in tqdm(range(total_win)))

    sampleInfoAll = []
    # sampleInfoAll += r
    for v in r:
        if v is not None:
            sampleInfoAll.append(v)

    dfAll = pd.DataFrame(sampleInfoAll)

    os.remove(mosaic_reproj) 
    
    return dfAll