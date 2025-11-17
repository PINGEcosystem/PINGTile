
'''
Copyright (c) 2025 Cameron S. Bodine
'''

#########
# Imports

import os, sys
from joblib import Parallel, delayed
from tqdm import tqdm
import rasterio as rio

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

    # # Subset movWin gdf to those that intersect mask_reproj (mosaic geotiff)
    # # For raster: create polygon from non-nodata pixels
    # with rio.open(mask_reproj) as src:
    #     # Read first band
    #     data = src.read(1)
    #     # Create mask of valid (non-nodata) pixels
    #     if src.nodata is not None:
    #         valid_mask = data != src.nodata
    #     else:
    #         # If no nodata value, assume 0 or NaN are invalid
    #         valid_mask = (data != 0) & ~np.isnan(data)
        
    #     # Extract shapes (polygons) from valid data regions
    #     shapes_gen = features.shapes(valid_mask.astype('uint8'), mask=valid_mask, transform=src.transform)
        
    #     # Collect all valid data polygons
    #     geoms = [shape(geom) for geom, val in shapes_gen if val == 1]
        
    #     if geoms:
    #         # Combine all valid data polygons into one geometry
    #         from shapely.ops import unary_union
    #         data_footprint = unary_union(geoms)
            
    #         # Ensure same CRS
    #         if movWin.crs != src.crs:
    #             footprint_gdf = gpd.GeoDataFrame([1], geometry=[data_footprint], crs=src.crs)
    #             footprint_gdf = footprint_gdf.to_crs(movWin.crs)
    #             data_footprint = footprint_gdf.geometry.iloc[0]
            
    #         # Filter windows that intersect the actual data footprint
    #         movWin = movWin[movWin.intersects(data_footprint)]
    #     else:
    #         # No valid data found
    #         movWin = movWin.iloc[0:0]  # empty GeoDataFrame

    # # save to file
    # outFile = os.path.join(outDir, 'movWin.shp')
    # movWin.to_file(outFile, driver='ESRI Shapefile')

    # # print(movWin)

    # sys.exit()

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