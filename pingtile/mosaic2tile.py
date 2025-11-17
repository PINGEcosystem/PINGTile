
'''
Copyright (c) 2025 Cameron S. Bodine
'''

#########
# Imports

import sys, os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import rasterio as rio
from shapely.ops import unary_union
from shapely.geometry import shape, box
import geopandas as gpd

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

    # # debug
    # mosaic_reproj = r'Z:\scratch\HabiMapper_Test\R00107_rect_wcr_mosaic_0_reproj.tif'
        
    # Get the moving window
    movWin = getMovingWindow_rast(sonRast=mosaic_reproj, windowSize=windowSize, windowStride_m=windowStride_m)

    # # Subset movWin gdf to those that intersect mask_reproj (mosaic geotiff)
    # # For raster: create polygon from non-nodata pixels
    # with rio.open(mosaic_reproj) as src:
    #     # Read first band
    #     data = src.read(1)
    #     # Create mask of valid (non-nodata) pixels
    #     if src.nodata is not None:
    #         valid_mask = data != src.nodata
    #     else:
    #         # If no nodata value, assume 0 or NaN are invalid
    #         valid_mask = (data != 0) & ~np.isnan(data)

    #     # Set valid values to 1, invalid to 0
    #     valid_mask = valid_mask.astype('uint8')
    #     valid_mask[valid_mask >= 1] = 1
        
    #     # Extract shapes (polygons) from valid data regions
    #     shapes_gen = rio.features.shapes(valid_mask.astype('uint8'), mask=valid_mask, transform=src.transform)
        
    #     # Collect all valid data polygons
    #     geoms = [shape(geom) for geom, val in shapes_gen if val == 1]
        
    #     if geoms:
    #         # Combine all valid data polygons into one geometry
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

    # Debug save geodataframe to shp
    out_file = os.path.join(outDir, 'mov_win.shp')
    movWin.to_file(out_file, driver='ESRI Shapefile')

    # Do moving window
    total_win = len(movWin)
    r = Parallel(n_jobs=threadCnt)(delayed(doMovWin)(i, movWin.iloc[i], mosaic_reproj, target_size, outDir, outName, minArea_percent, windowSize) for i in tqdm(range(total_win)))

    sampleInfoAll = []
    # sampleInfoAll += r
    for v in r:
        if v is not None:
            sampleInfoAll.append(v)

    dfAll = pd.DataFrame(sampleInfoAll)

    # os.remove(mosaic_reproj) 
    
    return dfAll