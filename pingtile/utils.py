'''
Copyright (c) 2025 Cameron S. Bodine
'''

#########
# Imports
import os, sys
from osgeo import gdal
import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import shapely
import numpy as np
from skimage.transform import resize

#========================================================
def reproject_raster(src_path: str, 
                     dst_path: str, 
                     dst_crs: str):

    file_name = os.path.basename(src_path)
    file_type = file_name.split('.')[-1]
    out_file = file_name.replace('.'+file_type, '_reproj.tif')
    dst_path = os.path.join(dst_path, out_file)

    if os.path.exists(dst_path):
        try:
            os.remove(dst_path)
        except:
            pass
    
    cell_size = 0.05

    dst_tmp = dst_path.replace('.tif', '_tmp.tif')

    with rio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'count': 1,  # Ensure single band
            'dtype': 'uint8'  # Greyscale
        })

        src_crs = int(str(src.crs).split(':')[-1])

        if src_crs == dst_crs:
            return src_path

        with rio.open(dst_tmp, 'w', **kwargs) as dst:
            reproject(
                source = rio.band(src, 1),
                destination=rio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest if src_path.endswith('.png') else Resampling.bilinear,
                dst_nodata=src.nodata
            )

    t = gdal.Warp(dst_path, dst_tmp, xRes = cell_size, yRes = cell_size, targetAlignedPixels=True)

    t = None

    os.remove(dst_tmp)

    return dst_path

#========================================================
def getMovingWindow(sonRast: str,
                    windowSize: tuple,
                    windowStride_m: int):

    # Open the raster
    with rio.open(sonRast) as sonRast:

        # Calculate window size and stride in pixels
        windowSize_px = (
            int(windowSize[0] / sonRast.res[0]),
            int(windowSize[1] / sonRast.res[1])
        )
        windowStride_px = int(windowStride_m / sonRast.res[0])

        movWindow = []

        # Create moving windows
        for i in range(0, sonRast.width - windowSize_px[0] + 1, windowStride_px):
            for j in range(0, sonRast.height - windowSize_px[1] + 1, windowStride_px):
                window = rio.windows.Window(i, j, windowSize_px[0], windowSize_px[1])
                window_transform = sonRast.window_transform(window)
                window_extent = rio.windows.bounds(window, transform=window_transform)
                movWindow.append(window_extent)

        # Convert movWindow into a gdf
        # Convert movWindow into a list of geometries
        geometries = [shapely.geometry.box(extent[0], extent[1], extent[2], extent[3]) for extent in movWindow]

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=sonRast.crs)

    return gdf

#========================================================
def doMovWin(i: int,
             movWin: gpd.GeoDataFrame,
             mosaic: str,
             target_size: list,
             outSonDir: str,
             outName: str,
             minArea_percent: float,
             windowSize: tuple,
             ):

    mosaicName = os.path.basename(mosaic)

    # Open the raster
    with rio.open(mosaic) as sonRast:

        window_geom = movWin.geometry

        # Get the bounds
        window_bounds = window_geom.bounds

        win_coords = ''
        for b in window_bounds:
            b = int(round(b, 0))

            win_coords += str(b)+'_'

        win_coords = win_coords[:-1]
        
        try:
            clipped_mosaic, clipped_transform = mask(sonRast, [window_geom], crop=True)

            clipped_mosaic = clipped_mosaic[0, :, :]

            # Check if there is data in clipped mosaic
            if np.any(clipped_mosaic > 0):
                # There is data > 0 in the clipped mosaic
                # Resize to target_size
                clipped_raster_resized = resize(clipped_mosaic, target_size, preserve_range=True, anti_aliasing=True).astype('uint8')

                # Calculate the percentage of non-zero pixels
                non_zero_percentage = np.count_nonzero(clipped_raster_resized) / clipped_raster_resized.size

                # Check if the cropped raster has any valid (non-zero) values
                if clipped_raster_resized.any() and non_zero_percentage >= minArea_percent:
                        # Recalculate the transform for the resized raster
                    new_transform = rio.transform.from_bounds(
                        window_bounds[0], window_bounds[1], window_bounds[2], window_bounds[3],
                        target_size[1], target_size[0]
                    )

                    # Save the clipped raster and shapefile
                    mosaicName = mosaicName.split('.tif')[0]
                    if outName:
                        fileName = f"{outName}_{mosaicName}_{windowSize[0]}m_{win_coords}"
                    else:
                        fileName = f"{mosaicName}_{windowSize[0]}m_{win_coords}"
                    out_raster_path = os.path.join(outSonDir, 'images', f"{fileName}.png")
                    
                    with rio.open(
                        out_raster_path,
                        'w',
                        driver='GTiff',
                        height=clipped_raster_resized.shape[0],
                        width=clipped_raster_resized.shape[1],
                        count=1,
                        dtype=clipped_raster_resized.dtype,
                        crs=sonRast.crs,
                        transform=new_transform,
                    ) as dst:
                        dst.write(clipped_raster_resized, 1)

                    

                    # Store everythining in a dictionary
                    sampleInfo = {'mosaic': mosaic,
                                    'window_size': windowSize[0],
                                    'x_min': window_bounds[0],
                                    'y_min': window_bounds[1],
                                    'x_max': window_bounds[2],
                                    'y_max': window_bounds[3],
                                    'total_pix': clipped_raster_resized.shape[0]*clipped_raster_resized.shape[1],
                                    'nonzero_prop': non_zero_percentage}

                    return sampleInfo
                pass
            else:
                # No data > 0 in the clipped mosaic
                pass

        except:
            pass



# #========================================================
# def doMovWin_img_lbl(i: int, 
#                      movWin: gpd.GeoDataFrame, 
#                      lbl: str, 
#                      mosaic: str
#                      ):

#     mosaicName = os.path.basename(mosaic)

#     # Open the raster
#     sonRast = rio.open(mosaic)
#     maskRast = rio.open(lbl)

#     window_geom = movWin.geometry

#     # Get the bounds
#     window_bounds = window_geom.bounds

#     win_coords = ''
#     for b in window_bounds:
#         b = int(round(b, 0))

#         win_coords += str(b)+'_'

#     win_coords = win_coords[:-1]
    
#     try:
#         clipped_mosaic, clipped_transform = rio.mask(sonRast, [window_geom], crop=True)
#         clipped_mask, clipped_transform = rio.mask(maskRast, [window_geom], crop=True)

#         clipped_mosaic = clipped_mosaic[0, :, :]
#         clipped_mask = clipped_mask[0, :, :]

#         # Set 0 to 1 and 255 to 0
#         clipped_mask[clipped_mask == 0] = 1
#         clipped_mask[clipped_mask == 255] = 0

#         # Check if there is data in clipped mosaic
#         if np.any(clipped_mosaic > 0):
#             # There is data > 0 in the clipped mosaic
#             # Resize to target_size
#             clipped_raster_resized = resize(clipped_mosaic, target_size, preserve_range=True, anti_aliasing=True).astype('uint8')
#             clipped_mask_resized = resize(clipped_mask, target_size, preserve_range=True, anti_aliasing=True).astype('uint8')

#             # Calculate the percentage of non-zero pixels
#             non_zero_percentage = np.count_nonzero(clipped_raster_resized) / clipped_raster_resized.size

#             # Check if the cropped raster has any valid (non-zero) values
#             if clipped_raster_resized.any() and non_zero_percentage >= minArea_percent:
#                     # Recalculate the transform for the resized raster
#                 new_transform = rio.transform.from_bounds(
#                     window_bounds[0], window_bounds[1], window_bounds[2], window_bounds[3],
#                     target_size[1], target_size[0]
#                 )

#                 # Save the clipped raster and shapefile
#                 mosaicName = mosaicName.split('.tif')[0]
#                 if outName:
#                     fileName = f"{outName}_{mosaicName}_{windowSize[0]}m_{win_coords}"
#                 else:
#                     fileName = f"{mosaicName}_{windowSize[0]}m_{win_coords}"
#                 out_raster_path = os.path.join(outSonDir, f"{fileName}.png")
                
#                 with rio.open(
#                     out_raster_path,
#                     'w',
#                     driver='GTiff',
#                     height=clipped_raster_resized.shape[0],
#                     width=clipped_raster_resized.shape[1],
#                     count=1,
#                     dtype=clipped_raster_resized.dtype,
#                     crs=sonRast.crs,
#                     transform=new_transform,
#                 ) as dst:
#                     dst.write(clipped_raster_resized, 1)

#                 # Save the rasterized habitat map
#                 out_rasterized_path = os.path.join(outMaskDir, f"{fileName}.png")
#                 with rio.open(
#                     out_rasterized_path,
#                     'w',
#                     driver='GTiff',
#                     height=clipped_raster_resized.shape[0],
#                     width=clipped_raster_resized.shape[1],
#                     count=1,
#                     dtype=clipped_raster_resized.dtype,
#                     crs=sonRast.crs,
#                     transform=new_transform,
#                 ) as dst:
#                     dst.write(clipped_mask_resized, 1)

#                 # Get class count
#                 noData_cnt = np.sum(clipped_raster_resized == 0)
#                 sonData_cnt = np.sum(clipped_raster_resized > 0)
#                 fishData_cnt = np.sum(clipped_mask_resized == 1)
#                 noFishData_cnt = sonData_cnt - fishData_cnt

#                 # Store everythining in a dictionary
#                 sampleInfo = {'mosaic': mosaic,
#                                 'habitat': carp_mask,
#                                 'window_size': windowSize[0],
#                                 'x_min': window_bounds[0],
#                                 'y_min': window_bounds[1],
#                                 'x_max': window_bounds[2],
#                                 'y_max': window_bounds[3],
#                                 'fishGroup_cnt': fishData_cnt,
#                                 'noFishGroup_cnt': noFishData_cnt,
#                                 'noData_cnt': noData_cnt,
#                                 'total_pix': clipped_mask_resized.shape[0]*clipped_mask_resized.shape[1]}
                
#                 print(sampleInfo)
                
#                 # Make a plot
#                 img_f = out_raster_path
#                 lbl_f = out_rasterized_path

#                 img = imread(img_f)
#                 lbl = imread(lbl_f)

#                 plt.imshow(img, cmap='gray')

#                 #blue,red, yellow,green, etc
#                 class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
#                                         '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
#                                         '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

#                 color_label = label_to_colors(lbl, img[:,:]==0,
#                                     alpha=128, colormap=class_label_colormap,
#                                         color_class_offset=0, do_alpha=False)

#                 plt.imshow(color_label,  alpha=0.5)

#                 file = os.path.basename(img_f)
#                 out_file = os.path.join(pltDir, file)


#                 plt.axis('off')
#                 plt.title(file)
#                 plt.savefig(out_file, dpi=200, bbox_inches='tight')
#                 plt.close('all')

#                 return sampleInfo                                


#             pass
#         else:
#             # No data > 0 in the clipped mosaic
#             pass

#     except:
#         pass

