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
from rasterio.transform import from_origin
import geopandas as gpd
import pandas as pd
import shapely
import numpy as np
from skimage.transform import resize, warp, PiecewiseAffineTransform
from joblib import Parallel, delayed
from tqdm import tqdm
from shapely.geometry import box, shape
from PIL import ImageColor

from skimage.io import imsave, imread
import matplotlib.pyplot as plt

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


#========================================================
def reproject_shp(src_path: str, 
                  dst_crs: str,
                  ):
    '''
    Rproject shapefile to dst_crs and save to new shapefile.
    '''

    out_file = src_path.replace('.shp', '_reproj.shp')

    if os.path.exists(out_file):
        try:
            os.remove(out_file)
        except:
            pass

    gdf = gpd.read_file(src_path)

    gdf = gdf.to_crs(epsg=int(dst_crs))

    # # Reclassify based on classCrossWalk
    # outClassName = '_reclass_'

    # gdf[outClassName] = gdf[classFieldName].map(classCrossWalk).fillna(0).astype(int)

    gdf.to_file(out_file, driver='ESRI Shapefile')

    return out_file

#========================================================
def getMovingWindow_rast(sonRast: str,
                         windowSize: tuple,
                         windowStride_m: float):

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
def getMovingWindow(df: pd.DataFrame,
                         windowSize: tuple,
                         windowStride_m: int,
                         epsg: int = 4326):

    '''
    '''

    min_x = df['x_min'].min()
    max_x = df['x_max'].max()
    min_y = df['y_min'].min()
    max_y = df['y_max'].max()

    movWindow = []

    x = min_x
    while x + windowSize[0] <= max_x:
        y = min_y
        while y + windowSize[1] <= max_y:
            extent = (x, y, x + windowSize[0], y + windowSize[1])
            movWindow.append(extent)
            y += windowStride_m
        x += windowStride_m    

    # Convert movWindow into a gdf
    # Convert movWindow into a list of geometries
    geometries = [shapely.geometry.box(extent[0], extent[1], extent[2], extent[3]) for extent in movWindow]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=f'epsg:{epsg}')

    return gdf

#========================================================
def doMovWin_imgshp(i: int,
                 movWin: gpd.GeoDataFrame,
                 mosaic: str,
                 shp: str,
                 target_size: list,
                 outSonDir: str,
                 outMaskDir: str,
                 outPltDir: str,
                 outName: str,
                 classFieldName: str,
                 minArea_percent: float,
                 windowSize: tuple,
                 classCrossWalk: dict={},
                 doPlot: bool=False
                 ):

    minArea = minArea_percent * windowSize[0]*windowSize[1]
    
    mosaicName = os.path.basename(mosaic)
    mosaicName = mosaicName.split('_reproj.tif')[0]

    # Open the raster
    sonRast = rio.open(mosaic)

    # Iterate each window
    # for i, movWin in movWin.iterrows():
    # print(f"{i} of {total_win}")
    # Get the geometry of the window
    window_geom = movWin.geometry

    # Get the bounds
    window_bounds = window_geom.bounds

    win_coords = ''
    for b in window_bounds:
        b = int(round(b, 0))

        win_coords += str(b)+'_'

    win_coords = win_coords[:-1]

    hmDF = gpd.read_file(shp)

    # Clip the habitat map using the window geometry
    clipped_hmDF = gpd.overlay(hmDF, gpd.GeoDataFrame(geometry=[window_geom], crs=hmDF.crs), how='intersection')


    # Calculate the area of the clipped habitat map
    clipped_hmDF['area'] = clipped_hmDF.geometry.area

    totalArea = clipped_hmDF['area'].sum()

    if totalArea >= minArea:

        # Calculate cross walk class
        clipped_hmDF['value'] = clipped_hmDF[classFieldName].map(classCrossWalk)

        # Calculate the total area for each class
        class_areas = clipped_hmDF.groupby(classFieldName)['area'].sum()
        class_areas /= totalArea

        class_areas = class_areas.to_dict()

        # Clip the raster using the window geometry
        try:
            clipped_raster, clipped_transform = mask(sonRast, [window_geom], crop=True)
            
            # Drop the first two dimensions of clipped_raster if they have a size of 1
            if clipped_raster.shape[0] == 1:
                clipped_raster = clipped_raster[0]
            if clipped_raster.shape[0] == 1:
                clipped_raster = clipped_raster[:, 0]

            # Resize to target_size
            clipped_raster_resized = resize(clipped_raster, target_size, preserve_range=True, anti_aliasing=True).astype('uint8')

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
                fileName = f"{outName}_{mosaicName}_{windowSize[0]}m_{win_coords}"
                out_raster_path = os.path.join(outSonDir, f"{fileName}.png")
                # out_shapefile_path = os.path.join(outMaskDir, f"{fileName}.shp")

                # Create a mask from clipped_raster_resized
                clipped_raster_mask = np.where(clipped_raster_resized > 0, 1, 0)

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

                # clipped_hmDF.to_file(out_shapefile_path)
                
                del clipped_raster_resized

                # Rasterize the clipped_hmDF based on the "value" field
                shapes = ((geom, value) for geom, value in zip(clipped_hmDF.geometry, clipped_hmDF['value']))
                rasterized_hmDF = rio.features.rasterize(
                    shapes,
                    out_shape=clipped_raster.shape,
                    transform=clipped_transform,
                    fill=0,
                    dtype=clipped_raster.dtype
                )

                # Resize to target_size
                clipped_raster_resized = resize(rasterized_hmDF, target_size, order=0, preserve_range=True, clip=True).astype('uint8')

                # Mask the habitat map
                clipped_raster_resized = (clipped_raster_resized * clipped_raster_mask).astype('uint8')

                # Save the rasterized habitat map
                out_rasterized_path = os.path.join(outMaskDir, f"{fileName}.png")
                with rio.open(
                    out_rasterized_path,
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
                                'habitat': shp,
                                'window_size': windowSize[0],
                                'x_min': window_bounds[0],
                                'y_min': window_bounds[1],
                                'x_max': window_bounds[2],
                                'y_max': window_bounds[3]}
                
                for k, v in class_areas.items():
                    sampleInfo[k] = v

                if doPlot:
                    # Make a plot
                    img_f = out_raster_path
                    lbl_f = out_rasterized_path

                    img = imread(img_f)
                    lbl = imread(lbl_f)

                    plt.imshow(img, cmap='gray')

                    #blue,red, yellow,green, etc
                    class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                                            '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                                            '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

                    color_label = label_to_colors(lbl, img[:,:]==0,
                                        alpha=128, colormap=class_label_colormap,
                                            color_class_offset=0, do_alpha=False)

                    plt.imshow(color_label,  alpha=0.5)

                    file = os.path.basename(img_f)
                    out_file = os.path.join(outPltDir, file)


                    plt.axis('off')
                    plt.title(file)
                    plt.savefig(out_file, dpi=200, bbox_inches='tight')
                    plt.close('all')

                return sampleInfo
        except:
            pass

##========================================================       
def label_to_colors(
        img,
        mask,
        alpha,  # =128,
        colormap,  # =class_label_colormap, #px.colors.qualitative.G10,
        color_class_offset,  # =0,
        do_alpha,  # =True
        ):
        """
        Take MxN matrix containing integers representing labels and return an MxNx4
        matrix where each label has been replaced by a color looked up in colormap.
        colormap entries must be strings like plotly.express style colormaps.
        alpha is the value of the 4th channel
        color_class_offset allows adding a value to the color class index to force
        use of a particular range of colors in the colormap. This is useful for
        example if 0 means 'no class' but we want the color of class 1 to be
        colormap[0].
        """

        colormap = [
            tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
            for h in [c.replace("#", "") for c in colormap]
        ]

        cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
        minc = np.min(img)
        maxc = np.max(img)

        for c in range(minc, maxc + 1):
            cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

        cimg[mask == 1] = (0, 0, 0)

        if do_alpha is True:
            return np.concatenate(
                (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
            )
        else:
            return cimg
    
##========================================================
def fromhex(n):
    """hexadecimal to integer"""
    return int(n, base=16)       

#========================================================
def doMovWin(i: int,
             movWin: gpd.GeoDataFrame,
             mosaic: str,
             target_size: list,
             outSonDir: str,
             outName: str,
             minArea_percent: float,
             windowSize: tuple,
             reclassify: dict={},
             ):
    
    print(i)

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
                # clipped_raster_resized = resize(clipped_mosaic, target_size, preserve_range=True, anti_aliasing=True).astype('uint8')
                clipped_raster_resized = clipped_mosaic

                # If reclassify is provided, reclassify the raster
                if reclassify:
                    reclass_map = np.vectorize(reclassify.get)(clipped_raster_resized, clipped_raster_resized)
                    clipped_raster_resized = reclass_map.astype('uint8')

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
                    
                    print(out_raster_path)
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
                    sampleInfo = {'mosaic': out_raster_path,
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

#========================================================
def avg_npz_files_batch(df: pd.DataFrame,
                        win: pd.Series,
                        arr_shape: tuple,
                        in_dir: str,
                        out_dir: str,
                        outName: str,
                        windowSize_m: tuple,
                        epsg: int,
                        ):
    
    '''
    '''

    win_minx, win_miny, win_maxx, win_maxy = win.geometry.bounds

    # Calculate pixel size for the window array
    win_height, win_width = arr_shape[1], arr_shape[2]  # (bands, height, width)
    win_pixel_size_x = (win_maxx - win_minx) / win_width
    win_pixel_size_y = (win_maxy - win_miny) / win_height


    win_coords = ''
    for b in win.geometry.bounds:
        b = int(round(b, 0))

        win_coords += str(b)+'_'

    win_coords = win_coords[:-1]

    # Find overlapping npz files
    overlaps = df[
        (df['x_min'] < win_maxx) & (df['x_max'] > win_minx) &
        (df['y_min'] < win_maxy) & (df['y_max'] > win_miny)
    ]

    if overlaps.empty:
        return

    # Determine output array shape for this window
    sum_arr = np.zeros(arr_shape, dtype=np.float64)
    count_arr = np.zeros((arr_shape[1], arr_shape[2]), dtype=np.int32)

    for _, row in overlaps.iterrows():
        base = os.path.splitext(os.path.basename(row['mosaic']))[0]
        npz_path = os.path.join(in_dir, f"{base}.npz")
        npz = np.load(npz_path)
        arr = npz['softmax']
        arr_minx, arr_miny, arr_maxx, arr_maxy = row[['x_min', 'y_min', 'x_max', 'y_max']]
        arr_shape = arr.shape  # (bands, height, width)
        arr_pixel_size_x = (arr_maxx - arr_minx) / arr_shape[2]
        arr_pixel_size_y = (arr_maxy - arr_miny) / arr_shape[1]


        # Calculate overlap in world coordinates
        overlap_minx = max(win_minx, arr_minx)
        overlap_maxx = min(win_maxx, arr_maxx)
        overlap_miny = max(win_miny, arr_miny)
        overlap_maxy = min(win_maxy, arr_maxy)

        # Convert world coordinates to array indices
        win_x0 = int(np.floor((overlap_minx - win_minx) / win_pixel_size_x))
        win_x1 = int(np.ceil((overlap_maxx - win_minx) / win_pixel_size_x))
        win_y0 = int(np.floor((overlap_miny - win_miny) / win_pixel_size_y))
        win_y1 = int(np.ceil((overlap_maxy - win_miny) / win_pixel_size_y))

        arr_x0 = int(np.floor((overlap_minx - arr_minx) / arr_pixel_size_x))
        arr_x1 = int(np.ceil((overlap_maxx - arr_minx) / arr_pixel_size_x))
        arr_y0 = int(np.floor((overlap_miny - arr_miny) / arr_pixel_size_y))
        arr_y1 = int(np.ceil((overlap_maxy - arr_miny) / arr_pixel_size_y))

        # Check for valid overlap and matching shapes
        win_slice_shape = (win_y1 - win_y0, win_x1 - win_x0)
        arr_slice_shape = (arr_y1 - arr_y0, arr_x1 - arr_x0)
        if (win_slice_shape[0] > 0 and win_slice_shape[1] > 0 and
            arr_slice_shape[0] > 0 and arr_slice_shape[1] > 0 and
            win_slice_shape == arr_slice_shape):
            sum_arr[:, win_y0:win_y1, win_x0:win_x1] += arr[:, arr_y0:arr_y1, arr_x0:arr_x1]
            count_arr[win_y0:win_y1, win_x0:win_x1] += 1


        # print('\n\n', win_x0, win_x1, win_y0, win_y1, sum_arr.shape)
        # print(arr_x0, arr_x1, arr_y0, arr_y1, arr.shape)
        # print('Overlap:', overlap_minx, overlap_miny, overlap_maxx, overlap_maxy)
        # print(arr_pixel_size_x, arr_pixel_size_y)

    # Avoid division by zero
    avg_arr = np.divide(sum_arr, count_arr, out=np.zeros_like(sum_arr), where=count_arr != 0)

    # print('\n\n', avg_arr)

    # Save to npz
    # Save the clipped raster and shapefile
    if outName:
        fileName = f"{outName}_{windowSize_m[0]}m_{win_coords}"
    else:
        fileName = f"{windowSize_m[0]}m_{win_coords}"
    out_npz = os.path.join(out_dir, f"{fileName}.npz")

    df['npz'] = out_npz

    np.savez_compressed(out_npz, softmax=avg_arr)

    # Convert to GeoDataFrame
    geometry = box(win_minx, win_miny, win_maxx, win_maxy)
    # df['geometry'] = gpd.GeoSeries.from_bounds(win_minx, win_miny, win_maxx, win_maxy, crs=f"EPSG:{epsg}").geometry
    df['geometry'] = gpd.GeoSeries([geometry], crs=f"EPSG:{epsg}")

    return df

#========================================================
def avg_npz_files(df: pd.DataFrame,
                  in_dir: str,
                  out_dir: str,
                  outName: str,
                  windowSize_m: tuple,
                  stride: int,
                  epsg: int):
    '''
    Average overlapping npz files
    '''

    # Get non-overlapping moving window geodataframe
    movWin = getMovingWindow(df=df, windowSize=windowSize_m, windowStride_m=stride, epsg=epsg)
    
    # Save moving window to shapefile
    out_file = os.path.join(out_dir, 'Map_Tiles.shp')
    movWin.to_file(out_file, driver='ESRI Shapefile')

    # Assume all arrays have the same shape and resolution
    # Load one sample to get array shape and pixel size
    base = os.path.splitext(os.path.basename(df.iloc[0]['mosaic']))[0]
    base = base.split('.png')[0]
    npz_path = os.path.join(in_dir, f"{base}.npz")
    sample_npz = np.load(npz_path)
    arr_shape = sample_npz['softmax'].shape

    # Use joblib to parallelize the averaging process
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(avg_npz_files_batch)(df, win, arr_shape, in_dir, out_dir, outName, windowSize_m, epsg)
        for idx, win in tqdm(movWin.iterrows(), total=len(movWin), desc="Processing windows")
    )

    results = [res for res in results if res is not None]
    results = pd.concat(results, ignore_index=True)

    results = gpd.GeoDataFrame(results, geometry='geometry', crs=movWin.crs)                    

    return results


# def label_array_to_shapefile(df, in_dir, out_dir, outName, windowSize_m, epsg):
#     """
#     Convert a label array to polygons and save as a shapefile.

#     label: 2D numpy array of class labels
#     transform: affine transform for the array (e.g., from rasterio)
#     out_shp: output shapefile path
#     """

#     # Load npz
#     # base = os.path.splitext(os.path.basename(df.iloc[0]['mosaic']))[0]
#     # npz_path = os.path.join(in_dir, f"{base}.npz")
#     npz = np.load(df['npz'].values[0])

#     softmax = npz['softmax']
#     label = np.argmax(softmax, axis=0).astype(np.uint8)  # Assuming softmax shape is (classes, height, width)

#     # x_min, y_min, x_max, y_max = df[['x_min', 'y_min', 'x_max', 'y_max']].values[0]

#     # transform = rio.transform.from_bounds(x_min, y_min, x_max, y_max, label.shape[1], label.shape[0])

#     geometry = df['geometry'].values[0]
#     transform = rio.transform.from_bounds(*geometry.bounds, label.shape[1], label.shape[0])

#     # Generate polygons from the label array
#     mask = label != 0  # Optional: mask out background if label 0 is background
#     results = (
#         {'properties': {'class': int(v)}, 'geometry': s}
#         for s, v in shapes(label, mask=mask, transform=transform)
#     )

#     # Convert to GeoDataFrame
#     geoms = []
#     classes = []
#     for result in results:
#         geoms.append(shape(result['geometry']))
#         classes.append(result['properties']['class'])
#     gdf = gpd.GeoDataFrame({'class': classes, 'geometry': geoms}, crs=f"EPSG:{epsg}")

#     # # Save to shapefile
#     # gdf.to_file(out_shp)

#     if len(gdf) == 0:
#         return None
    
#     return gdf

#========================================================
def label_array_to_raster(df, in_dir, out_dir, outName, windowSize_m, epsg, filt=50):
    """
    Convert a label array to polygons and save as a gtiff.

    label: 2D numpy array of class labels
    transform: affine transform for the array (e.g., from rasterio)
    out_shp: output shapefile path
    """

    # Load npz
    # base = os.path.splitext(os.path.basename(df.iloc[0]['mosaic']))[0]
    # npz_path = os.path.join(in_dir, f"{base}.npz")
    npz = np.load(df['npz'].values[0])

    softmax = npz['softmax']
    label = np.argmax(softmax, axis=0).astype(np.uint8)  # Assuming softmax shape is (classes, height, width)

    #################################
    # Prepare pixel (pix) coordinates
    ## Pix coordinates describe the size of the coordinates in pixels
    ## Coordinate Order
    ## top left of dat == port(range, 0)
    ## bot left of dat == star(range, 0)
    ## top next == port(range, 0+filt)
    ## bottom next == star(range, 0+filt)
    ## ....
    rows, cols = label.shape # Determine number rows/cols
    pix_cols = np.array([0, cols-1]) # Create array of column indices
    pix_rows = np.array([0, rows-1]) # Create array of row indices
    pix_rows, pix_cols = np.meshgrid(pix_rows, pix_cols) # Create grid arrays that we can stack together
    pixAll = np.dstack([pix_rows.flat, pix_cols.flat])[0] # Stack arrays to get final map of pix pixel coordinats [[row1, col1], [row2, col1], [row1, col2], [row2, col2]...]

    #######################################
    # Prepare destination (dst) coordinates
    ## Destination coordinates describe the geographic location in lat/lon
    ## or easting/northing that directly map to the pix coordinates.

    ###
    # Get top (port range) coordinates
    trkMeta = pd.read_csv(portTrkMetaFile)
    trkMeta = trkMeta[trkMeta['chunk_id']==chunk].reset_index(drop=False) # Filter df by chunk_id

    # Get range (outer extent) coordinates [xR, yR] to transposed numpy arrays
    xTop, yTop = trkMeta[xRange].to_numpy().T, trkMeta[yRange].to_numpy().T
    xyTop = np.vstack((xTop, yTop)).T # Stack the arrays

    ###
    # Get bottom (star range) coordinates
    trkMeta = pd.read_csv(starTrkMetaFile)
    trkMeta = trkMeta[trkMeta['chunk_id']==chunk].reset_index(drop=False) # Filter df by chunk_id

    # Get range (outer extent) coordinates [xR, yR] to transposed numpy arrays
    xBot, yBot = trkMeta[xRange].to_numpy().T, trkMeta[yRange].to_numpy().T
    xyBot = np.vstack((xBot, yBot)).T # Stack the arrays

    # Stack the coordinates (port[0,0], star[0,0], port[1,1]...) following
    ## pattern of pix coordinates
    dstAll = np.empty([len(xyTop)+len(xyBot), 2]) # Initialize appropriately sized np array
    dstAll[0::2] = xyTop # Add port range coordinates
    dstAll[1::2] = xyBot # Add star range coordinates

    # Filter dst using previously made mask
    dst = dstAll[mask]

    ########################
    # Perform transformation
    # PiecewiseAffineTransform
    # tform = PiecewiseAffineTransform()
    tform = FastPiecewiseAffineTransform()
    tform.estimate(pixAll, dst)

    # First get the min/max values for x,y geospatial coordinates
    x_min, y_min, x_max, y_max = df[['x_min', 'y_min', 'x_max', 'y_max']].values[0]

    # Calculate x,y resolution of a single pixel
    xres = (x_max - x_min) / windowSize_m
    yres = (y_max - y_min) / windowSize_m

    # Calculate transformation matrix by providing geographic coordinates
    ## of upper left corner of the image and the pixel size
    transform = from_origin(x_min - xres/2, y_max - yres/2, xres, yres)

    # Warp image from the input shape to output shape
    out = warp(label.T,
                tform.inverse,
                output_shape=(windowSize_m, windowSize_m),
                mode='constant',
                cval=np.nan,
                clip=False,
                preserve_range=True)

    # Rotate 180 and flip
    # https://stackoverflow.com/questions/47930428/how-to-rotate-an-array-by-%C2%B1-180-in-an-efficient-way
    out = np.flip(np.flip(np.flip(out,1),0),1).astype('uint8')

    # Prepare colors
    class_colormap = {0: '#3366CC',
                        1: '#DC3912',
                        2: '#FF9900',
                        3: '#109618',
                        4: '#990099', 
                        5: '#0099C6',
                        6: '#DD4477',
                        7: '#66AA00',
                        8: '#B82E2E'}
    
    for k, v in class_colormap.items():
        rgb = ImageColor.getcolor(v, 'RGB')
        class_colormap[k] = rgb

    # Prepare output file name
    npz_name = os.path.splitext(os.path.basename(df['npz'].values[0]))[0]
    if outName:
        gtiff = os.path.join(out_dir, f"{outName}_{npz_name}_{windowSize_m[0]}m.tif")
    else:
        gtiff = os.path.join(out_dir, f"{npz_name}_{windowSize_m[0]}m.tif")

    print(gtiff)

    # Export georectified image
    with rio.open(
        gtiff,
        'w',
        driver='GTiff',
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype=out.dtype,
        crs=epsg,
        transform=transform,
        compress='lzw'
        ) as dst:
            dst.nodata=0
            dst.write(out,1)
            dst.write_colormap(1, class_colormap)
            dst=None
    
    print('yep')

#========================================================
def map_npzs(df: pd.DataFrame, in_dir: str, out_dir: str, outName: str, windowSize_m: tuple, epsg: int):

    '''
    '''

    # # Iterate each row
    # # for idx, row in tqdm(df.iterrows(), total=len(df), desc="Mapping npz files"):

    # r = Parallel(n_jobs=-1, verbose=10)(
    #     delayed(label_array_to_shapefile)(df, in_dir, out_dir, outName, windowSize_m, epsg)
    #     for idx, win in tqdm(df.iterrows(), total=len(df), desc="Processing windows")
    #     )
    
    # # Concatenate results
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    # if len(r) == 0:
    #     print("No valid polygons found in any npz files.")
    #     return None
    # r = pd.concat(r, ignore_index=True)

    # if outName:
    #     out_shp = os.path.join(out_dir, f"{outName}.shp")
    # else:
    #     out_shp = os.path.join(out_dir, 'mapped_polygons.shp')

    # r.to_file(out_shp, driver='ESRI Shapefile')

    r = Parallel(n_jobs=-1, verbose=10)(
        delayed(label_array_to_raster)(df, in_dir, out_dir, outName, windowSize_m, epsg)
        for idx, win in tqdm(df.iterrows(), total=len(df), desc="Processing windows")
        )
    

# =========================================================
class FastPiecewiseAffineTransform(PiecewiseAffineTransform):
    def __call__(self, coords):
        coords = np.asarray(coords)

        simplex = self._tesselation.find_simplex(coords)

        affines = np.array(
            [self.affines[i].params for i in range(len(self._tesselation.simplices))]
        )[simplex]

        pts = np.c_[coords, np.ones((coords.shape[0], 1))]

        result = np.einsum("ij,ikj->ik", pts, affines)
        result[simplex == -1, :] = -1

        return result        














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

