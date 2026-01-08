

'''
Copyright (c) 2025- Cameron S. Bodine

Compatible only with PINGMapper outputs.
'''

#########
# Imports

import os, sys
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import cv2
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

from pingtile.utils import addZero

###########
# Functions

def resize_to_pingMax(img: np.ndarray, 
                      cropMax: int) -> np.ndarray:
    ndims = img.ndim
    current_size = img.shape[0]
    if current_size < cropMax:
        # Pad with zeros 
        if ndims == 2:
            padding = ((0, cropMax - current_size), (0, 0))
        else:
            padding = ((0, cropMax - current_size), (0, 0), (0,0))
        resized_img = np.pad(img, padding, mode='constant', constant_values=0)
    elif current_size > cropMax:
        # Truncate the array
        resized_img = img[:cropMax, :]
    else:
        # No change needed
        resized_img = img
    return resized_img

#=======================================================================
def exportMovWin(inDir: str,
                 i: int,
                 nchunk: int,
                 lastChunk: int,                 
                 stride: int,
                 tileType: list,
                 pingMax: int,
                 depMax: int,):
    
    '''
    Export moving window tile.
    '''

    # Set chunk index
    a_idx = i
    b_idx = i+1

    outImgs = []

    # Iterate each tile type
    for t in tileType:
        if t == 'wco':
            cropMax = depMax
        else:
            cropMax = pingMax
        inDir_t = os.path.join(inDir, t)
        outDir = os.path.join(inDir, t+'_mw')

        if not os.path.exists(outDir):
            try:
                os.mkdir(outDir)
            except:
                pass

        # Find the images
        images = os.listdir(inDir_t)
        images.sort()

        # Get each image
        a_img = images[a_idx]

        if b_idx <= lastChunk:
            b_img = images[b_idx]
        else:
            b_img = a_img # Dummy

        # Get image name
        img_name = a_img.split('.')[0]

        # Open each image
        a_img = imread(os.path.join(inDir_t, a_img))
        if b_idx < lastChunk:
            b_img = imread(os.path.join(inDir_t, b_img))
        else:
            b_img = np.zeros_like(a_img)

        # Resize a_img and b_img
        a_img = resize_to_pingMax(a_img, cropMax)
        b_img = resize_to_pingMax(b_img, cropMax)

        # Set stride based on first image
        # stride = int(round(a_img.shape[1] * stride, 0))
        if stride == 0:
            to_stride = 1
        else:
            to_stride = stride

        # Set window size based on first image
        # winSize = a_img.shape[1]
        winSize = nchunk

        # Concatenate images
        movWin = np.concatenate((a_img, b_img), axis=1)

        # Last window idx
        lastWinIDX = a_img.shape[1] # nchunk

        win = 0
        # Iterate each window
        while win < lastWinIDX:
            window = movWin[:, win:win+winSize]

            zero = addZero(win)

            # Save window
            outFile = os.path.join(outDir, img_name+'_'+zero+str(win)+'.jpg')
            imsave(outFile, window)

            outImgs.append(outFile)
            win += to_stride

    return outImgs

#=======================================================================
def doSonogramMovWin(inDir: str,
                     projName: str,
                     channel: str,
                     sonMetaFile: str,
                     nchunk: int,
                     stride: int,
                     tileType: list,
                     exportVid: bool=False,
                     threadCnt: int=4
                     ):
    '''
    Generate moving window tiles from input sonar sonogram images.
    '''

    # Open sonogram dataframe
    sDF = pd.read_csv(sonMetaFile)

    # Iterate each transect
    for name, group in sDF.groupby('transect'):

        # Crop all images to most common range
        
        rangeCnt = np.unique(group['ping_cnt'], return_counts=True)
        pingMaxi = np.argmax(rangeCnt[1])
        pingMax = int(rangeCnt[0][pingMaxi])

        depCnt = np.unique(group['dep_m'], return_counts=True)
        depMaxi = np.argmax(depCnt[1])
        depMax = int(depCnt[0][depMaxi]/group['pixM'].iloc[0])
        depMax += 50

        # Get chunks
        chunks = group['chunk_id'].unique().tolist()

        # Get last chunk index
        chunks.sort()
        lastChunk = chunks[-1]

        imgs = Parallel(n_jobs=int(np.min([len(chunks), threadCnt])))(delayed(exportMovWin)(inDir=inDir, nchunk=nchunk, i=i, lastChunk=lastChunk, stride=stride, tileType=tileType, pingMax=pingMax, depMax=depMax) for i in tqdm(chunks))

        # Flatten list
        imgs = [item for sublist in imgs for item in sublist]
        imgs.sort()

        # Export video
        if exportVid:
            for t in tileType:
                # Get input dir from an image
                inDir_t = os.path.dirname(imgs[0])
                outDir = inDir_t+'_results'

                if not os.path.exists(outDir):
                    os.makedirs(outDir)
                vid_path = os.path.join(outDir, f'{projName}_{channel}_{t}_{name}_movWin.mp4')

                # Determine width and height from first image
                first_image = imread(imgs[0])
                height, width = first_image.shape[:2]

                video = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 8, (width, height), )
                for image in imgs:
                    frame = cv2.imread(os.path.join(outDir, image))
                    video.write(frame)

# Main do work
if __name__ == '__main__':
    doSonogramMovWin(inDir=sys.argv[1],
                     projName=sys.argv[1].split('/')[-3],
                     channel=sys.argv[1].split('/')[-2],
                     sonMetaFile=sys.argv[2],
                     nchunk=int(sys.argv[3]),
                     stride=int(sys.argv[4]),
                     tileType=sys.argv[5].split(','),
                     exportVid=bool(int(sys.argv[6])),
                     threadCnt=int(sys.argv[7])
    )