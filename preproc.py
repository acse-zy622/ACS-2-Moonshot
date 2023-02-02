'''For preprocessing images for ML input'''

import numpy as np
import xarray as xr
import pygmt, os, argparse
import pandas as pd
from PIL import Image
from numpy import asarray
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import product
from tqdm import tqdm

def transcoor(x, y, arrlenx, arrleny, 
              stlon, enlon, stlat, enlat,
              r=None, diam=3474.8):
    '''
    For transforming x/y pixel indices to real Long/Lat 
    values.
    
    Parameters
    ---------------
    x: int, float, or np.array
        pixel position in array along horizontal x-axis
    
    y: int, float, or np.array
        pixel position in array along vertical y-axis
        
    arrlenx: int
        image array size in x axis
        
    arrleny: int
        image array size in y axis
        
    stlon: int or float
        longitude of top-left image pixel
        
    enlon: int or float
        longitude of top-right image pixel
        
    stlat: int or float
        latitude of top-left image pixel
        
    enlat: int or float
        latitude of bottom-left image pixel
        
    r: int, float, np.array or None
        radius of crater in number of pixels
        
    diam: int or float
        Diameter of planetary body in km
        
    Returns
    ---------------
    lon: float, or np.array
        Transformed Longitude
        
    lat: float, or np.array
        Transformed Latitude
        
    r: None, float, or np.array
        Transformed radius in km
    '''
    lon = x*(enlon-stlon)/arrlenx + stlon
    pixelStart = np.log(np.tan(np.pi/4 + (stlat/2) * np.pi/180)) / (2*np.pi)  
    pixelEnd = np.log(np.tan(np.pi/4 + (enlat/2) * np.pi/180)) / (2*np.pi)
    lat = (np.arctan(np.exp( ((y*(pixelEnd-pixelStart)/arrleny) + pixelStart) * (2*np.pi) )) - np.pi/4) * 360/np.pi
    if r is not None:
        r = abs(r*(enlat-stlat)/360*np.pi*3474.8)
    return lon, lat, r


def revtranscoor(lon, lat, arrlenx, arrleny, 
                 stlon, enlon, stlat, enlat, 
                 r=None, diam=3474.8):
    '''
    For transforming real Long/Lat values to x/y pixel indices.
    
    Parameters
    ---------------
    lon: int, float, or np.array
        Longitude
    
    lat: int, float, or np.array
        Latitude
        
    arrlenx: int
        image array size in x axis
        
    arrleny: int
        image array size in y axis
        
    stlon: int or float
        longitude of top-left image pixel
        
    enlon: int or float
        longitude of top-right image pixel
        
    stlat: int or float
        latitude of top-left image pixel
        
    enlat: int or float
        latitude of bottom-left image pixel
        
    r: int, float, np.array or None
        radius of crater in km
        
    diam: int or float
        Diameter of planetary body in km
        
    Returns
    ---------------
    x: float, or np.array
        pixel position in array along horizontal x-axis
        
    y: float, or np.array
        pixel position in array along vertical y-axis
        
    r: None, float, or np.array
        Transformed radius in number of pixels
    '''
    x = (lon - stlon)*arrlenx/(enlon-stlon)
    pixelStart = np.log(np.tan(np.pi/4 + (stlat/2) * np.pi/180)) / (2*np.pi)  
    pixelEnd = np.log(np.tan(np.pi/4 + (enlat/2) * np.pi/180)) / (2*np.pi)
    y = (np.log(np.tan(lat*np.pi/360 + np.pi/4))/(2*np.pi) - pixelStart)*arrleny/(pixelEnd-pixelStart)
    if r is not None:
        r = abs((r*360/np.pi/diam)/(enlat-stlat))
    return x, y, r


def reproject(imgpath, size, stlon, enlon, stlat, enlat):
    '''
    Reproject Equirectangular projected images 
    into Mercator projection.
    
    Parameters
    -------------
    imgpath: str
        Relative path to B/W image
        
    size: float
        Resize percentage

    stlon: int or float
        longitude of top-left image pixel
        
    enlon: int or float
        longitude of top-right image pixel
        
    stlat: int or float
        latitude of top-left image pixel
        
    enlat: int or float
        latitude of bottom-left image pixel
        
    Returns
    -------------
    np.array
    B/W image in Mercator projection    
    '''
    imgarr = asarray(Image.open(imgpath))
    fig = pygmt.Figure()
    fig.grdimage(imgpath, region=f"{stlon}/{enlon}/{enlat}/{stlat}", 
                 projection=f"M{size*imgarr.shape[1]*0.0264583}c")
    fig.savefig("temp.png")
    newimg = Image.open("temp.png")
    newimgarr = asarray(newimg)
    return (np.sum(newimgarr, axis=2)/3).astype(np.uint8)


def splitarr(arr, nrows, ncols, stride, stlon, enlon, stlat, enlat):
    '''
    Split image array into smaller arrays.
    
    Parameters
    -------------
    arr: np.array
        Input image array
        
    nrows: int
        Size of window in y axis
        
    ncols: int
        Size of window in x axis
        
    stride: int
        stride

    stlon: int or float
        longitude of top-left image pixel
        
    enlon: int or float
        longitude of top-right image pixel
        
    stlat: int or float
        latitude of top-left image pixel
        
    enlat: int or float
        latitude of bottom-left image pixel
        
    Returns
    ------------
    altarr: np.array
        Collection of smaller arrays 
        from input image array
        
    boundaries: set of np.arrays
    '''
    altarr = stride_tricks.sliding_window_view(
        arr, window_shape=(nrows,ncols)
    )[::stride,::stride].reshape(-1,nrows,ncols)
    
    xbound = stride_tricks.sliding_window_view(
        np.arange(arr.shape[1]), window_shape = ncols
    )[::stride].reshape(
        -1,ncols
    )[:,::ncols-1].reshape(-1)
    
    ybound = stride_tricks.sliding_window_view(
        np.arange(arr.shape[0]), window_shape = nrows
    )[::stride].reshape(
        -1,nrows
    )[:,::nrows-1].reshape(-1)
    
    boundaries = transcoor(xbound, ybound, arr.shape[1], arr.shape[0], stlon, enlon, stlat, enlat)
    
    return altarr, boundaries[:2]


def preproc(data, altarr, dimens, label, diam=3474.8, outpath="data/train/"):
    '''
    Preprocess training data into csv files.
    
    Parameters
    --------------
    data: pd.DataFrame
        Training data
        
    altarr: np.array
        collection of image arrays
        
    dimens: set of np.arrays
        Longitude and Latitude boundaries
        
    diam: int or float
        Diameter of planetary body in km
        
    outpath: str
        write directory
        
    Returns
    -------------
    None
    '''
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    dimenlist = list(product(dimens[1].reshape(-1,2), dimens[0].reshape(-1,2)))

    for idx, ((lat1, lat2), (lon1, lon2)) in tqdm(enumerate(dimenlist)):

        train = data[(data.LAT_CIRC_IMG < lat1) & (data.LAT_CIRC_IMG >= lat2)
                 & (data.LON_CIRC_IMG > lon1) & (data.LON_CIRC_IMG <= lon2)
                ][['LAT_CIRC_IMG','LON_CIRC_IMG','DIAM_CIRC_IMG']]

        train.LON_CIRC_IMG, train.LAT_CIRC_IMG, train.DIAM_CIRC_IMG = revtranscoor(
            train.LON_CIRC_IMG, train.LAT_CIRC_IMG, altarr[idx].shape[1], 
            altarr[idx].shape[0], lon1, lon2, lat1, lat2, train.DIAM_CIRC_IMG)
        train.LON_CIRC_IMG = train.LON_CIRC_IMG.apply(lambda x: abs(x/altarr[idx].shape[1]))
        train.LAT_CIRC_IMG = train.LAT_CIRC_IMG.apply(lambda x: abs(x/altarr[idx].shape[0]))

        train['Class'] = 0
        train['DIAM_CIRC_IMG2'] = train.DIAM_CIRC_IMG
        train = train.reindex(['Class', 'LAT_CIRC_IMG', 'LON_CIRC_IMG', 'DIAM_CIRC_IMG', 'DIAM_CIRC_IMG2'], 
                              axis=1)

        with open(outpath + f'/{label}_{str(idx).zfill(4)}.jpg', 'w') as outfile:
            im = Image.fromarray(altarr[idx])
            im.save(outfile)

        with open(outpath + f'/{label}_{str(idx).zfill(4)}.csv', 'w') as outfile:
            train.set_axis(
                [idx, lat1, lat2, lon1, lon2], axis=1
            ).to_csv(outfile, index=False, sep=' ')

        with open(outpath + f'/{label}_{str(idx).zfill(4)}_filtered.csv', 'w') as outfile:
            train[train.DIAM_CIRC_IMG2 > 0.05].set_axis(
                [idx, lat1, lat2, lon1, lon2], axis=1
            ).to_csv(outfile, index=False, sep=' ')          
            
    return None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type = float, 
                        help = 'resize percentage')
    parser.add_argument('-lon1', '--longitude1', type = float, 
                        help = 'longitude of top-left image pixel')
    parser.add_argument('-lon2', '--longitude2', type = float, 
                        help = 'longitude of top-right image pixel')
    parser.add_argument('-lat1', '--latitude1', type = float, 
                        help = 'latitude of top-left image pixel')
    parser.add_argument('-lat2', '--latitude2', type = float, 
                        help = 'latitude of bottom-left image pixel')
    parser.add_argument('-img', '--image', type = str, 
                        help = 'read image filepath')
    parser.add_argument('-f', '--file', type = str, 
                        help = 'read CSV filepath')
    parser.add_argument('-o', '--out', type = str, 
                        help = 'write directory')
    args = parser.parse_args()
    
    Image.MAX_IMAGE_PIXELS = None

    newarr = reproject(args.image, args.size, args.longitude1, args.longitude2, args.latitude1, args.latitude2)
    altarr, dimens = splitarr(newarr, 416, 416, 350, args.longitude1, args.longitude2, args.latitude1, args.latitude2)
    
    if args.file is None:
        args.file = 'data/Moon_WAC_Training/Moon_WAC_Training/labels/lunar_crater_database_robbins_train.csv'
        
    data = pd.read_csv(args.file)
    preproc(data, altarr, dimens, args.out, outpath=args.out)