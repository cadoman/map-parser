import matplotlib.pyplot as plt
import skimage.color
import time
import pytesseract
import pandas as pd
from shapely.geometry import box
import os
from map_extractor.utils import check_rgb_image
from map_extractor.PolygonGroup import PolygonGroup
import numpy as np
from shapely.geometry import Polygon

def apply_tesseract(img) :
    '''
    Recognize countries' names on a map with Tesseract. 

    Parameters
    img (np.ndarray) : an RGB image of the map, with countries' names appearing

    Returns
    pd.DataFrame : The OCR results, a dataframe with columns 'text' and 'bbox' (text's bounding box)
    '''
    check_rgb_image(img)
    tmp_filename = str(time.time()) + '.png'
    # Stores the images as black and white to perform analysis on both
    plt.imsave(tmp_filename, skimage.color.rgb2gray(img), cmap='gray')
    d = pytesseract.image_to_data(tmp_filename, config="--oem 3 --psm 11",lang='eng', output_type=pytesseract.Output.DICT)
    os.remove(tmp_filename)
    
    #Now text and bounding boxes are extracted
    boxes = []
    bboxes = pd.DataFrame(columns=['bbox'])
    # Filtering unisgnificant results from text recognition (ex : @àé won't pass the filter  )
    df = filter_recognition_results(pd.DataFrame(d))
    #Building shapely bounding boxes
    for i, row in df.iterrows():
        (x, y, w, h) = (row['left'], row['top'], row['width'], row['height'])
        boxes.append(box(x, y, x+w, y+h))
        bboxes.loc[i] =  [box(x, y, x+w, y+h)]
    # Only returns text and shapely bbox
    return bboxes.join(df.text).reset_index()


def label_polygon_group(ocr_results : pd.DataFrame, pg_group : PolygonGroup) :
    '''
    Given text data and text boxes (in ocr results), gives a name to the polygon group in parameter

    Parameters
    ocr_results (pd.DataFrame) : The dataframe which contains columns 'text' and 'bbox', from the ocr performed on the map
    pg_group (PolygonGroup) : A group of polygon 

    Returns
    PolygonGroup[] : A list of Polygon Groups (named if possible). Split the polygons if different names are found (hence the list result type)
    '''
    found_names = []

    for land_shape in pg_group :
        candidates = get_text_lying_in(ocr_results, land_shape)
        found_names.append(select_biggest_font(candidates))
    # The default name of unlabelled countries will be the biggest in found names
    max_size = max([match['size'] if match else 0 for match in found_names])
    if max_size== 0:
        # When no name was found
        pg_group.name = None
        return [pg_group]
    
    dominant_name = (next((found_name['text'] for found_name in found_names if found_name and found_name['size']==max_size)))
    available_names = set([row['text']  for row in found_names if row])
    polygons_named = {key: [] for key in available_names}

    for i , found_name in enumerate(found_names):
        if found_name:
            polygons_named[found_name['text']].append(pg_group.polygon_list[i])
        else :
            polygons_named[dominant_name].append(pg_group.polygon_list[i])
    return [PolygonGroup(polygons_named[name], name) for name in polygons_named]
    
def select_biggest_font(candidates : pd.DataFrame):
    '''
    In the name candidates, select the one which has the biggest font size (i.e. max height)

    Parameters
    candidates (pd.DataFrame) : The list of possible country names

    Returns
    dict : {"text" : str , "size" : int}
    '''
    if(len(candidates)):
        candidates = candidates.assign(font_size=candidates.bbox.apply(lambda bbox : list(bbox.bounds)[3] - list(bbox.bounds)[1]))
        best_match =  candidates.loc[candidates['font_size'].idxmax()]
        return {'text' : best_match['text'], 'size' : best_match['font_size']}
    else:
        return None

def get_text_lying_in(text_data : pd.DataFrame, polygon:Polygon):
    '''
    From an ocr results dataframe, returns the rows if their text are contained in the polygon

    Parameters
    text_data (pd.DataFrame) : The OCR results dataframe
    polygon (Polygon) : The polygon that may contains text

    Returns
    pd.DataFrame : The rows of text_data if text lies in the polygon
    '''
    row_lying_bool_array = [polygon.contains(row['bbox']) for _, row in text_data.iterrows()]
    return text_data[row_lying_bool_array]



def filter_recognition_results(df):
    significantLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    authorizedLetters = significantLetters + "'-. ()"
    
    #At least 3 significant letters
    isLongEnough = lambda x : len([char for char in x if char in significantLetters]) >= 3
    noForbidden = lambda x : ''.join([char for char in x if char in authorizedLetters]) == x
    # Removing empty
    validity_array = [isLongEnough(row['text']) and noForbidden(row['text']) for _, row in df.iterrows()]
    return df[validity_array]


