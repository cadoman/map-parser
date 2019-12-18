import matplotlib.pyplot as plt
from map_extractor.utils import check_rgb_image
from map_extractor.PolygonGroup import PolygonGroup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.color, skimage.filters, skimage.morphology, skimage.measure
from shapely.geometry import Polygon
import cv2
from tqdm import tqdm

def disp_polygons(img, polygons, saveTo='') :
    todisp = img.copy()
    for polygon in polygons :
        (minx, miny, maxx, maxy) = np.array(polygon.bounds).astype(int)
        cv2.rectangle(todisp, (minx, miny), (maxx, maxy), (0, 0, int(np.max(todisp))), 2)
    plt.imshow(todisp)
    plt.show()

def print_colors(extracted_colors):
    toplot = extracted_colors[:25]
    percentage = 100 * len(toplot)/len(extracted_colors)
    plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.bar(np.arange(len(toplot)) , toplot['count'].values , color=toplot['color'].values)
    plt.title('Major colours found in map ('+str(percentage)+'% of colours represented)')
    plt.show()

def display_colours(color_list) :
    size = len(color_list)
    plt.bar(range(size), [1]*size  , color=color_list)
    plt.show()

def extract_shapes(image:np.ndarray) :
    '''
    Extract the land shapes visible on the map (previously clustered)
    Considering the sea is the most visible color

    Parameters
    image (np.ndarray) : the clustered map from which to extract land shapes

    Returns
    PolygonGroup[] : The extracted land masses, regrouped by color
    '''
    print('Counting colors ...')
    major_colors = get_major_colors(image)
    land_colors = remove_sea_colors(major_colors)
    print('Extracting masks...')

    masks = get_masks(image, land_colors[:])
    print('len masks', len(masks))

    print('Detecting polygons')
    polygon_groups = []
    for mask in masks :
        polygons = get_polygons(mask).filter_polygons(60)
        polygon_groups.append(polygons)

    
    res = []
    print('Remove contained elements of polygon groups')
    for i, pg_group in enumerate(polygon_groups):

        other_pg_groups = np.concatenate((polygon_groups[:i] , polygon_groups[i+1:]))
        for other_pg_group in other_pg_groups :
            pg_group = pg_group.without_polygons_contained_in_other_group(other_pg_group)
        res.append(pg_group)
        
    return res

def get_polygons(mask):
    '''
    Computes the polygons on the mask using skimage.measure.find_contours method

    Parameters
    mask (np.ndarray) : The boolean image from which to extract the countours

    Returns :
    PolygonGroup : A set of detected polygons
    '''
    contours = skimage.measure.find_contours(mask, 0.8)
    res = PolygonGroup([Polygon(list(zip(contour[:, 1], contour[:, 0]))) for contour in contours])
    return res

def get_major_colors(image : np.ndarray) :
    '''
    Extracts the major colors in the image in a sorted dataframe

    Parameters
    image (np.ndarray) : The image to analyze

    Returns
    pd.DataFrame ('count', 'value') : A dataframe with the value of the color and the number of times it appears in the image
    '''
    unique, counts = np.unique(image.reshape(int(image.size/3), 3), axis=0, return_counts=True)
    scores = list(zip(counts, unique))
    df = pd.DataFrame(scores, columns=['count', 'color']).sort_values(by=['count'], ascending=False)
    return df

def get_masks(image : np.ndarray, colors: pd.DataFrame):
    '''
    Extract a list of mask (binary image), 1 mask for each color mentioned in colors
    Performs a noise reduction before returning the masks
    Filters out scattered masks

    Parameters
    image (np.ndarray) : The rgb image from which to extract the masks
    colors (pd.DataFrame) : The dataframe that represents the colors in the image

    Returns
    np.ndarray (bool array) [] : A list of masks (for each mask, 1 if image matches color, 0 otherwise) 
    '''
    check_rgb_image(image)
    #Passing image to grayscale (only to make color comparison simpler)
    grayscale_img = skimage.color.rgb2gray(image)
    grayscale_colors = skimage.color.rgb2gray(
        # reshape the colors to fit rgb2gray methods
        np.reshape(np.concatenate(colors.color.values), (1, len(colors), 3))
    ).flatten()
    noisy_masks = [grayscale_img==color for color in grayscale_colors]
    # Denoise masks
    denoised_masks = [skimage.filters.median(mask, selem=np.ones((3, 3))) for mask in noisy_masks]
    
    # Filter out scattered masks
    non_scattered_masks = remove_scattered_masks(denoised_masks)
    return non_scattered_masks

def remove_scattered_masks(masks:list):
    '''
    Remove the scattered masks by computing their eroded version and checking these are not empty
    On an eroded version scattered points are invisible, but big land masses remain
    This does not return eroded versions

    Parameters
    masks (np.ndarray[]) : A list of binary images

    Returns
    np.ndarray[] : masks filtered from its scattered masks
    '''
    get_eroded = lambda mask : skimage.morphology.erosion(mask, skimage.morphology.disk(6))
    not_scattered = lambda mask : np.count_nonzero(get_eroded(mask)) > 0
    return [mask for mask in masks if not_scattered(mask)]

def colour_distance(rgb1, rgb2):
    [[lab1, lab2]] = skimage.color.rgb2lab([[rgb1, rgb2]])
    return np.linalg.norm(lab1 - lab2)


def remove_sea_colors(colordf:pd.DataFrame):
    '''
    Removes the first color from the provided dataframe, and the colors that are similar with this first color
    Uses LAB representation to determine color similarity 

    Parameters
    colordf (pd.DataFrame) : The dataframe which contains all colors found on the map, ordered by count descending

    Returns
    pd.DataFrame : The dataframe, stripped from the first color and  
    '''
    # Assuming the sea is the most represented color
    sea_color = colordf.iloc[0]['color']
    computed_distance = colordf.assign(distance_sea=colordf.color.apply(lambda color: colour_distance(sea_color, color)))
    return colordf[computed_distance.distance_sea > 10]
