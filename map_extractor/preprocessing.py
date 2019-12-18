from shapely import geometry
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import skimage.color
from map_extractor.utils import check_rgb_image

def remove_ignored_areas(image : np.ndarray, ignored_areas : geometry.Polygon):
    '''
    Fill the specified area on the image by their surrounding colors
    
    Parameters:
    image (ndarray): The image from which the color should be removed
    ignored_areas (geometry.Polygon[]) : A list of rectangles representing the areas to fill
    
    Returns:
    ndarray: The image with areas filled

    '''
    check_rgb_image(image)
    res = image.copy()
    for polygon in ignored_areas :
        [_, (xb, yb), _, (xa, ya), _] = [(int(x), int(y)) for (x, y) in polygon.exterior.coords]
        res[ya:yb, xa:xb] = __get_surrounding_color(image, polygon)
    return res

def regroup_image_colors(image : np.ndarray, nb_colors : int):
    check_rgb_image(image)
    (h, w) = image.shape[:2]
    # Lab is a colour representation (l, a , b). 
    # The kmeans computes an euclidian distance, lab is made so that if the perceived colour differ, the euclidian distance differs more than an rgb euclidian distance
    data = skimage.color.rgb2lab(image)
    
    #With location added, kmeans
    clt = MiniBatchKMeans(n_clusters = nb_colors, random_state = 0)
    labels = clt.fit_predict(data.reshape((h*w, 3)))
    res = clt.cluster_centers_[labels].reshape((h, w, 3))
    res = skimage.color.lab2rgb(res)
    return res



def __get_surrounding_color(img: np.ndarray, rectangle : geometry.Polygon):
    '''
    Extracts the dominant color of the border of the rectangle on the image

    Parameters:
    img (ndarray) : The image from which the color should be extracted
    rectangle (geometry.Polygon) : The rectangle that has the surrounding colors you're interested in

    Returns :
    ndarray or number : The median color on the surroundings
    '''
    offset = 1
    x_left = int(np.array(rectangle.exterior.coords)[:, 0].min()) - offset
    x_right = int(np.array(rectangle.exterior.coords)[:, 0].max()) + offset
    y_top = int(np.array(rectangle.exterior.coords)[:, 1].min()) - offset
    y_bottom = int(np.array(rectangle.exterior.coords)[:, 1].max()) + offset
    corners =np.concatenate((
        img[y_top, x_left:x_right],
        img[y_bottom, x_left:x_right],
        img[y_top:y_bottom, x_left],
        img[y_top:y_bottom, x_right],
    ))
    return np.median(corners, axis=0)
