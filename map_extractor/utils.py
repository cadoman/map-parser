import numpy as np

def check_rgb_image(image : np.ndarray):
    '''
    Raise an Index error if the provided image is not an rgb image
    '''
    if image.ndim != 3 or image.shape[2] != 3:
        raise IndexError('The image must be RGB with shape (heigt, width, 3). Shape provided : ' +str(image.shape)+ '')
