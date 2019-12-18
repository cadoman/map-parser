import argparse
import matplotlib.pyplot as plt
from shapely import geometry
from map_extractor import preprocessing, shape_extraction, country_naming
import skimage.color
from map_extractor.PolygonGroup import PolygonGroup
import json
import numpy as np

import cv2


def display_img(img, title="", big=False) : 
    if big : 
        plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()

def parse_ignored_boxes(box_string):
    return [geometry.box(tup[0], tup[1], tup[2], tup[3]) for tup in eval(box_string)]

def get_args() :
    parser = argparse.ArgumentParser(description='Parse a map to labelled svgs')
    parser.add_argument("--input", type=str, dest="input_file",
                        help="Path of the map to parse", required=True)
    parser.add_argument("--ignore", type=str, dest="ignored_boxes", help='A list of comma separated rectangle (minx, miny, maxx, maxy), ex : "(20, 30, 200, 400) , (0, 0, 100, 100)"')
    parser.add_argument(
        "--colors", 
        type=int, 
        dest="nb_colors", 
        help="The estimated number of color on the map (usually works better by over-estimating the number of colors)",
        required=True
        )
    return parser.parse_args()

def main() :
    args = get_args()
    print('Loading image...')
    image = skimage.color.rgba2rgb(plt.imread(args.input_file))
    if args.ignored_boxes :
        print('Removing ignored areas...')
        image = preprocessing.remove_ignored_areas(image, parse_ignored_boxes(args.ignored_boxes))
    print('Clustering image colors..')
    image_clustered = preprocessing.regroup_image_colors(image, args.nb_colors)
    # image_clustered = skimage.color.rgba2rgb(plt.imread('notebooks/tmp/clustered_europe.png'))
    
    polygon_groups = shape_extraction.extract_shapes(image_clustered)
    
    # for i, shape in enumerate(shapes) :
    #     with open('/tmp/shape_'+str(i)+'.json', 'w') as f :
    #         dict_shape = shape.to_dict()
    #         json_rep = json.dumps(dict_shape) 
    #         f.write(json_rep)

    # for i in range(16) : 
    #     with open('/tmp/shape_'+str(i)+'.json', 'r') as f :
    #         jsonrep = f.read()
    #         a = PolygonGroup.from_dict(json.loads(jsonrep))
    #         polygon_groups.append(a)
    # img = skimage.color.rgba2rgb(plt.imread('notebooks/tmp/europe_cleaned.png'))
    
    pg_group_named = []
    print('Performing OCR with Tesseract ...')
    ocr_results = country_naming.apply_tesseract(image)
    print('Extracting OCR results..')
    for group in polygon_groups:
        pg_group_named = np.concatenate((pg_group_named, country_naming.label_polygon_group(ocr_results, group)))
    print('Done')

def disp_ocr_results(img, ocr_results) :
    todisp = img.copy()
    disp_color = (int(np.max(img)) , 0, 0)
    for i, row in ocr_results.iterrows():
        (minx, miny, maxx, maxy) = np.array(row['bbox'].bounds).astype(int)
        cv2.rectangle(todisp, (minx, miny), (maxx, maxy),disp_color , 2)
        cv2.putText(todisp,row['text'], (maxx+3,maxy), cv2.FONT_HERSHEY_SIMPLEX, 1, disp_color, 2)
    display_img(todisp, '', True)
    
if __name__ == "__main__" :
    main()

