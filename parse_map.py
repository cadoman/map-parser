import argparse
import matplotlib.pyplot as plt
from shapely import geometry
from map_extractor import preprocessing, shape_extraction
import skimage.color

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
    shapes = shape_extraction.extract_shapes(image_clustered)
    for shape in shapes :
        shape.display()
if __name__ == "__main__" :
    main()

