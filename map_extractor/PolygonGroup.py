from shapely.geometry import Polygon
from pprint import pprint
import numpy as np
import shapely.wkt
import matplotlib.pyplot as plt
import json


class PolygonGroup:
    def __init__(self, polygons: list, name=""):
        assert isinstance(polygons, list)
        for pol in polygons:
            assert isinstance(pol, Polygon)
        self.polygon_list = polygons
        self.name = name

    def __iter__(self):
        return iter(self.polygon_list)

    def __len__(self):
        return len(self.polygon_list)

    def filter_polygons(self, min_area: int):
        '''
        Filter out the polygons which are too small and those which are contained in other polygons

        Parameter
        min_area (int) : The minimum area for a polygon to be considered
        '''
        big_polygons = [
            pol for pol in self.polygon_list if pol.area >= min_area]

        not_contained = []
        for i, pol in enumerate(big_polygons):
            if not is_contained(pol, np.concatenate((big_polygons[:i], big_polygons[i+1:]))):
                not_contained.append(pol)
        return PolygonGroup(not_contained, self.name)

    def to_dict(self):
        polygon_json = [list(pol.exterior.coords) for pol in self.polygon_list]
        return{
            "name": self.name,
            "polygon_list": polygon_json
        }

    @staticmethod
    def from_dict(dict_pg: dict):
        polygons = [Polygon(points_array)
                            for points_array in dict_pg['polygon_list']]
        return PolygonGroup(polygons, dict_pg['name'])

    def without_polygons_contained_in_other_group(self, other_group):
        '''
        Filter out the polygon of the polygongroup which are contained in the polygons of another polygon group

        Parameters
        other_group (PolygonGroup) : The group representing potential containers

        Returns
        PolygonGroup : A new group created from self, stripped from contained polygons
        '''
        not_contained = [pol for pol in self.polygon_list if not is_contained(
            pol, other_group.polygon_list)]
        return PolygonGroup(not_contained, self.name)

    def display(self):
        points = np.concatenate(
            ([pol.exterior.coords for pol in self.polygon_list])).astype(int)
        maxx, maxy = points.max(axis=0) + 1
        image = np.zeros((maxy, maxx))
        rr, cc = points[:, 1], points[:, 0]
        image[rr, cc] = 1
        plt.imshow(image, cmap='gray')
        plt.title('Name : '+str(self.name))
        plt.show()

def is_contained(polygon: Polygon, other_polygons:list):
    '''
    Determine if a polygon lies in one the polygons provided in other_polygons

    Parameters
    polygon (Polygon) : The potentially contained polygon
    other_polygons (list) : List of potential container polygons

    Returns
    boolean : True if contained, False otherwise
    '''
    for potential_container in other_polygons:
        if polygon.within(potential_container):
            return True
    return False
