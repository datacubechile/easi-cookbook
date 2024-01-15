import json
from decimal import Decimal

import boto3
import geojson
from shapely import geometry
from shapely.ops import unary_union


class DecimalEncoder(json.JSONEncoder):
    """Custom JSONEncoder to parse Decimal values back to Float.
    https://docs.python.org/3/library/json.html#json.JSONEncoder.default

    easi-aoi-mgmt uploads the geojson with geojson.load(f, parse_float=Decimal)
    E.g., see https://stackoverflow.com/a/64808799
    """
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

def validate_geojson(o: dict) -> (bool, object):  # -> geojson.geometry
    """Validate a geojson string or dict. Return a geojson.geometry object"""
    try:
        if isinstance(o, dict):
            o = json.dumps(o)
        boundary = geojson.loads(o)
        if not boundary.is_valid:
            return False, boundary.errors()
    except (TypeError, AttributeError) as e:
        return False, e
    return True, boundary

def get_boundary(names: list, aws_region: str) -> (bool, object):  # -> '__geo_interface__' dict
    """Get the __geo_interface__ boundary (dict) from an easi-aois DynamoDB table.
    A list of AOI_names will return the unary_union of the boundaries.
    """
    def _get_boundary(aoi_name: str) -> 'geojson.geometry.MultiPolygon':
        response = db_table.get_item(Key={'aoi_name': aoi_name})
        if not response.get('Item'):
            return False, f'AOI name not found in database: {aoi_name}'
        aoi_json = json.dumps(response['Item']['aoi_polygon'], cls=DecimalEncoder)
        return validate_geojson(aoi_json)

    dynamodb = boto3.resource('dynamodb', region_name=aws_region)
    try:
        db_table = dynamodb.Table('easi-aois')
        boundaries = []
        for aoi in names:
            s,r = _get_boundary(aoi)
            if not s:
                return s, r
            boundaries.append(geometry.shape(r))  # To shapely
        geom = unary_union(boundaries)
        boundary = geometry.mapping(geom)  # To __geo_interface__ dict
    except Exception as e:
        return False, f'Error retrieving data from AOI database: {e}'
    return True, boundary