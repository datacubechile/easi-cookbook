import boto3
import logging
import pytest
from  tasks.gridded.tile_processor import TileProcessor
from  tasks.gridded.tile_generator import TileGenerator

@pytest.fixture(autouse=True)
def tile_generator():
    logging.basicConfig(level=logging.DEBUG)

    # Input params for local test
    # TODO Should really have a range of fixtures to trial
    # TODO @pytest.parameterize is useful for testing different input/result combinations on a single test.

    input_params = [
        {
            "name": "product",
            "value": "landsat8_c2l2_sr",
        },
        {
            "name": "odc_query",
            "value": '{ "output_crs": "epsg:3577", "resolution": [30, 30], '
            '"group_by": "solar_day" }',
        },
        {
            "name": "roi",
            "value": '{"time_start": "2022-01-01", "time_end": "2022-02-28", '
            '"boundary": { "type": "Polygon", "crs": { "type": "name", "properties": '
            '{ "name": "EPSG:4326" } }, "coordinates": [ [ [ 149.21, '
            "-35.21 ], [ 148.98,-35.21 ], "
            "[ 148.98,-35.39 ], [ 149.21, "
            "-35.39 ], [ 149.21, -35.21 ] ] ] } }",
        },
        {
            "name": "size",
            "value": "61440",
        },
        {
            "name": "tiles_per_worker",
            "value": "2",
        },
        {
            "name": "aws_region",
            "value": "",
        },
    ]

    generator = TileGenerator(input_params)
    generator.generate_tiles()
    logging.info("Completed tile generation.")
    return

def test_tile_processor():
    logging.basicConfig(level=logging.INFO)

    client = boto3.client('sts')
    userid = client.get_caller_identity()['UserId']

    # Input params for local test
    input_params = [
        {
            "name": "measurements",
            "value": '["red", "green", "blue", "pixel_qa"]',
        },
        {
            "name": "output",
            # TODO Replace with values suitable for your deployment environment.
            "value": f'{{"bucket": "my-cluster-user-scratch", "prefix": "{userid}/geomedian"}}',
        },
        {
            "name": "key",
            "value": "[[24, -65], [25, -65]]",
        },
    ]
    logging.info("Start tile-process...")
    processor = TileProcessor(input_params)
    processor.process_tile()