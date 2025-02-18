import datetime
import json
import logging
import math
import sys
import os
from typing import Iterable

import boto3
import dateutil
import geojson
import pandas as pd
from pathlib import Path
from botocore.exceptions import ClientError

from .geometry import get_boundary, validate_geojson

logger = logging.getLogger(__name__)

def by_chunk(items: list, chunk_size: int = 1000) -> Iterable:
    """Separate iterable objects by chunks"""
    chunk = []
    for item in items:
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
        chunk.append(item)
    yield chunk


def s3_delete_folder(prefix:str, bucket:str):
    """Delete a folder prefix from S3

    :param prefix: The parent prefix for deletion
    :param bucket: Bucket to delete from
    :return: True if file was deleted, else False
    """

    if "s3" not in locals():
        s3 = boto3.client("s3")
    files_to_delete = []
    try:
        logger.info(f"Deleting files from s3://{bucket}/{prefix}")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if page['KeyCount'] > 0:
                for obj in page['Contents']:
                    files_to_delete.append({"Key": obj['Key']})
            else:
                logger.info(f"No items to delete from s3://{bucket}/{prefix}")
        if len(files_to_delete):
            logger.info(f"Deleting {files_to_delete} item{'s'[:len(files_to_delete)^1]}")
            for items in by_chunk(files_to_delete, 1000):
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": items},
                )
        # ERROR:root:An error occurred (MalformedXML) when calling the DeleteObjects operation: The XML you provided was not well-formed or did not validate against our published schema
    except (ClientError, KeyError) as e:
        logger.error(e)
        return False
    return True

def s3_upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # TODO Deprecation candidate? Newer workflows use aioboto3
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # boto3
    if "s3" not in locals():
        s3 = boto3.client("s3")
    # Upload the file
    try:
        s3.upload_file(file_name, bucket, object_name)  # Config=config
    except (ClientError, FileNotFoundError) as e:
        logger.error(e)
        return False
    return True

def s3_download_folder(prefix:str, bucket:str, path:str):
    """Download a folder prefix from S3

    :param prefix: The parent prefix for download
    :param bucket: Bucket to download from
    :param path: The local folder path to download to
    :return: True if file was downloaded, else False
    """
    # TODO Deprecation candidate? Newer workflows use aioboto3

    # boto3
    if "s3" not in locals():
        s3 = boto3.client("s3")
    # Upload the file
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page['Contents']:
                key = Path(obj['Key'])
                tmp_dir =  Path(path) / '/'.join(str(key.relative_to(prefix)).split('/')[0:-1])
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                s3.download_file(bucket, str(key), tmp_dir / key.name)
    except (ClientError, KeyError) as e:
        logger.error(e)
        return False
    return True

def s3_download_file(key:str, bucket:str, path:str):
    """Upload a file to an S3 bucket

    :param key: The S3 key to download
    :param bucket: Bucket to download from
    :param path: The local path to write to
    :return: True if file was downloaded, else False
    """
    # TODO Deprecation candidate? Newer workflows use aioboto3

    # boto3
    if "s3" not in locals():
        s3 = boto3.client("s3")
    # Upload the file
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            s3.download_file(bucket, key, path + "/" + Path(key).name)
    except (ClientError) as e:
        logger.error(e)
        return False
    return True

def process_order_params(order_params: dict, aws_region: str):
    """
    Process the order_params and return a tuple of the components: datetime, bounding_box, boundary and query
    """
    # TODO This can be refactored and merged with validate order
    # Put bounding_box or boundary into a variable.
    # Make sure that at least one of either a bounding_box or a boundary exists in order_params
    bounding_box = None
    boundary = None

    bbox_exists = "bounding_box" in order_params
    boundary_exists = "boundary" in order_params
    aoi_exists = "aoi_name" in order_params

    if not any([bbox_exists, boundary_exists, aoi_exists]):
        raise RuntimeError(
            "No bounding_box, aoi_name or boundary polygon exists in order_params"
        )

    if sum([bbox_exists, boundary_exists, aoi_exists]) > 1:
        raise RuntimeError(
            "Can only have one of: boundary polygon, bounding_box or aoi_name."
        )

    if bbox_exists:
        bounding_box = order_params["bounding_box"]  # [west, south, east, north]

    if boundary_exists:
        try:
            boundary = geojson.loads(json.dumps(order_params["boundary"]))
            assert boundary.is_valid
        except (TypeError, AttributeError) as e:
            sys.exit(f"{type(e).__name__}: Boundary polygon is not valid geojson")
        except AssertionError as e:
            sys.exit(
                f"{type(e).__name__}: Boundary polygon is geojson but validation failed. The error was: '{boundary.errors()}'"
            )

    if aoi_exists:
        aoi_name = order_params["aoi_name"]
        # Handle aoi_name as list or as string
        if type(aoi_name).__name__ != "list":
            aoi_name = [aoi_name]
        status, boundary = get_boundary(aoi_name, aws_region)
        if not status:
            print(boundary)  # this will be an error string if status is false
            boundary = None

    if all(v is None for v in [boundary, bounding_box]):
        raise RuntimeError(
            "Failed to create a boundary or bounding box. Unable to continue."
        )

    print(f"Order_params: {order_params}")
    datetime = [order_params["time_start"], order_params["time_end"]]

    return datetime, bounding_box, boundary


def validate_json(o: dict) -> (bool, object):
    """Validate a JSON string or dict"""
    try:
        if isinstance(o, dict):
            o = json.dumps(o)
        data = json.loads(o)
    except (TypeError, AttributeError) as e:
        return False, e
    return True, data


def validate_order(o: dict, aws_region: str = None) -> (bool, object):
    """Validate an order_params dict.

    Order_params dict is consistent with data-pipelines; indeed this function could/should be shared.
    Each parameter is optional, in that not all use-cases require all parameters.

    order_params = {
        driver: 'name'                      # Driver name for data-pipelines [Optional]
        product: 's2-l2a',                  # Product name [Optional]
        time_start: '2020-01-01',           # Validated by to_utc_isoformat() [Required]
        time_end: '2020-03-01',             # Validated by to_utc_isoformat() [Required]
        bounding_box: [west, south, east, north],   # Bounding box [Require one of]
        boundary: geojson | __geo_interface__,      # Boundary geometry [Require one of]
        aoi_name: ['chile'],                # List or String of AOI name(s) in a DynamoDB easi-aois table [Require one of]'
        api_params: dict,                   # Dict or JSON string of extra params to pass to the API [Optional]
    }

    Only one of 'bounding_box', 'boundary' or 'aoi_name' may be given.
    'AOI_name' will get the geojson from the easi-aois table and return it as the 'boundary' value.
    - A list of AOI_names will append the geojsons into a multipolygon.
    - 'aws_region' must be given
    """
    # TODO This can be refactored and merged with process order params

    order = {}
    in_keys = set(o.keys())

    # driver, product [Optional]
    for test in ("driver", "product"):
        if test in in_keys:
            order[test] = o[test]

    # time [Required]
    values = []
    for test in ("time_start", "time_end"):
        if test not in in_keys:
            return False, f"This key must be given: {test}"
        values.append(o[test])
    d1, d2 = normalise_dates(*values)
    order["time_start"] = d1
    order["time_end"] = d2

    # space [Require one of]
    test = {"bounding_box", "boundary", "aoi_name"}
    common = in_keys & test
    if len(common) != 1:
        return False, f"Only one of these keys may be given: {test}"
    key = common.pop()
    val = o[key]

    ## space: 'bounding_box'
    if key in ("bounding_box",):
        if isinstance(val, str):
            val = val.split(",")
        if not isinstance(val, (list, tuple)):
            return (
                False,
                f"Expecting a comma-separated string or [west,south,east,north]: {val}",
            )
        if len(val) != 4:
            return False, f"Expecting west,south,east,north coordinates: {val}"
        order["bounding_box"] = [float(x) for x in val]

    ## space: 'boundary'
    elif key in ("boundary",):
        s, r = validate_geojson(val)
        if not s:
            return s, r
        order["boundary"] = r

    ## space: 'aoi_name'
    else:
        if aws_region is None:
            return False, "Must provide aws_region with aoi_name"
        if isinstance(val, str):
            val = [x.strip() for x in val.split("|")]
        s, r = get_boundary(val, aws_region)
        if not s:
            return s, r
        order["boundary"] = r

    # JSON dicts or strings [Optional]
    for test in ("api_params",):
        if test in in_keys:
            s, r = validate_json(o[test])
            if not s:
                return s, r
            order[test] = r

    return True, order


def normalise_dates(start_date: str, end_date: str, as_dt: bool = False) -> (str, str):
    """Ensure that start_date <= end_date and end_date <= today+1"""
    start_date = to_utc_isoformat(start_date, as_dt=True)
    end_date = to_utc_isoformat(end_date, as_dt=True)
    # Switch start and end if needed
    if start_date > end_date:
        temp = start_date
        start_date = end_date
        end_date = temp
    # Max datetime is tomorrow UTC
    tomorrow = today_utc() + datetime.timedelta(days=1)
    if end_date > tomorrow:
        end_date = tomorrow
    return to_utc_isoformat(start_date, as_dt), to_utc_isoformat(end_date, as_dt)

def to_utc_isoformat(dt: str, as_dt: bool = False) -> str:
    """Parse the date-time str (or datetime object) and add or change to UTC timezone.
    Return the result as an ISO string (default) or as a datetime.datetime object.
    Borrowed from pystac_client.item_search.ItemSearch._format_datetime._to_utc_isoformat()
    """
    if dt is None:
        return None
    if not isinstance(dt, (datetime.date, datetime.time, datetime.datetime)):
        dt = dateutil.parser.parse(dt)  # will raise if it fails
    dt = dt.astimezone(datetime.timezone.utc)
    if as_dt:
        return dt
    dt = dt.replace(tzinfo=None)
    return dt.isoformat("T") + "Z"

def today_utc() -> datetime.datetime:
    """Return the datetime for today with UTC timezone"""
    return datetime.datetime.combine(
            datetime.date.today(),
            datetime.time()
        ).replace(tzinfo=datetime.timezone.utc)

def calc_chunk(val, target):
    if target >= val:
        res = val # full chunk if less than target
    elif (val//target == 1) and ((val-target)/target >= 0.5):
        res = target # use target if leftover is at least 50%
    else:
        option1 = math.ceil(val/(val//target))
        option2 = math.floor(val/(val//target))
        
        res = option1 if (val/option1)%1 >= (val/option2)%1 else option2 # calculate best option if needed
    return res

# def get_most_recent_dates(bucket, prefix, dt_format='%Y%m%d'):
#     if "s3" not in locals():
#         s3 = boto3.client("s3")
#     try:
#         if not prefix.endswith("/"):
#             prefix = prefix + "/"
#         objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
#         dates = [datetime.datetime.strptime(p['Prefix'].split("/")[-2],dt_format) for p in objs['CommonPrefixes']]
#         # dates = [p['Prefix'].split("/")[-2] for p in objs['CommonPrefixes']]
#         latest_date = max(dates)
#         dates.remove(latest_date)
#         second_date = max(dates) if len(dates) != 0 else None

#     except (ClientError) as e:
#         logger.error(e)
#         return False, False
#     return latest_date, second_date

def get_prior_date(bucket, prefix, date_key, dt_format='%Y%m%d'):
    prefix = str(prefix)
    if "s3" not in locals():
        s3 = boto3.client("s3")
    try:
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        dates = [datetime.datetime.strptime(p['Prefix'].split("/")[-2],dt_format) for p in objs['CommonPrefixes']]
        latest_date = max(dates)
        if len(dates) > 1:
            dates.remove(latest_date)
            prior_date = max(dates)
        else:
            prior_date = ""

    except (ClientError) as e:
        logger.error(e)
        return False
    return prior_date

# Function to convert timestamps to datetime - must be a 1-D array
def ts_to_datetime(arr):
    return pd.to_datetime(arr, unit='s')
