import os
import sys
import json
import logging
import pickle
import math
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(1, '/home/jovyan/SAMSARA/lib-samsara/src')
sys.path.insert(1, 'lib-samsara/src/')
sys.path.insert(1, '.')

import datetime
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from dask.distributed import Client, LocalCluster
from datacube import Datacube
from datacube.api import GridWorkflow
from datacube.utils.masking import make_mask, mask_invalid_data
from eodatasets3 import DatasetPrepare
from tasks.argo_task import ArgoTask
from datacube.utils.rio import configure_s3_access
from datacube.utils.cog import write_cog

from tasks.common import calc_chunk, s3_download_file, s3_download_folder

from tasks import samsara_prepare

# from tasks import samsara_prepare
from multiprocessing.pool import ThreadPool
import tqdm
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from tasks.argo_task import ArgoTask

class Assemble(ArgoTask):
    def __init__(self, input_params: [{str, str}]) -> None:
        """Check and cast input params as required.

        The `input_params` gets mapped to instance attributes. The constructor allows to
        optionally check, cast or otherwise modify these parameters as required. In this
        example, we set `measurements` to an empty list if it's an empty string and
        convert the `key` from a list or lists to a list of tuples.
        """
        super().__init__(input_params)
        names = [i['name'] for i in input_params]
        self.id_ = 0

        self._client = None
        self._cluster = None
        self.dask_workers = 4 if not 'dask_workers' in names else self.dask_workers

        self.temp_dir = TemporaryDirectory()

    def start_client(self) -> None:
        """Start a local dask cluster, if needed."""
        if self._client is None:
            self._cluster = LocalCluster(n_workers=self.dask_workers)
            self._client = Client(self._cluster)
            configure_s3_access(aws_unsigned=False, requester_pays=True, client=self._client)

    def close_client(self) -> None:
        """Close a local dask cluster, if running."""
        if self._client is not None:
            self._client.close()
            self._cluster.close()
            self._client = None
            self._cluster = None

    def assemble(self) -> None:
        for log in ['distributed', 'distributed.nanny','distributed.scheduler','distributed.client']:
            logger = logging.getLogger(log)
            logger.setLevel(logging.ERROR)

        # Download data
        bucket = self.output["bucket"]
        prefix = str(Path(self.output['prefix']) / 'predict/RF')
        path = Path(self.temp_dir.name)

        self._logger.info(f"    Downloading s3://{bucket}/{prefix} to {path}")
        self.s3_download_folder(
            prefix=prefix,
            bucket=bucket,
            path=str(path)
        )

        # Get all the files
        sam_bool = path.rglob('sam_bool*.tif')
        sam_dates = path.rglob('sam_dates*.tif')
        sam_mgs = path.rglob('sam_mgs*.tif')
        sam_prod = path.rglob('sam_products*.tif')

        # Merge into data arrays
        mgs_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_mgs]
        mgs = xr.combine_by_coords(mgs_).compute()

        dates_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_dates]
        dates_data = xr.combine_by_coords(dates_).compute()
        sam_timestamps = dates_data.where(dates_data != 0)

        bool_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_bool]
        bool_data = xr.combine_by_coords(bool_).compute()

        product_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_prod]
        product_data = xr.combine_by_coords(product_).compute()

        write_cog(mgs, fname = path / 'sam_mgs_RF_v03_all-trained_negative_of_first_last_negative.tif', nodata=np.nan, overwrite=True)
        write_cog(dates_data, fname = path / 'sam_dates_RF_v03_all-trained_negative_of_first_last_negative.tif', nodata=0, overwrite=True)
        write_cog(bool_data, fname = path / 'sam_bool_RF_v03_all-trained_negative_of_first_last_negative.tif', nodata=0, overwrite=True)
        write_cog(product_data, fname = path / 'sam_products_RF_v03_all-trained_negative_of_first_last_negative.tif', nodata=0, overwrite=True)

        self._logger.info("Data downloaded and assembled")
        if self.output['upload']:
            for file_path in path.glob("*.tif"):
                if not file_path.is_file():
                    continue
                key = str(Path(self.output['prefix']) / 'final' / file_path.relative_to(self.temp_dir.name))

                self._logger.info(f"    Uploading {file_path} to s3://{bucket}/{key}")
                self.s3_upload_file(
                    path=str(file_path),
                    bucket=bucket,
                    key=key,
                )
            self._logger.info("Completed upload of assembled data")

        self._logger.debug("Initialising local dask cluster")
        self.start_client()

        dc = Datacube()

        query = {
            "time":(self.roi["time_start"], self.roi["time_end"]),
            "y": mgs.geobox.extent.boundingbox.range_y,
            "x": mgs.geobox.extent.boundingbox.range_x,
            "measurements": "red",
            "crs": self.odc_query['output_crs'],
            "output_crs": self.odc_query['output_crs'],
            "resolution": self.odc_query['resolution'],
            "group_by": self.odc_query['group_by'],
            "dask_chunks": {'time':1}
        }

        products = self.product

        data = []
        
        for product in products:
            ds_ = dc.load(
                product=product,
                **query)
            if len(ds_) > 0:
                ds_['product'] = ('time', np.repeat(product, ds_.time.size))
                data.append(ds_)
        ds = xr.concat(data, dim='time')
        ds = ds.chunk({'time':1,'x':2000,'y':2000})
        ds = ds.sortby('time')
        self._logger.info(f"{ds.time.count().values.item()} available landsat dates loaded from data cube")
        
        datetimes = ds.time.where(ds.time.dt.year >= datetime.datetime.fromtimestamp(sam_timestamps.min().values.item()).year,drop=True)
        # TODO: EXPORT DAYS, NOT TIMESTAMPS
        dates = list(set(datetimes.dt.strftime('%Y%m%d').values))
        dates.sort()
        # timestamps = (datetimes.astype(int)*1e-09).astype(int)
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        dates = list(chunks(dates,10))

        self._logger.debug("Closing local dask cluster")
        self.close_client()

        with open('/tmp/dates_idx', 'w') as outfile:
            json.dump(list(range(0,len(dates))),outfile)

        with open('/tmp/dates','w') as outfile:
            json.dump(dates, outfile)

        with open('/tmp/way', 'w') as outfile:
            outfile.write([f.name for f in os.scandir(path) if f.is_dir()][0])
        return ds, sam_timestamps

class Finalise(ArgoTask):
    def __init__(self, input_params: [{str, str}]) -> None:
        """Check and cast input params as required.

        The `input_params` gets mapped to instance attributes. The constructor allows to
        optionally check, cast or otherwise modify these parameters as required. In this
        example, we set `measurements` to an empty list if it's an empty string and
        convert the `key` from a list or lists to a list of tuples.
        """
        super().__init__(input_params)
        names = [i['name'] for i in input_params]
        self.id_ = 0

        self._client = None
        self._cluster = None
        self.dask_workers = 4 if not 'dask_workers' in names else self.dask_workers

        self.temp_dir = TemporaryDirectory()

        with open('/tmp/dates', "r") as f:
            self.dates = json.load(f)

    def finalise(self) -> None:
        for date in self.dates[int(self.dates_idx)]:
            date = datetime.datetime.strptime(str(date), "%Y%m%d").date()
            self._logger.info(f"Processing {date}")

            way = self.way
            out_path = Path(self.temp_dir.name) / "dcc_format" / "v04" / way
            filename = "break000"

            # Download data
            bucket = self.output["bucket"]
            prefix = str(Path(self.output['prefix']) / 'predict' / 'final')
            path = Path(self.temp_dir.name)

            self._logger.info(f"    Downloading s3://{bucket}/{prefix} to {path}")
            self.s3_download_folder(
                prefix=prefix,
                bucket=bucket,
                path=str(path)
            )

            # Get all the files
            sam_bool = path.rglob('sam_bool*.tif')
            sam_dates = path.rglob('sam_dates*.tif')
            sam_mgs = path.rglob('sam_mgs*.tif')
            sam_prod = path.rglob('sam_products*.tif')

            mgs_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_mgs]
            mgs = xr.combine_by_coords(mgs_).compute()

            dates_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_dates]
            dates_data = xr.combine_by_coords(dates_).compute()
            sam_timestamps = dates_data.where(dates_data != 0)
            sam_dates = xr.DataArray(pd.to_datetime(sam_timestamps, unit='s').values,dims=['y','x'])

            bool_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_bool]
            bool_data = xr.combine_by_coords(bool_).compute()

            product_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_prod]
            product_data = xr.combine_by_coords(product_).compute()

            attrs = {'crs': 'epsg:32619', 'grid_mapping': 'spatial_ref'}

            # TODO: CHANGE TO DAY - should write out days not timestamps
            data_mgs = xr.where(sam_dates.dt.date == date, mgs, np.nan)
            data_mgs.attrs = attrs
            data_mgs.rio.write_nodata(np.nan, inplace=True)

            if data_mgs.count() != 0:
                mpath = out_path / date.strftime('%Y%m%d')
                mpath.mkdir(parents=True, exist_ok=True)

                combined_ds = data_mgs.to_dataset(name='mgs')
                combined_ds['product'] = xr.where(sam_dates.dt.date == date, product_data, np.nan).astype('float32')
                combined_ds.product.attrs = attrs
                combined_ds.product.rio.write_nodata(np.nan, inplace=True)

                fname = mpath / f"{date.strftime('%Y%m%d')}_{filename}"
                
                self._logger.info("Writing final data")
                write_cog(combined_ds.mgs, fname=f'{fname}_mag.tif', nodata=np.nan, overwrite=True)
                write_cog(combined_ds.product, fname=f'{fname}_product.tif', nodata=np.nan, overwrite=True)
                combined_ds.rio.to_raster(f'{fname}_multiband.tif',driver='COG')
                samsara_prepare.prepare_samsara(fname.parent)

                if self.output['upload']:
                    for file_path in out_path.rglob("*"):
                        if not file_path.is_file():
                            continue
                        key = str(Path(self.output['final_prefix']) / file_path.relative_to(out_path))

                        bucket = self.output["bucket"]

                        self._logger.info(f"    Uploading {file_path} to s3://{bucket}/{key}")
                        self.s3_upload_file(
                            path=str(file_path),
                            bucket=bucket,
                            key=key,
                        )
            else:
                self._logger.info(f"{date} has no data, skipping")
