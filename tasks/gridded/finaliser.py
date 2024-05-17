import os
import sys
import gc
import re
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
import dask.array as da
from datacube import Datacube
from datacube.api import GridWorkflow
from datacube.utils.masking import make_mask, mask_invalid_data
from odc.algo import to_f32, mask_cleanup
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
        # sam_bool = path.rglob('sam_bool*.tif')
        # sam_dates = path.rglob('sam_dates*.tif')
        # sam_mgs = path.rglob('sam_mgs*.tif')
        # sam_prod = path.rglob('sam_products*.tif')
        # sam_post_prod = path.rglob('sam_post_products*.tif')
        # sam_post_dates = path.rglob('sam_post_dates*.tif')

        sam_bool_path = path / '*/*/sam_bool*.tif'
        sam_dates_path = path / '*/*/sam_dates*.tif'
        sam_mgs_path = path / '*/*/sam_mgs*.tif'
        sam_prod_path = path / '*/*/sam_products*.tif'
        sam_post_prod_path = path / '*/*/sam_post_products*.tif'
        sam_post_dates_path = path / '*/*/sam_post_dates*.tif'

        # Merge into data arrays
        # mgs_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_mgs]
        # mgs = xr.combine_by_coords(mgs_).compute()

        # dates_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_dates]
        # dates_data = xr.combine_by_coords(dates_).compute()
        # sam_timestamps = dates_data.where(dates_data != 0)

        # bool_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_bool]
        # bool_data = xr.combine_by_coords(bool_).compute()

        # product_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_prod]
        # product_data = xr.combine_by_coords(product_).compute()

        # post_prod_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_post_prod]
        # post_product_data = xr.combine_by_coords(post_prod_).compute()

        # post_dates_ = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_post_dates]
        # post_dates_data = xr.combine_by_coords(post_dates_).compute()

        mgs = xr.open_mfdataset(str(sam_mgs_path), parallel=True).squeeze(drop=True).rename({'band_data':'mgs'}).mgs
        dates_data = xr.open_mfdataset(str(sam_dates_path), parallel=True).squeeze(drop=True).rename({'band_data':'dates'}).dates
        bool_data = xr.open_mfdataset(str(sam_bool_path), parallel=True).squeeze(drop=True).rename({'band_data':'bool'}).bool
        product_data = xr.open_mfdataset(str(sam_prod_path), parallel=True).squeeze(drop=True).rename({'band_data':'product'}).product
        post_product_data = xr.open_mfdataset(str(sam_post_prod_path), parallel=True).squeeze(drop=True).rename({'band_data':'post_product'}).post_product
        post_dates_data = xr.open_mfdataset(str(sam_post_dates_path), parallel=True).squeeze(drop=True).rename({'band_data':'post_dates'}).post_dates

        sam_timestamps = dates_data.where(dates_data != 0)

        self._logger.info("Data downloaded and assembled")

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

        # Prepare list of primary landsat dates
        datetimes = ds.time.where(ds.time.dt.year >= datetime.datetime.fromtimestamp(sam_timestamps.min().values.item()).year,drop=True)

        dates = list(set(datetimes.dt.strftime('%Y%m%d').values))
        dates.sort()

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        dates = list(chunks(dates,10)) # Process 10 days at a time

        self._logger.debug("Closing local dask cluster")
        
        
        # Function to count neighbours based on a rolling window and a given number of days
        def count_neighbours(data,days=1):
            # Remove zero values and convert to integer
            tmp_ = da.where(data > 0, data, 0).astype(int)
            
            # Retrieve the middle pixel values and keep a compatible array shape
            centroid_ = np.take(tmp_,[2],axis=2)
            centroid_ = np.take(centroid_,[2],axis=3)
            
            # Round unix timestamps to "days"
            seconds_per_day = 60*60*24
            centroid_ = centroid_ - (centroid_ % seconds_per_day)
            
            # Add 1 second less than a full day to result in times of 23:59:59 to ensure correct date 
            centroid_ = da.where(centroid_!=0, centroid_ + (seconds_per_day-1), 0)
            # Get the target previous date value
            centroid_previous_ = da.where(centroid_!=0,centroid_ - (days * seconds_per_day),0,)
            
            # Count the number of non-zero values within the spatial and temporal window
            res = da.count_nonzero(((tmp_>centroid_previous_) & (tmp_<=centroid_)),axis=(2,3))
            
            # Convert to back xarray
            arr = xr.zeros_like(dates_data)
            arr.data = res
            # res = xr.DataArray(res,dims=['y','x'],coords=sam_ts.coords).astype('int32')
            
            # Filter to original change pixels and compute
            arr = arr.where(sam_ts!=0).compute()
            return arr

        # Configure rolling window - currently 5 x 5 pixels
        # This has to be an odd number so that we can get the central pixel
        self._logger.info("Counting repetitions")
        # sam_timestamps = sam_timestamps.chunk({'x':2000,'y':2000})

        sam_ts = xr.where(sam_timestamps.isnull(), 0, sam_timestamps.astype('int32'))

        rolling = sam_ts.rolling(y=5,x=5,min_periods=1,center=True).construct(y='y_window',x='x_window')
        
        rep_1d = count_neighbours(rolling,days=1).where(sam_ts!=0) - 1 # Don't count the central pixel
        rep_60d = count_neighbours(rolling,days=60).where(sam_ts!=0) - 1 # Don't count the central pixel
        rep_60d = rep_60d-rep_1d
        self.close_client()

        # Prepare geotiffs for output
        PATTERN = re.compile('^sam_\w*_(?P<filespec>RF_(?P<rf_version>v\w{2})_.*)_\w{8}.tif$')
        match = PATTERN.match(next(path.rglob('sam_mgs*.tif')).name)
        filespec = match['filespec']
        # filespec = 'all-trained_negative_of_first_last_negative'

        # TODO: CHECK FLOAT32 ISSUE
        self._logger.info(f"Writing output geotiffs for {filespec}")
        write_cog(mgs, fname = path / f'sam_mgs_{filespec}.tif', nodata=np.nan, overwrite=True).compute()
        write_cog(dates_data, fname = path / f'sam_dates_{filespec}.tif', nodata=0, overwrite=True).compute()
        write_cog(bool_data, fname = path / f'sam_bool_{filespec}.tif', nodata=0, overwrite=True).compute()
        write_cog(product_data, fname = path / f'sam_products_{filespec}.tif', nodata=0, overwrite=True).compute()
        write_cog(post_product_data, fname = path / f'sam_post_products_{filespec}.tif', nodata=0, overwrite=True).compute()
        write_cog(post_dates_data, fname = path / f'sam_post_dates_{filespec}.tif', nodata=0, overwrite=True).compute()
        write_cog(rep_1d, fname = path / f'sam_rep_1d_{filespec}.tif', nodata=np.nan, overwrite=True).compute()
        write_cog(rep_60d, fname = path / f'sam_rep_60d_{filespec}.tif', nodata=np.nan, overwrite=True).compute()
        
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
        # Clean up memory before starting
        gc.collect()

        for date in self.dates[int(self.dates_idx)]:
            date = datetime.datetime.strptime(str(date), "%Y%m%d").date()
            self._logger.info(f"Processing {date}")

            # Download data
            bucket = self.output["bucket"]
            prefix = str(Path(self.output['prefix']) / 'final')
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
            sam_post_prod = path.rglob('sam_post_products*.tif')
            sam_post_dates = path.rglob('sam_post_dates*.tif')
            sam_rep_1d = path.rglob('sam_rep_1d_*.tif')
            sam_rep_60d = path.rglob('sam_rep_60d_*.tif')

            mgs = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_mgs]
            mgs = xr.combine_by_coords(mgs).compute()

            dates_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_dates]
            dates_data = xr.combine_by_coords(dates_data).compute()
            sam_timestamps = dates_data.where(dates_data != 0)
            del dates_data
            sam_dates = xr.DataArray(pd.to_datetime(sam_timestamps, unit='s').values,dims=['y','x'])

            bool_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_bool]
            bool_data = xr.combine_by_coords(bool_data).compute()

            product_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_prod]
            product_data = xr.combine_by_coords(product_data).compute()

            post_product_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_post_prod]
            post_product_data = xr.combine_by_coords(post_product_data).compute()

            post_dates_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_post_dates]
            post_dates_data = xr.combine_by_coords(post_dates_data).compute()

            rep_1d_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_rep_1d]
            rep_1d_data = xr.combine_by_coords(rep_1d_data).compute()

            rep_60d_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_rep_60d]
            rep_60d_data = xr.combine_by_coords(rep_60d_data).compute()

            attrs = {'crs': 'epsg:32619', 'grid_mapping': 'spatial_ref'}

            PATTERN = re.compile('^sam_\w*_(?P<filespec>RF_(?P<rf_version>v\w{2})_.*)_\w{8}.tif$')
            match = PATTERN.match(next(path.rglob('sam_mgs*.tif')).name)
            rf_version = match['rf_version']

            out_path = Path(self.temp_dir.name) / "dcc_format" / rf_version / self.way
            filename = "break000"

            # TODO: CHANGE TO DAY - should write out days not timestamps
            data_mgs = xr.where(sam_dates.dt.date == date, mgs, np.nan)
            data_mgs.attrs = attrs
            data_mgs.rio.write_nodata(np.nan, inplace=True)

            if data_mgs.count() != 0:
                mpath = out_path / date.strftime('%Y%m%d')
                mpath.mkdir(parents=True, exist_ok=True)

                combined_ds = data_mgs.to_dataset(name='mgs')

                combined_ds['product'] = xr.where(sam_dates.dt.date == date, product_data, 0).astype('uint32')
                combined_ds['product_post'] = xr.where(sam_dates.dt.date == date, post_product_data, 0).astype('uint32')
                combined_ds['date_post'] = xr.where(sam_dates.dt.date == date, post_dates_data, 0).astype('uint32')
                combined_ds['rep_1d'] = xr.where(sam_dates.dt.date == date, rep_1d_data, 0).astype('uint32')
                combined_ds['rep_60d'] = xr.where(sam_dates.dt.date == date, rep_60d_data, 0).astype('uint32')
                for var in combined_ds.data_vars:
                    combined_ds[var].attrs = attrs
                    combined_ds[var].rio.write_nodata(0, inplace=True)

                fname = mpath / f"{date.strftime('%Y%m%d')}_{filename}"

                del data_mgs
                del mgs
                del product_data
                del post_product_data
                del post_dates_data
                del rep_1d_data
                del rep_60d_data
                gc.collect()

                self._logger.info("Writing final data")
                write_cog(combined_ds.mgs, fname=f'{fname}_mag.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.product, fname=f'{fname}_product.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.product_post, fname=f'{fname}_product_post.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.date_post, fname=f'{fname}_date_post.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.rep_1d, fname=f'{fname}_rep_1d.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.rep_60d, fname=f'{fname}_rep_60d.tif', nodata=0, overwrite=True)
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
                