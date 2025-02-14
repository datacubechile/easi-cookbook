import sys
import gc
import re
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(1, '/home/jovyan/SAMSARA/lib-samsara/src')
sys.path.insert(1, 'lib-samsara/src/')
sys.path.insert(1, '.')

import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import rioxarray
import xarray as xr
import geopandas as gpd
from dask.distributed import Client, LocalCluster
import dask.array as da
from datacube import Datacube
from tasks.argo_task import ArgoTask
from datacube.utils.rio import configure_s3_access
from datacube.utils.cog import write_cog
from rasterio.features import rasterize

from tasks.common import get_prior_date, ts_to_datetime
from tasks import geohash as gh

from tasks import samsara_prepare

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class Assemble(ArgoTask):
    def __init__(self, input_params: [dict[str, str]]) -> None:
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
        self.dask_workers = 4 if "dask_workers" not in names else self.dask_workers

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

    def assemble(self, get_dates_only=False) -> None:
        for log in ['distributed', 'distributed.nanny','distributed.scheduler','distributed.client']:
            logger = logging.getLogger(log)
            logger.setLevel(logging.ERROR)

        # Download data
        bucket = self.output["bucket"]
        prefix = str(Path(self.output['prefix']) / 'predict/RF')
        path = Path(self.temp_dir.name)

        dt_format = '%Y%m%d'

        date_key = min(datetime.datetime.now(), datetime.datetime.strptime(self.roi["time_end"][:10],'%Y-%m-%d'))
        date_key_str = date_key.strftime(dt_format)

        self._logger.info(f"    Downloading s3://{bucket}/{prefix} to {path}")
        self.s3_download_folder(
            prefix=prefix,
            bucket=bucket,
            path=str(path)
        )

        sam_bool_path = path / '*/*/sam_bool*.tif'
        sam_dates_path = path / '*/*/sam_dates*.tif'
        sam_mgs_path = path / '*/*/sam_mgs*.tif'
        sam_prod_path = path / '*/*/sam_products*.tif'
        sam_post_prod_path = path / '*/*/sam_post_products*.tif'
        sam_post_dates_path = path / '*/*/sam_post_dates*.tif'

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

        if get_dates_only:
            with open('/tmp/dates_idx', 'w') as outfile:
                json.dump(list(range(0,len(dates))),outfile)

            with open('/tmp/dates','w') as outfile:
                json.dump(dates, outfile)

            with open("/tmp/changes", "w") as outfile:
                outfile.write("")

            with open("/tmp/prior_date", "w") as outfile:
                outfile.write("")

            with open("/tmp/latest_date", "w") as outfile:
                outfile.write(date_key_str)
            return

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

        def classify_with_shapefile(dataset,shapefile):
            gdf = gpd.read_file(shapefile)
            gdf = gdf.to_crs(dataset.rio.crs)
            polys = [geometry for geometry in gdf.geometry]
            mask = rasterize(
                    polys,
                    out_shape = (dataset.dims['y'],dataset.dims['x']),
                    transform = dataset.affine
            )
            da = dataset[list(dataset.data_vars.keys())[0]].compute()
            da = xr.where(da.notnull() & mask,1,0)
            return da

        # Download shapefiles
        prefix = str(Path(self.output['prefix']).parent / 'resources' / 'areas_protegidas')
        tmp_path = Path(self.temp_dir.name) / 'areas_protegidas'
        self._logger.info(f"    Downloading s3://{bucket}/{prefix} to {tmp_path}")
        self.s3_download_folder(
            prefix=prefix,
            bucket=bucket,
            path=str(tmp_path)
        )

        areas_protegidas = classify_with_shapefile(bool_data.to_dataset(), tmp_path / 'Areas Protegidas_metropolitana.shp')

        prefix = str(Path(self.output['prefix']).parent / 'resources' / 'sitios_prioritarios')
        tmp_path = Path(self.temp_dir.name) / 'sitios_prioritarios' 
        self._logger.info(f"    Downloading s3://{bucket}/{prefix} to {tmp_path}")
        self.s3_download_folder(
            prefix=prefix,
            bucket=bucket,
            path=str(tmp_path)
        )

        sitios_prioritarios = classify_with_shapefile(bool_data.to_dataset(), tmp_path / 'Sitios Prioritarios.shp')

        # Configure rolling window - currently 5 x 5 pixels
        # This has to be an odd number so that we can get the central pixel
        self._logger.info("Counting repetitions")
        # sam_timestamps = sam_timestamps.chunk({'x':2000,'y':2000})

        sam_ts = xr.where(sam_timestamps.isnull(), 0, sam_timestamps.astype('int32'))

        rolling = sam_ts.rolling(y=5,x=5,min_periods=1,center=True).construct(y='y_window',x='x_window')

        rep_1d = count_neighbours(rolling,days=1).where(sam_ts!=0) - 1 # Don't count the central pixel
        rep_60d = count_neighbours(rolling,days=60).where(sam_ts!=0) - 1 # Don't count the central pixel
        rep_60d = rep_60d-rep_1d

        prior_date = get_prior_date(bucket, Path(self.output['prefix']) / 'final', date_key, dt_format)
        # latest_date, prior_date = get_most_recent_dates(bucket, prefix, dt_format)
        prior_date_str = prior_date.strftime(dt_format) if prior_date else ""

        if prior_date:
            self._logger.info(f"Found a previous date to compare to: {prior_date_str}")
            self._logger.info(f"    Downloading s3://{bucket}/{self.output['prefix']}final/{prior_date_str} to {path / prior_date_str}")
            self.s3_download_folder(
                prefix=f"{self.output['prefix']}final/{prior_date_str}/",
                bucket=bucket,
                path=str(path / prior_date_str)
            )

            prior_date_path = path / prior_date_str
            # Get the files for the second most recent run
            sam_dates_prior = prior_date_path.rglob('sam_dates*.tif')
            sam_dates_data_prior = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_dates_prior]
            sam_dates_data_prior = xr.combine_by_coords(sam_dates_data_prior).compute()
            sam_dates_data_prior = sam_dates_data_prior.fillna(0) # fill NAs with 0s so that we get new dates
            sam_timestamps_changed = sam_timestamps.where(sam_timestamps > sam_dates_data_prior) # Unlikely to be backward changes, but just in case, filter only to "more recent" changes
            sam_timestamps_changed = sam_timestamps_changed.where(sam_timestamps_changed > (prior_date - relativedelta(years=1)).timestamp()) # Only return changes in the last 12 months
            # write_cog(sam_timestamps_changed.compute(), fname = prior_date_path / f'sam_timestamps_changed.tif', nodata=np.nan, overwrite=True)
            # TODO: upload changed file
            sam_timestamps_changed = sam_timestamps_changed.compute()
            sam_dates_changed = xr.apply_ufunc(
                ts_to_datetime,
                sam_timestamps_changed,
                input_core_dims=[['x']],
                output_core_dims=['x'],
                vectorize=True,
            )
            sam_changed_ds = sam_dates_changed.to_dataset(name='dates')
            sam_changed_ds['mag'] = mgs.where(~np.isnan(sam_changed_ds.dates))
            sam_changed_ds["rep_1d"] = rep_1d.where(~np.isnan(sam_changed_ds.dates))
            sam_changed_ds['rep_60d'] = rep_60d.where(~np.isnan(sam_changed_ds.dates))
            sam_changed_ds['areas_protegidas'] = areas_protegidas.where(~np.isnan(sam_changed_ds.dates))
            sam_changed_ds['sitios_prioritarios'] = sitios_prioritarios.where(~np.isnan(sam_changed_ds.dates))

            sam_changed_ds = sam_changed_ds.where(~np.isnan(sam_changed_ds.dates), drop=True)
            sam_changed_df = sam_changed_ds.to_dataframe().dropna()
            sam_changed_df['geohash'] = list(map(gh.encode_from_xy, sam_changed_df.index.get_level_values('x'), sam_changed_df.index.get_level_values('y')))
            sam_changed_df = sam_changed_df.to_json()
        else:
            sam_changed_df = []

        self._logger.debug("Closing local dask cluster")
        self.close_client()
        with open('/tmp/changes','w') as outfile:
            json.dump(sam_changed_df, outfile)
        with open('/tmp/prior_date','w') as outfile:
            outfile.write(prior_date_str)
        with open('/tmp/latest_date','w') as outfile:
            outfile.write(date_key_str)

        # Prepare geotiffs for output
        PATTERN = re.compile(r'^sam_\w*_(?P<filespec>RF_(?P<rf_version>v\w{2})_.*)_\w{8}.tif$')
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
        write_cog(areas_protegidas, fname = path / f'sam_areas_protegidas_{filespec}.tif', nodata=0, overwrite=True)
        write_cog(sitios_prioritarios, fname = path / f'sam_sitios_prioritarios_{filespec}.tif', nodata=0, overwrite=True)

        if self.output['upload']:
            for file_path in path.glob("*.tif"):
                if not file_path.is_file():
                    continue
                key = str(Path(self.output['prefix']) / 'final' / date_key_str / file_path.relative_to(self.temp_dir.name))

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
        self.dask_workers = 4 if 'dask_workers' not in names else self.dask_workers

        self.temp_dir = TemporaryDirectory()

        with open('/tmp/dates', "r") as f:
            self.dates = json.load(f)

    def finalise(self, latest_date) -> None:
        # Clean up memory before starting
        gc.collect()

        # Download data
        bucket = self.output["bucket"]
        prefix = str(Path(self.output['prefix']) / 'final' / latest_date)
        path = Path(self.temp_dir.name)

        self._logger.info(f"    Downloading s3://{bucket}/{prefix} to {path}")
        self.s3_download_folder(
            prefix=f"{prefix}",
            bucket=bucket,
            path=str(path)
        )
        
        # Get all the files for the most recent run
        sam_bool = path.rglob('sam_bool*.tif')
        sam_dates = path.rglob('sam_dates*.tif')
        sam_mgs = path.rglob('sam_mgs*.tif')
        sam_prod = path.rglob('sam_products*.tif')
        sam_post_prod = path.rglob('sam_post_products*.tif')
        sam_post_dates = path.rglob('sam_post_dates*.tif')
        sam_rep_1d = path.rglob('sam_rep_1d_*.tif')
        sam_rep_60d = path.rglob('sam_rep_60d_*.tif')
        sam_areas_protegidas = path.rglob('sam_areas_protegidas_*.tif')
        sam_sitios_prioritarios = path.rglob('sam_sitios_prioritarios_*.tif')

        mgs = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_mgs]
        mgs = xr.combine_by_coords(mgs).compute()

        dates_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_dates]
        dates_data = xr.combine_by_coords(dates_data).compute()
        sam_timestamps = dates_data.where(dates_data != 0)
        del dates_data

        # sam_dates = xr.DataArray(pd.to_datetime(sam_timestamps*1e9, unit='ns').values,dims=['y','x'])
        # Apply ts_to_datetime along the y-dimension
        sam_dates = xr.apply_ufunc(
            ts_to_datetime,
            sam_timestamps,
            input_core_dims=[['x']],
            output_core_dims=['x'],
            vectorize=True,
        )

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

        areas_protegidas_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_areas_protegidas]
        areas_protegidas_data = xr.combine_by_coords(areas_protegidas_data).compute()

        sitios_prioritarios_data = [rioxarray.open_rasterio(f).isel(band=0,drop=True) for f in sam_sitios_prioritarios]
        sitios_prioritarios_data = xr.combine_by_coords(sitios_prioritarios_data).compute()

        attrs = {'crs': 'epsg:32619', 'grid_mapping': 'spatial_ref'}

        PATTERN = re.compile(r'^sam_\w*_(?P<filespec>RF_(?P<rf_version>v\w{2})_.*)_\w{8}.tif$')
        match = PATTERN.match(next(path.rglob('sam_mgs*.tif')).name)
        rf_version = match['rf_version']

        out_path = Path(self.temp_dir.name) / "dcc_format" / rf_version / self.neighbor_params['way']
        filename = "break000"


        for date in self.dates[int(self.dates_idx)]:
            date = datetime.datetime.strptime(str(date), "%Y%m%d").date()
            self._logger.info(f"Processing {date}")

            # TODO: CHANGE TO DAY - should write out days not timestamps
            data_mgs = xr.where(sam_dates.dt.date == date, mgs, np.nan)
            data_mgs.attrs = attrs
            data_mgs.rio.write_nodata(np.nan, inplace=True)

            if data_mgs.count() != 0:
                mpath = out_path / date.strftime('%Y%m%d')
                mpath.mkdir(parents=True, exist_ok=True)

                combined_ds = data_mgs.to_dataset(name='mgs')

                combined_ds['product'] = xr.where(sam_dates.dt.date == date, product_data, 0).astype('uint8')
                combined_ds['product_post'] = xr.where(sam_dates.dt.date == date, post_product_data, 0).astype('uint8')
                combined_ds['date_post'] = xr.where(sam_dates.dt.date == date, post_dates_data, 0).astype('uint32')
                combined_ds['rep_1d'] = xr.where(sam_dates.dt.date == date, rep_1d_data, 0).astype('uint16')
                combined_ds['rep_60d'] = xr.where(sam_dates.dt.date == date, rep_60d_data, 0).astype('uint16')
                combined_ds['areas_protegidas'] = xr.where(sam_dates.dt.date == date, areas_protegidas_data, 0).astype('uint8')
                combined_ds['sitios_prioritarios'] = xr.where(sam_dates.dt.date == date, sitios_prioritarios_data, 0).astype('uint8')
                for var in combined_ds.data_vars:
                    combined_ds[var].attrs = attrs
                    combined_ds[var].rio.write_nodata(0, inplace=True)

                fname = mpath / f"{date.strftime('%Y%m%d')}_{filename}"

                self._logger.info("Writing final data")
                write_cog(combined_ds.mgs, fname=f'{fname}_mag.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.product, fname=f'{fname}_product.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.product_post, fname=f'{fname}_product_post.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.date_post, fname=f'{fname}_date_post.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.rep_1d, fname=f'{fname}_rep_1d.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.rep_60d, fname=f'{fname}_rep_60d.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.areas_protegidas, fname=f'{fname}_areas_protegidas.tif', nodata=0, overwrite=True)
                write_cog(combined_ds.sitios_prioritarios, fname=f'{fname}_sitios_prioritarios.tif', nodata=0, overwrite=True)
                # combined_ds.rio.to_raster(f'{fname}_multiband.tif',driver='COG')
                samsara_prepare.prepare_samsara_raw(fname.parent)

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
