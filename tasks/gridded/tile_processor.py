"""
Example of geomedian calculation for a tile, using a local dask cluster.
"""
import sys
import logging
import pickle
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import joblib
import numpy as np
import rioxarray
import xarray as xr

sys.path.insert(1, '/home/jovyan/SAMSARA/lib-samsara/src')
sys.path.insert(1, 'lib-samsara/src/')
sys.path.insert(1, '.')

import datetime
import pandas as pd
from dask.distributed import Client, LocalCluster, wait, as_completed
from datacube import Datacube
from datacube.api import GridWorkflow
from datacube.utils.masking import make_mask, mask_invalid_data
from eodatasets3 import DatasetPrepare
from odc.algo import from_float, to_f32, xr_geomedian
from tasks.argo_task import ArgoTask
from tasks.gridded.tile_generator import TileGenerator
from xarray import Dataset, DataArray
from datacube.utils.rio import configure_s3_access
from datacube.utils.cog import write_cog
from dea_tools.classification import predict_xr
from rasterio.enums import Resampling

from tasks.common import calc_chunk, s3_download_file

import samsara.images as simages
import samsara.pelt as spelt
import samsara.filter as sfilter
import samsara.stats.neighborhood as sns
import samsara.kernel as skernel
import samsara.stats.glcm as sglcm

logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.WARNING)

class TileProcessor(ArgoTask):
    PRODUCT = "landsat8_geomedian_monthly"
    """ODC product name being generated."""

    PRODUCT_TEMPLATE = "{product}_{yref}{xref}_{year:04}{month:02}"
    """File name template."""

    SCALE = 0.0000275
    """Scale to apply to GA-stored USGS data to bring it back to [0.0, 1.0]."""

    ONE_MONTH = pd.DateOffset(months=1)
    """A one month time delta."""

    ONE_DAY = pd.DateOffset(days=1)
    """A one day time delta."""

    def __init__(self, input_params: [{str, str}]) -> None:
        """Check and cast input params as required.

        The `input_params` gets mapped to instance attributes. The constructor allows to
        optionally check, cast or otherwise modify these parameters as required. In this
        example, we set `measurements` to an empty list if it's an empty string and
        convert the `key` from a list or lists to a list of tuples.
        """
        super().__init__(input_params)
        self.id_ = 0
        self.measurements = [] if self.measurements == "" else self.measurements

        self.compute_pelt = "True" if not self.pelt_params.get('compute_pelt') else self.pelt_params.get('compute_pelt')
        self.compute_neighbors = "True" if not self.neighbor_params.get('compute_neighbors') else self.neighbor_params.get('compute_neighbors')
        self.compute_textures = "True" if not self.texture_params.get('compute_textures') else self.texture_params.get('compute_textures')
        self.compute_rf = "True" if not self.rf_params.get('compute_rf') else self.rf_params.get('compute_rf')

        # Convert key from list[list] to list[tuple]
        self.key = [tuple(k) for k in self.key]

        self._client = None
        self._cluster = None
        self.dask_workers = 4 if self.dask_workers == "" else self.dask_workers

        self.temp_dir = TemporaryDirectory()

        # Unpickle the product cells from file
        with open(TileGenerator.FILEPATH_CELLS, "rb") as fh:
            self.product_cells = pickle.load(fh)

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

    def load_from_grid(self, key: (int, int)) -> Dataset:
        """Load data from grid flow."""
        
        def load_product(product, query, chunks):
            dc = Datacube()
            ds = dc.load(product=product, **query)
            # # Rechunking early seems to be best...
            ds = ds.chunk(chunks)
            # # Calculate NDVI
            ndvi = simages.mask_and_calculate_ndvi(ds)
            ndvi = ndvi.to_dataset(name='ndvi')
            # # Add product name to the output
            ndvi['product'] = ('time', np.repeat(product, ndvi.time.size))
            # # Start calculating
            ndvi.attrs = ds.attrs
            return ndvi.persist()
            # result = 1+1
            # return result

        cell = self.product_cells.get(key)
        
        # All geoboxes for the tiles are the same shape. Use this for the chunk size in
        # dask so each tile spatially is a single chunk. Note that the geobox resolution
        # is in (y, x) order

        if self.tile_buffer:
            cell.geobox = cell.geobox.buffered(*self.tile_buffer) if self.tile_buffer else cell.geobox

        chunk_dim = cell.geobox.shape
        chunks = {"time": 1, "x": chunk_dim[1], "y": chunk_dim[0]}
        try:
            dc = Datacube()
            products = self.product
            query = {
                "measurements": self.measurements,
                "time":(self.roi["time_start"], self.roi["time_end"]),
                "like":cell.geobox,
                "dask_chunks":{"time":4},
                "group_by":"solar_day"
            }

            chunks = {'x':int(self.pelt_params['processing_chunk_size']), 'y':int(self.pelt_params['processing_chunk_size']), 'time':-1}
            
            # This approach increases parallelisation across the multi-product loads
            datasets = self._client.map(load_product, products, query=query, chunks=chunks)
            results = self._client.gather(datasets)
            wait(results)

            # Concatenate product datasets, make sure chunks are correct and sort on time
            combined = xr.concat(results, dim='time')
            combined = combined.chunk(chunks)
            combined = combined.sortby('time')

            self._logger.debug(f"Dataset for key {key} has dims: {str(combined.dims)}")
        except Exception as e:
            self._logger.error(f"load_from_grid: Exception: {e}")
            raise
        return combined

    def mask(self, ds: Dataset) -> Dataset:
        """Mask clouds and no data in data based on `oa_fmask` values."""
        # Use datacube masking methods
        # https://docs.dea.ga.gov.au/notebooks/How_to_guides/Masking_data.html
        cloud_free_mask = make_mask(
            ds.qa_pixel, water="land_or_cloud", clear="clear", nodata=False
        )

        # Set all nodata pixels to `NaN`:
        # float32 has sufficient precision for original uint16 SR_bands and saves memory
        cloud_free = mask_invalid_data(
            ds[["red", "nir08"]].astype("float32", casting="same_kind")
        )  #  remove the invalid data on Surface reflectance bands prior to masking clouds
        cloud_free = cloud_free.where(cloud_free_mask)

        return cloud_free

    def scale(self, ds: Dataset) -> Dataset:
        """Scale Landsat data before geomedian calculation.

        GA Landsat data is in the range [0, 10000] and needs scaling to [0.0,
        1.0] to calculate the geomedian.
        """
        return to_f32(ds, scale=self.SCALE, offset=-0.2)

    def run_pelt(self, ds: Dataset) -> Dataset:
        """Run PELT on the dataset."""
        # rupture parameters
        model = self.pelt_params['model']
        min_size = int(self.pelt_params['min_size'])
        jump = int(self.pelt_params['jump'])
        penalty = int(self.pelt_params['penalty'])

        # samsara parameters
        n_breaks = int(self.pelt_params['n_breaks'])
        start_date = self.pelt_params['start_date']

        # Make sure the chunks are right here... concatenating doesn't maintain chunk dimensions
        ds = ds.chunk({'x':int(self.pelt_params['processing_chunk_size']), 'y':int(self.pelt_params['processing_chunk_size']), 'time':-1})

        # ndvi = simages.mask_and_calculate_ndvi(ds)#.compute()
        # ndvi = ndvi.rechunk({'time':-1, "x": 65, "y": 65})
        
        # ds_scattered = self._client.scatter(ds)
        # pelt_args = {
        #     'array': ds_scattered,
        #     'n_breaks':n_breaks, 
        #     'penalty':penalty, 
        #     'start_date':start_date,
        #     'model':model,
        #     'min_size':min_size,
        #     'jump': jump,
        #     'backend': 'dask'
        # }
        # future = self._client.submit(spelt.pelt, **pelt_args)
        # fpelt = self._client.gather(future)
        fpelt = spelt.pelt(
            array = ds,
            n_breaks=n_breaks, 
            penalty=penalty, 
            start_date=start_date,
            model=model,
            min_size=min_size,
            jump = jump,
            backend = 'dask'
        )

        # fpelt.date.data = spelt.datetime_to_year_fraction(fpelt.date.data.astype('datetime64[s]'))
        # fpelt['date_original'] = fpelt.date
        # fpelt.date.data = spelt.datetime_to_timestamp(fpelt.date.data.astype('datetime64[s]'))

        return fpelt
    
    def run_neighbors(self, ds: Dataset) -> (Dataset, Dataset):
        pelt_filtered = sfilter.filter_by_variable(ds, filter_type = 'last_negative', variable = 'magnitude').chunk({'x':200, 'y':200})

        date_std = sns.stats(pelt_filtered, stat = "std", kernel = int(self.neighbor_params['neighbor_radius']), variable = 'date')
        date_cnt = sns.stats(pelt_filtered, stat = "count", kernel = int(self.neighbor_params['neighbor_radius']), variable = 'date')

        inputImg = pelt_filtered.magnitude.to_dataset(name = 'magnitude', promote_attrs = True)
        inputImg[['ngbh_stdev']] = date_std
        inputImg[['ngbh_count']] = date_cnt
        inputImg.attrs = ds.attrs
        for var in inputImg.variables:
            inputImg[var].attrs = ds.attrs
        for var in pelt_filtered.variables:
            pelt_filtered[var].attrs = ds.attrs

        inputImg = inputImg.compute()
        pelt_filtered = pelt_filtered.compute()
        
        return inputImg, pelt_filtered

    def run_textures(self, inputImg: Dataset, pelt_filtered: Dataset) -> (Dataset, DataArray):

        target_chunk_size = 1024
        x_chunk = calc_chunk(inputImg.magnitude.shape[1],target_chunk_size)
        y_chunk = calc_chunk(inputImg.magnitude.shape[0],target_chunk_size)
        inputImg = inputImg.chunk({'x': x_chunk, 'y': y_chunk})
        glcm_radius = int(self.texture_params['glcm_radius'])
        levels = 2**3.
        distances = [1, 4, 7]
        angles = [0*(math.pi/4), 1*(math.pi/4), 2*(math.pi/4), 3*(math.pi/4)]

        r15_circle_kernel = skernel.circle(int(self.texture_params['smooth_radius']))
        focal =  sns.stats(inputImg, stat = "mean", kernel = r15_circle_kernel, variable = 'magnitude')

        pelt_filtered_magnitude_smooth = focal.where(abs(inputImg.magnitude - focal) > 0.2, other=inputImg.magnitude)

        if (pelt_filtered_magnitude_smooth.isnull().any().data):
            new_levels = int(levels + 1)
        else:
            new_levels = int(levels)

        data3 = simages.xr_transform(pelt_filtered_magnitude_smooth, levels = levels, dtype = 'uint8')

        darr = data3.chunk({'x':x_chunk,'y':y_chunk})
        nmetrics = 7
        chunks_ = tuple([(nmetrics,)] + list(darr.chunks))

        mb_kwargs = {
            'distances' : distances,
            'angles' : angles,
            'levels' : new_levels,
            'symmetric' : True,
            'normed' : True,
            'skip_nan' : True,
            'nan_supression' : 3,
            'rescale_normed' : True,
        }

        glcm = sglcm.glcm_textures(darr, radius = glcm_radius, n_feats = 7, **mb_kwargs)
    
        glcm = glcm.compute()
        
        return inputImg, glcm
    
    def run_rf(self, inputImg: Dataset, pelt_filtered: Dataset) -> (Dataset, Dataset, Dataset):
        rf_model = self.rf_params["rf_model"]
        prefix = Path(self.output["prefix"])
        s3_download_file(str(prefix / rf_model),self.output['bucket'],self.temp_dir.name)
        nname = rf_model.split(".")[0]
        classifier = joblib.load(self.temp_dir.name + '/' + rf_model)

        inputImgf = inputImg.chunk({"x": 500, "y": 500})
        classified = predict_xr(model = classifier, 
                                input_xr = inputImgf, 
                                clean=True).rename(name_dict = {"Predictions": "y_predict"})#.compute()
        classified.attrs = inputImg.attrs
        classified=classified.chunk({'x':500,'y':500})

        # filtrar por landcover especial y alinear a classified
        s3_download_file(str(prefix / 'LC_union_4y7_cog.tif'),self.output['bucket'],self.temp_dir.name)
        lc_ = xr.load_dataset(self.temp_dir.name+"/LC_union_4y7_cog.tif", engine="rasterio").band_data.squeeze()
        lc = lc_.rio.reproject(inputImg.rio.crs,
                            transform=inputImg.rio.transform(),
                            shape=inputImg.rio.shape,
                            resampling = Resampling.nearest)

        ## Lugares con cambios
        sam_bool = xr.where((inputImg.magnitude.notnull()) & (lc.astype(bool)), classified.y_predict, 255).astype(dtype = "uint8").rio.write_crs("epsg:32619", inplace=True).compute()

        ## Magnitudes con cambios
        sam_mgs = xr.where((classified.y_predict == 1) & (lc.astype(bool)), inputImg.magnitude, np.nan).astype(dtype = "float32").rio.write_crs("epsg:32619", inplace=True).compute()

        ## Fechas con cambios
        sam_dates = xr.where((classified.y_predict == 1) & (lc.astype(bool)) & (inputImg.magnitude.notnull()), pelt_filtered.date, 0).astype(dtype = "uint32").rio.write_crs("epsg:32619", inplace=True).compute()

        return sam_bool, sam_mgs, sam_dates

    def download_folder(self, prod_dir: Path) -> None:
        """Download S3 files to `prod_dir`.

        Uses the bucket and prefix defined in `self.output`.
        """
        bucket = self.output["bucket"]
        prefix = Path(self.output["prefix"])
        
        prefix = str(prefix / prod_dir.relative_to(Path(self.temp_dir.name+'/outputs')))
        self._logger.info(f"    Downloading s3://{bucket}/{prefix} to {prod_dir}")
        self.s3_download_folder(
            prefix=prefix,
            bucket=bucket,
            path=str(prod_dir)
        )

    def upload_files(self, prod_dir: Path) -> None:
        """Upload local `prod_dir` and its contents to S3.

        Uses the bucket and prefix defined in `self.output`.
        """
        bucket = self.output["bucket"]
        prefix = Path(self.output["prefix"])
        for path in prod_dir.rglob("*"):
            if not path.is_file():
                continue
            key = str(prefix / path.relative_to(self.temp_dir.name + "/outputs"))
            self._logger.info(f"    Uploading {path} to s3://{bucket}/{key}")
            self.s3_upload_file(
                path=str(path),
                bucket=bucket,
                key=key,
            )

    def process_key(self, key: (int, int)) -> None:
        """Process some tiles."""
        self._logger.info(f"Processing {key}")

        t0 = datetime.datetime.now()
        dataset = self.load_from_grid(key)
        ds_product = dataset.product.compute()
        t1 = datetime.datetime.now()
        self._logger.info(f"Data load and initial processing for {key} took: {t1-t0} at {int((dataset.ndvi.shape[1]*dataset.ndvi.shape[2])/(t1-t0).total_seconds())} pixels per second")

        self._logger.info(f"Key {key} has shape {dataset.ndvi.shape}")
        # self._logger.debug(f"- Dataset dims: {dataset.dims}")
        
        xref = f"{key[0]:+04}".replace("+", "E").replace("-", "W")
        yref = f"{key[1]:+04}".replace("+", "N").replace("-", "S")
        cellref = yref + xref
        
        out_folder = Path(self.temp_dir.name) / 'outputs/predict'
        out_folder.mkdir(parents=True, exist_ok=True)
        pelt_out = out_folder / 'PELT' / cellref
        pelt_out.mkdir(parents=True, exist_ok=True)

        if self.compute_pelt == 'True':
            # self.output["prefix"] = self.output["prefix"] + "/" + t1.strftime('%Y%m%d_%H%M%S')
            t2 = datetime.datetime.now()
            self._logger.info(f"Starting PELT at {t1}")
            # All cloud-masking and scaling is done inside the run_pelt step below
            
            pelt_output = self.run_pelt(dataset.ndvi)
            pelt_output = pelt_output.compute()
            # pelt_output['date_original'] = pelt_output.date
            # pelt_output.date.data = spelt.datetime_to_timestamp(pelt_output.date.data.astype('datetime64[s]'))

            # wait(pelt_output) # this line isn't required if you save the result
            # pelt_output['product'] = dataset.product
            t3 = datetime.datetime.now()
            self._logger.info(f"Finished PELT at {t2}")
            self._logger.info(f"PELT processing for {key} took: {t3-t2}")

            # Write to disk
            self._logger.info(f"Pixels per second: {int((dataset.ndvi.shape[1]*dataset.ndvi.shape[2])/(t3-t2).total_seconds())}")
            self._logger.info("Writing pelt output to file")
            
            # Write list of products and dates
            # dataset.to_dataframe()['product'].to_csv(f"{str(pelt_out)}/product_date_list_{cellref}.csv")
            
            for bkp in pelt_output['bkp'].values:
                # write_cog(pelt_output.date_original.sel({'bkp': bkp}), fname=f"{str(pelt_out)}/peltd_id{self.id_:03}_{cellref}_bks_dim-bkp-orig-{bkp}.tif", overwrite=True, nodata=np.nan)
                write_cog(pelt_output.date.sel({'bkp': bkp}), fname=f"{str(pelt_out)}/peltd_id{self.id_:03}_{cellref}_bks_dim-bkp-{bkp}.tif", overwrite=True, nodata=np.nan)#.compute()
                write_cog(pelt_output.magnitude.sel({'bkp': bkp}), fname=f"{str(pelt_out)}/peltd_id{self.id_:03}_{cellref}_mgs_dim-bkp-{bkp}.tif", overwrite=True, nodata=np.nan)#.compute()
                # self.save(ds=ds, key=key, start=start, upload=False)
            if self.output["upload"] == "True":
                self.upload_files(pelt_out)
            self._logger.info("PELT computation complete")
        else:
            self.download_folder(pelt_out)
            fls = list(pelt_out.glob("**/*"))
            fls.sort()
            mgs_ = [rioxarray.open_rasterio(f) for f in fls if "mgs" in f.name]
            mgs = xr.concat(mgs_, dim="band").rename({"band": "bkp"}).transpose("y", "x", "bkp")
            bks_ = [rioxarray.open_rasterio(f) for f in fls if "bks" in f.name]
            bks = xr.concat(bks_, dim="band").rename({"band": "bkp"}).transpose("y", "x", "bkp")
            mgs["bkp"] = range(len(mgs.bkp))
            bks["bkp"] = range(len(bks.bkp))
            pelt_output = mgs.to_dataset(name = "magnitude")
            pelt_output["date"] = bks

            pelt_output.attrs = dataset.attrs
            
            if len(pelt_output.bkp) != int(self.pelt_params['n_breaks']):
                self._logger.error("The loaded data does not have the expected number of breakpoints.")
                raise
            self._logger.info("Pelt files loaded from S3")

        for way in self.neighbor_params['ways']:
            print(f"Polygon: {self.id_}, using: {way}")

            neighs_out = out_folder / 'NEIGHS' / way / cellref
            neighs_out.mkdir(parents=True, exist_ok=True)
            
            t4 = datetime.datetime.now()
            
            if self.compute_neighbors == "True":
                inputImg, pelt_filtered = self.run_neighbors(ds=pelt_output)
                t5 = datetime.datetime.now()
                self._logger.info(f"Computing neighbors for {key} took: {t5-t4}")
                
                write_cog(inputImg.magnitude, fname = neighs_out / f'neighs_id{self.id_:03}_{cellref}_magnitude.tif', overwrite=True, nodata=np.nan)
                write_cog(inputImg.ngbh_stdev, fname = neighs_out / f'neighs_id{self.id_:03}_{cellref}_ngbh_stdev.tif', overwrite=True, nodata=np.nan)
                write_cog(inputImg.ngbh_count, fname = neighs_out / f'neighs_id{self.id_:03}_{cellref}_ngbh_count.tif', overwrite=True, nodata=np.nan)
                write_cog(pelt_filtered.date, fname = neighs_out / f'filtered_id{self.id_:03}_{cellref}_date.tif', overwrite=True, nodata=np.nan)
                write_cog(pelt_filtered.magnitude, fname = neighs_out / f'filtered_id{self.id_:03}_{cellref}_magnitude.tif', overwrite=True, nodata=np.nan)
                if self.output["upload"] == "True":
                    self.upload_files(neighs_out)
                self._logger.info("Neighbor computation complete")
            else:
                self.download_folder(neighs_out)
                inputImg = rioxarray.open_rasterio(neighs_out / f'neighs_id{self.id_:03}_{cellref}_magnitude.tif') \
                    .squeeze(drop=True) \
                    .to_dataset(name = 'magnitude', promote_attrs = True)
                inputImg[['ngbh_stdev']] = rioxarray.open_rasterio(neighs_out / f'neighs_id{self.id_:03}_{cellref}_ngbh_stdev.tif') \
                                                .squeeze(drop=True)
                inputImg[['ngbh_count']] = rioxarray.open_rasterio(neighs_out / f'neighs_id{self.id_:03}_{cellref}_ngbh_count.tif') \
                                                .squeeze(drop=True)
                inputImg.attrs = dataset.attrs
                for var in inputImg.variables:
                    inputImg[var].attrs = dataset.attrs

                pelt_filtered = rioxarray.open_rasterio(neighs_out / f'filtered_id{self.id_:03}_{cellref}_magnitude.tif') \
                    .squeeze(drop=True) \
                    .to_dataset(name = 'magnitude', promote_attrs = True)
                pelt_filtered[["date"]] = rioxarray.open_rasterio(neighs_out / f'filtered_id{self.id_:03}_{cellref}_date.tif') \
                    .squeeze(drop=True)
                pelt_filtered.attrs = dataset.attrs
                for var in pelt_filtered.variables:
                    pelt_filtered[var].attrs = dataset.attrs
                self._logger.info("Neighbor files loaded from S3")

            glcm_out = out_folder / 'GLCM' / way / cellref
            glcm_out.mkdir(parents=True, exist_ok=True)
            
            t6 = datetime.datetime.now()
            if self.compute_textures == 'True':
                inputImg, glcm = self.run_textures(inputImg, pelt_filtered)
                t7 = datetime.datetime.now()
                self._logger.info(f"Computing textures for {key} took: {t7-t6}")

                for prop in glcm['prop'].values:
                    write_cog(glcm.sel({'prop': prop}), fname=f"{str(glcm_out)}/glcm_id{self.id_:03}_{cellref}_dim-prop-{prop}.tif", overwrite=True, nodata=np.nan)
                
                if self.output["upload"] == "True":
                    self.upload_files(glcm_out)

                # simages.write_to_cogs(glcm, dim = "prop", fname = glcm_out / f"glcm_id{self.id_:03}_{cellref}_dim-prop-{prop}.tif")
                inputImg = xr.merge([inputImg, glcm.to_dataset(dim="prop", promote_attrs=True)])
                for var in inputImg.variables:
                    inputImg[var].attrs = dataset.attrs
                self._logger.info("Texture computation complete")
            else:
                textures_names = ['asm', 'contrast', 'corr', 'var', 'idm', 'savg', 'entropy']
                textures_names.sort()
                self.download_folder(glcm_out)
                fls = list(glcm_out.glob("**/*"))
                fls.sort()
                tex_ = [rioxarray.open_rasterio(f) for f in fls if f.name.endswith(".tif")]
                glcm = xr.concat(tex_, dim="band").rename({"band": "prop"}).transpose("y", "x", "prop")
                glcm["prop"] = textures_names
                inputImg = xr.merge([inputImg, glcm.to_dataset(dim="prop", promote_attrs=True)])
                for var in inputImg.variables:
                    inputImg[var].attrs = dataset.attrs
                self._logger.info("Texture files loaded from S3")

            inputImg = inputImg.compute()
            rf_out = out_folder / 'RF' / way / cellref
            rf_out.mkdir(parents=True, exist_ok=True)
            t8 = datetime.datetime.now()

            if self.compute_rf == "True":      
                sam_bool, sam_mgs, sam_dates = self.run_rf(inputImg, pelt_filtered)
                t9 = datetime.datetime.now()
                self._logger.info(f"Computing random forest for {key} took: {t9-t8}")

                # Remove extra buffer pixels before saving
                pad = int(self.tile_buffer[0]/30)
                sam_bool = sam_bool.isel(x=slice(pad,-pad),y=slice(pad,-pad))
                sam_mgs = sam_mgs.isel(x=slice(pad,-pad),y=slice(pad,-pad))
                sam_dates = sam_dates.isel(x=slice(pad,-pad),y=slice(pad,-pad))

                # Put the product list into its own dataset
                products=ds_product.to_dataset(name='product')
                # Add simple integer product numbering
                products['product_num'] = xr.where(products.product=='landsat8_c2l2_sr',9,np.nan)
                products['product_num'] = xr.where(products.product=='landsat8_c2l2_sr',8,products.product_num)
                products['product_num'] = xr.where(products.product=='landsat7_c2l2_sr',7,products.product_num)
                products['product_num'] = xr.where(products.product=='landsat5_c2l2_sr',5,products.product_num)
                
                # Match the dates to find the satellite product for each pixel and get rid of any unnecessary dimensions and variables
                sam_products = products.product_num.where(((products.time.astype(int)*1e-9).astype(int) == sam_dates)).max('time').squeeze().drop_vars('band')

                nname = self.rf_params['rf_model'].split('.')[0]
                write_cog(
                    geo_im = sam_bool,
                    fname = rf_out / f"sam_bool_{nname}_{way}_{cellref}.tif",
                    overwrite = True,
                    nodata = 255,
                    compress='LZW'
                )

                write_cog(
                    geo_im = sam_mgs,
                    fname = rf_out / f"sam_mgs_{nname}_{way}_{cellref}.tif",
                    overwrite = True,
                    nodata = np.nan,
                    compress='LZW'
                )

                write_cog(
                    geo_im = sam_dates,
                    fname = rf_out / f"sam_dates_{nname}_{way}_{cellref}.tif",
                    overwrite = True,
                    nodata = 0,
                    compress='LZW'
                )

                write_cog(
                    geo_im = sam_products,
                    fname = rf_out / f"sam_products_{nname}_{way}_{cellref}.tif",
                    overwrite = True,
                    nodata = 0,
                    compress='LZW'
                )

                if self.output["upload"] == "True":
                    self.upload_files(rf_out)
                self._logger.info("RF computation complete")

        self._logger.info(f"    Done. {key} with shape {dataset.ndvi.shape} took: {datetime.datetime.now()-t0}")

    def process_tile(self) -> None:
        """Process all tiles associated with the keys for this processor."""
        for log in ['distributed', 'distributed.nanny','distributed.scheduler','distributed.client']:
            logger = logging.getLogger(log)
            logger.setLevel(logging.ERROR)

        if (self.compute_pelt == 'False' and self.compute_neighbors == 'False' and self.compute_textures == 'False' and self.compute_rf == 'False'):
            self._logger.info(f"No processing required for {self.key[0]}")
            return

        self._logger.debug("Initialising local dask cluster")
        self.start_client()

        for key in self.key:
            self.process_key(key)

        self._logger.debug("Closing local dask cluster")
        self.close_client()
        # self.temp_dir.cleanup()

    def __repr__(self) -> str:
        """Information about this object."""
        return (
            f"{self.__class__.__name__}{{\n"
            f" - key: {self.key}\n"
            f" - product_cells count: {len(self.product_cells)}\n"
            f" - measurements: {self.measurements}\n"
            f" - output: {self.output}\n"
            "}"
        )
