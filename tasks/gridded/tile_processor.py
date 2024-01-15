"""
Example of geomedian calculation for a tile, using a local dask cluster.
"""

import logging
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from dask.distributed import Client, LocalCluster, wait
from datacube.api import GridWorkflow
from datacube.utils.masking import make_mask, mask_invalid_data
from eodatasets3 import DatasetPrepare
from odc.algo import from_float, to_f32, xr_geomedian
from tasks.argo_task import ArgoTask
from tasks.gridded.tile_generator import TileGenerator
from xarray import Dataset
from datacube.utils.rio import configure_s3_access

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
        self.measurements = [] if self.measurements == "" else self.measurements

        # Convert key from list[list] to list[tuple]
        self.key = [tuple(k) for k in self.key]

        self._client = None
        self._cluster = None
        self._nworkers = 4

        # Unpickle the product cells from file
        with open(TileGenerator.FILEPATH_CELLS, "rb") as fh:
            self.product_cells = pickle.load(fh)

    def start_client(self) -> None:
        """Start a local dask cluster, if needed."""
        if self._client is None:
            self._cluster = LocalCluster(n_workers=self._nworkers)
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
        cell = self.product_cells.get(key)
        # All geoboxes for the tiles are the same shape. Use this for the chunk size in
        # dask so each tile spatially is a single chunk. Note that the geobox resolution
        # is in (y, x) order
        chunk_dim = cell.geobox.shape
        chunks = {"time": 10, "x": chunk_dim[1], "y": chunk_dim[0]}
        try:
            ds = GridWorkflow.load(
                cell,
                measurements=self.measurements,
                dask_chunks=chunks,
                # Ignore exceptions until re-indexer is running regularly to
                # correct for reprocessed data
                skip_broken_datasets=True,
            )
            self._logger.debug(f"Dataset for key {key} has dims: {str(ds.dims)}")
        except Exception as e:
            self._logger.error(f"load_from_grid: Exception: {e}")
            raise
        return ds

    def mask(self, ds: Dataset) -> Dataset:
        """Mask clouds and no data in data based on `oa_fmask` values."""
        # Use datacube masking methods
        # https://docs.dea.ga.gov.au/notebooks/How_to_guides/Masking_data.html
        cloud_free_mask = make_mask(
            ds.pixel_qa, water="land_or_cloud", clear="clear", nodata=False
        )

        # Set all nodata pixels to `NaN`:
        # float32 has sufficient precision for original uint16 SR_bands and saves memory
        cloud_free = mask_invalid_data(
            ds[["red", "green", "blue"]].astype("float32", casting="same_kind")
        )  #  remove the invalid data on Surface reflectance bands prior to masking clouds
        cloud_free = cloud_free.where(cloud_free_mask)

        return cloud_free

    def scale(self, ds: Dataset) -> Dataset:
        """Scale Landsat data before geomedian calculation.

        GA Landsat data is in the range [0, 10000] and needs scaling to [0.0,
        1.0] to calculate the geomedian.
        """
        return to_f32(ds, scale=self.SCALE, offset=-0.2)

    def calculate_geomedian(self, ds: Dataset) -> Dataset:
        """Calculate the geomedian and cast it to `int16`."""
        geomedian = xr_geomedian(
            ds,
            # disable internal threading, dask will run several concurrently
            num_threads=1,
            # Epsilon: 1/5 pixel value resolution. Undocumented in hdstats
            eps=0.2 * self.SCALE,
            # Disable checks that use too much ram
            nocheck=True,
        )
        geomedian = from_float(
            geomedian,
            dtype="int16",
            nodata=-999,
            scale=1 / self.SCALE,
            offset=0,
        )
        return geomedian

    def _save_cog(
        self,
        ds: Dataset,
        prod_dir: Path,
        key: (int, int),
        start: pd.Timestamp,
    ) -> None:
        """Save dataset to local `prod_dir` as datacube indexable COGS."""
        # Set timestamp as last day of month
        ts = start + self.ONE_MONTH - self.ONE_DAY
        metadata_path = prod_dir / "odc-metadata.yaml"
        with DatasetPrepare(metadata_path=metadata_path) as p:
            p.product_family = self.PRODUCT
            p.label = prod_dir.name
            p.datetime = ts.to_pydatetime()
            p.processed_now()
            for measurement in ds.keys():
                path = prod_dir / f"{prod_dir.name}_{measurement}.tif"
                self._logger.debug(f"    Saving COG to {path}")
                ds[measurement].rio.to_raster(path)
                p.note_measurement(
                    measurement.lower(), path.name, relative_to_dataset_location=True
                )
            p.properties["odc:file_format"] = "GeoTIFF"
            p.done(validate_correctness=False)
            self._logger.debug(f"    Saving metadata to {metadata_path}")

    def _save_zarr(
        self,
        ds: Dataset,
        prod_dir: Path,
        key: (int, int),
        start: pd.Timestamp,
    ) -> None:
        """Save dataset to local `prod_dir` as zarr."""
        path = prod_dir / f"{prod_dir.name}.zarr"
        self._logger.debug(f"    Saving {path}")
        ds.to_zarr(store=path)

    def upload_files(self, prod_dir: Path) -> None:
        """Upload local `prod_dir` and its contents to S3.

        Uses the bucket and prefix defined in `self.output`.
        """
        bucket = self.output["bucket"]
        prefix = Path(self.output["prefix"])
        for path in prod_dir.rglob("*"):
            if not path.is_file():
                continue
            key = str(prefix / path.relative_to(prod_dir.parent))
            self._logger.debug(f"    Uploading {path} to s3://{bucket}/{key}")
            self.s3_upload_file(
                path=str(path),
                bucket=bucket,
                key=key,
            )

    def save(
        self,
        ds: Dataset,
        key: (int, int),
        start: pd.Timestamp,
        storage: str = "cog",
    ) -> None:
        """Save dataset to cog or zarr in S3.

        The `key` and `start` date are used to format the product name based on
        `PRODUCT_TEMPLATE`. The files are saved to a local temporary directory before
        being uploaded to S3 in the bucket and prefix specified in `self.output`.
        """
        prod_name = self.PRODUCT_TEMPLATE.format(
            product=self.PRODUCT,
            xref=f"{key[0]:+04}".replace("+", "E").replace("-", "W"),
            yref=f"{key[1]:+04}".replace("+", "N").replace("-", "S"),
            year=start.year,
            month=start.month,
        )
        with TemporaryDirectory() as tmpdirname:
            prod_dir = Path(tmpdirname) / prod_name
            prod_dir.mkdir()
            if storage.lower() == "cog":
                self._save_cog(ds=ds, prod_dir=prod_dir, key=key, start=start)
            elif storage.lower() == "zarr":
                self._save_zarr(ds=ds, prod_dir=prod_dir, key=key, start=start)
            else:
                raise ValueError(f"Unknown storage: {storage}")
            self.upload_files(prod_dir)

    def process_key(self, key: (int, int)) -> None:
        """Process some tiles."""
        self._logger.info(f"Processing {key}")
        dataset = self.load_from_grid(key)
        self._logger.debug(f"- Dataset dims: {dataset.dims}")
        # Gather start and end times keeping year-month info only
        times = dataset.time.to_pandas().index.to_period("M").to_timestamp()
        # Create monthly date range using start of month (MS)
        dr = pd.date_range(times[0], times[-1], freq="MS")

        # Process geomedian month by month
        for start in dr:
            self._logger.info(f"  - Month starting on {start}")
            ds = dataset.sel(time=slice(start, start + self.ONE_MONTH))
            ds = self.mask(ds)
            ds = self.scale(ds)
            # ! For the sake of calculation time we'll naively use median rather than geomedian
            #  ds = self.calculate_geomedian(ds)
            ds = ds.median(dim='time')
            ds = ds.persist()
            # TODO uncomment the self.save line to save the results to s3 Storage
            wait(ds) # this line isn't required if you save the result
            # self.save(ds=ds, key=key, start=start)
            self._logger.debug("    Done.")

    def process_tile(self) -> None:
        """Process all tiles associated with the keys for this processor."""
        for log in ['distributed', 'distributed.nanny','distributed.scheduler','distributed.client']:
            logger = logging.getLogger(log)
            logger.setLevel(logging.ERROR)

        self._logger.debug("Initialising local dask cluster")
        self.start_client()

        for key in self.key:
            self.process_key(key)

        self._logger.debug("Closing local dask cluster")
        self.close_client()

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
