import os
import sys
import uuid
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(1, '/home/jovyan/SAMSARA/lib-samsara/src')
sys.path.insert(1, 'lib-samsara/src/')
sys.path.insert(1, '.')
sys.path.insert(1,'/home/jovyan/CSIRO/easi-workflows/tasks/eo3assemble')
sys.path.insert(1,'/opt/repo/easiwf/easi-workflows/tasks/eo3assemble')

import xarray as xr
import pandas as pd

from dask.distributed import Client, LocalCluster
from datacube import Datacube
from tasks.argo_task import ArgoTask
from tasks import samsara_prepare
from datacube.utils.rio import configure_s3_access
import rioxarray

import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

UUID_NAMESPACE = uuid.UUID("2c5ae732-3fb5-4bcd-b985-045c811ddaa6") 

class Summarise(ArgoTask):
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

    def upload_files(self, prod_dir: Path) -> None:
        """Upload local `prod_dir` and its contents to S3.

        Uses the bucket and prefix defined in `self.output`.
        """
        bucket = self.output["bucket"]
        prefix = Path(self.output["prefix"])
        for path in prod_dir.rglob("*"):
            if not path.is_file():
                continue
            key = str(prefix / path.relative_to(self.temp_dir.name))
            self._logger.info(f"    Uploading {path} to s3://{bucket}/{key}")
            self.s3_upload_file(
                path=str(path),
                bucket=bucket,
                key=key,
            )

    def summarise(self) -> None:
        for log in ['rasterio', 'distributed', 'distributed.nanny','distributed.scheduler','distributed.client']:
            logger = logging.getLogger(log)
            logger.setLevel(logging.ERROR)
        """Summarise the data."""

        self.start_client()

        dc = Datacube()
        ds = dc.load(**self.odc_query)
        ds = ds.mag.where(~ds.mag.isnull())

        blocks = int(self.summary_grid_size / abs(self.odc_query['resolution'][0]))

        coarsened_sum = ds.coarsen(x=blocks, boundary="pad").sum().coarsen(y=blocks, boundary="pad").sum()
        coarsened_sum = coarsened_sum.where(~coarsened_sum.isnull())
        coarsened_sum_agg = coarsened_sum.groupby('time.year').sum().rename('mag_total')
        coarsened_sum_agg = coarsened_sum_agg.where(coarsened_sum_agg != 0)

        data_for_count = xr.where(~ds.isnull(),1,0).astype('int16')
        coarsened_count = data_for_count.coarsen(x=blocks, boundary="pad").sum().coarsen(y=blocks, boundary="pad").sum()
        coarsened_count = coarsened_count.where(~coarsened_count.isnull())
        coarsened_count_agg = coarsened_count.groupby('time.year').sum().rename('mag_count')
        coarsened_count_agg = coarsened_count_agg.where(coarsened_count_agg != 0)

        dataset = coarsened_sum_agg.to_dataset()
        dataset['mag_count'] = coarsened_count_agg.astype('int16')

        self._logger.info('Computing summary')
        dataset = dataset.compute()
        data_sum = dataset.mag_total
        data_count = dataset.mag_count

        product = self.new_product

        times = len(data_sum.year)

        for index, t in enumerate(data_sum.year.values):
            ts = pd.to_datetime(str(t))
            d = ts.strftime('%Y')
            fname = d
            f_dir = Path(self.temp_dir.name) / fname
            metadata_path = f_dir / 'odc-metadata.yaml'
            fname = product.lower()+"_"+d

            if not f_dir.exists():
                os.makedirs(f_dir, exist_ok=True)

            self._logger.info(f'Exporting raster: {f_dir.stem}')
            data_sum.sel(year=t).rio.to_raster(str(f_dir / fname) + "_" + data_sum.name.lower() + ".tif")
            data_count.sel(year=t).rio.to_raster(str(f_dir / fname) + "_" + data_count.name.lower() + ".tif")

            self._logger.info(f'Preparing metadata: {f_dir.stem}')
            r = samsara_prepare.prepare_samsara_summary(f_dir)
            if not r:
                self._logger.error(f"Failed to prepare {f_dir.stem}")
                continue

            self._logger.info(f'Finished summarising timestep {index+1} of {len(data_sum.year)}')
        if self.output['upload']:
            self.upload_files(Path(self.temp_dir.name))

        self._logger.info('Done summarising')
        self.close_client()
