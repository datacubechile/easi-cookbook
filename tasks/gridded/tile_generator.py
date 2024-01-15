import pickle

import simplejson as json  # Needed to parse decimals in JSON w/o errors
from datacube import Datacube
from datacube.api import GridWorkflow
from datacube.model import GridSpec
from datacube.utils import geometry

from tasks.argo_task import ArgoTask
from tasks.common import process_order_params, validate_order


class TileGenerator(ArgoTask):
    FILEPATH_KEYS = "/tmp/keys.json"
    """Output path for the list of keys."""

    FILEPATH_CELLS = "/tmp/product_cells.pickle"
    """Output path for the gridded cells."""

    DEFAULT_ODC_QUERY = {
        "output_crs": "EPSG:3577",
        "resolution": [-30, 30],
        "group_by": "solar_day",
    }
    """Default ODC parameters, if the user doesn't specify them."""

    def __init__(self, input_params: [{str, str}]) -> None:
        """Check and cast input params as required."""
        super().__init__(input_params)
        self.size = float(self.size)  # Tile size in CRS units
        # Start from default values and update with whatever the user has set
        query = self.DEFAULT_ODC_QUERY.copy()
        query.update(self.odc_query)
        self.odc_query = query

    def generate_tiles(self) -> None:
        """Build a set of cells based on a grid workflow for the selected product.

        The user should have passed some `odc_query` params such as `output_crs`,
        `resolution`, `group_by`. This method updates them and creates a grid spec and
        cells corresponding to information passed in `roi`.
        """
        self._logger.info("Generating tiles...")
        # Validate params first, which also normalises dates among others
        valid, order_params = validate_order(self.roi, self.aws_region)
        if not valid:
            self._logger.error(order_params)
            raise RuntimeError(order_params)

        # Extract the query params from the order
        query_time, bbox, boundary = process_order_params(order_params, self.aws_region)
        self.odc_query["time"] = query_time

        if bbox:
            # Use CRS if defined in original roi dict
            if "crs" in self.roi:
                roi_crs = geometry.CRS(self.roi["crs"])
            else:
                roi_crs = geometry.CRS("EPSG:4326")
                self._logger.warning(f"Using default CRS {roi_crs} for bounding box")
            self.odc_query["latitude"] = (bbox[1], bbox[3])
            self.odc_query["longitude"] = (bbox[0], bbox[2])
        else:  # boundary: user-defined or from aoi
            try:
                # Use CRS if defined according to Geojson CRS specs
                roi_crs = geometry.CRS(boundary.crs["properties"]["name"])
            except (AttributeError, KeyError):
                # Else default to EPSG:4326, as used by aoi database (issued
                # from World Bank data)
                roi_crs = geometry.CRS("EPSG:4326")
                self._logger.warning(f"Using default CRS {roi_crs} for aoi/boundary")
            self.odc_query["geopolygon"] = geometry.Geometry(boundary, crs=roi_crs)

        gs_params = {
            "crs": self.odc_query["output_crs"],
            "tile_size": (self.size, self.size),
            "resolution": self.odc_query["resolution"],
        }
        self._logger.debug(f"Creating GridSpec with {gs_params}")
        gs = GridSpec(**gs_params)
        dc = Datacube(app="Time Series")
        gw = GridWorkflow(dc.index, grid_spec=gs)

        # Grab cell list and group in sublists of max size tiles_per_worker. Each
        # sublist will be sent to a separate Argo worker
        self._logger.debug(f"Creating cells for {self.product} with: {self.odc_query}")
        product_cells = gw.list_cells(product=self.product, **self.odc_query)
        unique_keys = list(set(product_cells.keys()))
        n = self.tiles_per_worker  # Max number of tiles to process per worker
        keys = [unique_keys[i : i + n] for i in range(0, len(unique_keys), n)]
        self._logger.info(
            f"{len(unique_keys)} keys to be processed by {len(keys)} workers"
        )
        # Saving the keys for Argo to split work
        with open(self.FILEPATH_KEYS, "w") as outfile:
            json.dump(keys, outfile)
        # Saving the product cells for workers; these are "dc.load-able" metadata
        with open(self.FILEPATH_CELLS, "wb") as outfile:
            pickle.dump(product_cells, outfile)


