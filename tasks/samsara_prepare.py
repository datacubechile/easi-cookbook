import click
from pathlib import Path
import pandas as pd
from datetime import datetime
from eodatasets3 import DatasetPrepare
from eodatasets3.images import ValidDataMethod
import uuid
import re
import sys
sys.path.insert(1,'/home/jovyan/CSIRO/easi-workflows/tasks/eo3assemble')
sys.path.insert(1,'/opt/repo/easiwf/easi-workflows/tasks/eo3assemble')

from easi_assemble import EasiPrepare

SAMSARA_RAW_UUID_NAMESPACE = uuid.UUID("e05aa9a6-dedb-405a-8146-53329bbb2a7a") 
SAMSARA_SUMMARY_UUID_NAMESPACE = uuid.UUID("2c5ae732-3fb5-4bcd-b985-045c811ddaa6")

def prepare_samsara_raw(dir):
    product = 'samsara_raw'
    # print(dir)
    f_dir = Path(dir)
    metadata_path = f_dir / 'odc-metadata.yaml'

    # files = {
    #     'bool':'clasified_bool_LC.tif',
    #     'dates':'clasified_dates_LC.tif',
    #     'change':'clasified_ndvi-neg-change_LC.tif'
    # }
    
    files = {
        # 'bool':'clasified_bool_LC.tif',
        'mag':next(f_dir.rglob('*_mag.tif'),None),
        'product': next(f_dir.rglob('*_product.tif'),None),
        'product_post': next(f_dir.rglob('*_product_post.tif'),None),
        'date_post': next(f_dir.rglob('*_date_post.tif'),None),
        'rep_1d': next(f_dir.rglob('*_rep_1d.tif'),None),
        'rep_60d': next(f_dir.rglob('*_rep_60d.tif'),None),
        'areas_protegidas': next(f_dir.rglob('*_areas_protegidas.tif'),None),
        'sitios_prioritarios': next(f_dir.rglob('*_sitios_prioritarios.tif'),None),
        # 'change':'clasified_ndvi-neg-change_LC.tif'
    }

    if files['mag'] is not None:
        ts = pd.to_datetime('today')
        with EasiPrepare(
            dataset_path=f_dir,
            product_yaml=Path(__file__).parent / 'samsara_raw.yaml' # This should be in hte same directory as this file
        ) as p:
            dataset = Path(f_dir).name
            date = datetime(int(dataset[:4]),int(dataset[4:6]),int(dataset[6:8]),12) # Add 12 hours to make sure timezone mostly works
            ## IDs and Labels
            version = 'v04'
            unique_name = f"{Path(files['mag']).stem.replace('_mag','')}-{version}"  # Unique dataset name
            p.dataset_id = uuid.uuid5(SAMSARA_RAW_UUID_NAMESPACE, unique_name)  # Unique dataset UUID
            unique_name_replace = re.sub('\.', '_', unique_name)
            p.label = f"{unique_name_replace}-{p.product_name}"  # Can not have '.' in label
            p.product_uri = f"https://products.datacubechile.cl/{p.product_name}"  # product_name is added by EasiPrepare().init()
            p.product_family = product
            p.producer = 'uai.cl'
            p.datetime = date# datetime(int(ts.strftime('%Y')),int(ts.strftime('%m')),int(ts.strftime('%d')))
            p.processed_now()
            p.dataset_version = version
            p.valid_data_method = ValidDataMethod.filled
            for key, file in files.items():
                if file is not None:
                    p.note_measurement(
                        key.lower(),
                        Path(f_dir / file),
                        relative_to_metadata = True
                    )
                else:
                    Exception(f"File {key} not found")
            
            p.properties["odc:file_format"] = "GeoTIFF"
            p.done(validate_correctness=False)

        # cmd = ['datacube', 'dataset', 'add', f'{f_dir}/odc-metadata.yaml' ]
        # subprocess.run(cmd, check=True)
        return f'{f_dir}/odc-metadata.yaml'

def prepare_samsara_summary(dir):
    product = 'samsara_summary'
    f_dir = Path(dir)
    metadata_path = f_dir / 'odc-metadata.yaml'

    ts = pd.to_datetime('today')
    with EasiPrepare(
        dataset_path=f_dir,
        product_yaml=Path(__file__).parent / 'samsara_summary.yaml' # This should be in hte same directory as this file
    ) as p:
        dataset = Path(f_dir).name
        file = list(f_dir.rglob('*.tif'))
        file = file[0] if len(file) == 1 else None
        if not file:
            return None
        date = datetime(int(dataset),12,31,12) # Add 12 hours to make sure timezone mostly works
        ## IDs and Labels
        version = 'v04'
        unique_name = f"{Path(file).stem.replace('_mag_total','')}-{version}"  # Unique dataset name
        p.dataset_id = uuid.uuid5(SAMSARA_SUMMARY_UUID_NAMESPACE, unique_name)  # Unique dataset UUID
        unique_name_replace = re.sub('\.', '_', unique_name)
        p.label = f"{unique_name_replace}-{p.product_name}"  # Can not have '.' in label
        p.product_uri = f"https://products.datacubechile.cl/{p.product_name}"  # product_name is added by EasiPrepare().init()
        p.product_family = product
        p.producer = 'uai.cl'
        p.datetime = date
        p.processed_now()
        p.dataset_version = version
        p.valid_data_method = ValidDataMethod.filled

        # for key in data.keys():
        p.note_measurement(
            'mag_total', Path(file),
            relative_to_metadata=True
        )
        p.properties["odc:file_format"] = "GeoTIFF"

        p.done(validate_correctness=False)

        return f'{f_dir}/odc-metadata.yaml'

@click.command()
@click.argument('dataset_dir', type=click.STRING)
@click.option('--product', type=click.STRING, default='samsara_raw')
def cli(dataset_dir, product):
    if product == 'samsara_raw':
        prepare_samsara_raw(dataset_dir)
    elif product == 'samsara_summary':
        prepare_samsara_summary(dataset_dir)

if __name__ == '__main__':
    cli()