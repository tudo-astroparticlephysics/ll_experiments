import click

from pyaml import yaml
import os

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

from gammapy.data import DataStore
from gammapy.maps import Map
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.spectrum import SpectrumExtraction

from regions import CircleSkyRegion

from plots import plot_counts

crab_position = SkyCoord(ra='83d37m59.0988s', dec='22d00m52.2s')
exclusion_map = Map.read(f"./data/exclusion_mask.fits.gz")


def create_data(input_dir, dataset_config):
    telescope = dataset_config['telescope']
    on_region_radius = dataset_config['on_radius']
    energy_bins = dataset_config['bins']
    containment = dataset_config['containment_correction']

    ds = DataStore.from_dir(os.path.join(input_dir, telescope))
    observations = ds.get_observations(ds.hdu_table['OBS_ID'].data)

    on_region = CircleSkyRegion(center=crab_position, radius=on_region_radius)

    print('Estimating Background')
    bkg_estimate = ReflectedRegionsBackgroundEstimator(
        observations=observations, on_region=on_region, exclusion_mask=exclusion_map
    )
    bkg_estimate.run()

    print('Extracting Count Spectra')
    extract = SpectrumExtraction(
        observations=observations,
        bkg_estimate=bkg_estimate.result,
        e_true=energy_bins,
        e_reco=energy_bins,
        containment_correction=containment,
        use_recommended_erange=False,  # TODO this might have to be checked.
    )
    extract.run()
    return extract


def create_energy_bins(tel_config):
    n_bins = tel_config['bins_per_decade']
    return np.logspace(-2, 2, (4 * n_bins) + 1) * u.TeV


def config(config_file):
    with open(config_file) as f:
        d = yaml.load(f)
        for tel_config in d['datasets']:

            d = {
                'telescope': tel_config['name'],
                'on_radius': tel_config['on_radius'] * u.deg,
                'containment_correction': tel_config['containment_correction'],
                'stack': tel_config.get('stack', False),
                'bins': create_energy_bins(tel_config),
                'fit_range': tel_config['fit_range'] * u.TeV,
            }

            yield d


def add_meta_information(observations, telescope, dataset_config):
    lo, hi = dataset_config['fit_range'].to_value('TeV')
    for obs in observations:
        obs.meta['TELESCOP'] = telescope
        obs.meta['TEL'] = telescope
        obs.meta['LO_RANGE'] = lo
        obs.meta['HI_RANGE'] = hi


@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('config_file', type=click.Path(dir_okay=False))
@click.argument('output_dir', type=click.Path(dir_okay=True))
@click.option('-t', '--telescope', type=click.Choice(['fact', 'hess', 'magic', 'veritas', 'all']), default='all', help='If given, will only extract data for that telescope.')
def extract(input_dir, config_file, output_dir, telescope):
    '''
    Take the DL3 data and create OGIP observations from it.
    The background is estimated using the reflected regions method.

    CONFIG_FILE should point to the yaml file contianing the
    dataset specific settings such as fit_range etc.

    INPUT_DIR is the folder containing the subfolders for
    each seperate telescope.
    '''

    for dataset_config in config(config_file):
        tel = dataset_config['telescope']
        if telescope != 'all' and tel != telescope:
            continue

        on_radius = dataset_config['on_radius']
        stack = dataset_config['stack']
        print(f'Extracting data for {tel} with radius {on_radius}. Stacking: {stack}')
        extracted_data = create_data(input_dir, dataset_config)

        output_path = os.path.join(output_dir, tel)
        os.makedirs(output_path, exist_ok=True)

        print(f'Writing data for {tel} to {os.path.join(output_dir, tel)}')
        if stack:
            print('Stacking observations')
            obs = extracted_data.spectrum_observations.stack()
            # we are writing a single observation, as for Fermi
            add_meta_information([obs], telescope, dataset_config)
            obs.write(output_path, use_sherpa=True, overwrite=True)
        else:
            add_meta_information(extracted_data.spectrum_observations, telescope, dataset_config)
            extracted_data.write(output_path, ogipdir="", use_sherpa=True, overwrite=True)

        fig, _ = plot_counts(output_path, extracted_data, tel)
        plt.savefig(os.path.join(output_path, 'counts.pdf'))


if __name__ == '__main__':
    extract()
