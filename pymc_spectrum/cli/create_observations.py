import click

from pyaml import yaml
import os

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

from gammapy.data import DataStore
# from gammapy.maps import Map
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.spectrum import SpectrumExtraction

from regions import CircleSkyRegion

from ..plots import plot_counts


def create_data(input_dir, dataset_config, exclusion_map=None):
    telescope = dataset_config['telescope']
    on_region_radius = dataset_config['on_radius']
    energy_bins = dataset_config['e_reco_bins']
    containment = dataset_config['containment_correction']

    ds = DataStore.from_dir(os.path.join(input_dir, telescope))
    observations = ds.get_observations(ds.obs_table['OBS_ID'].data)
    t_obs = sum([o.observation_live_time_duration for o in observations])
    # from IPython import embed; embed()
    print(f'Total obstime for {telescope} is {t_obs.to("h")}')
    source_position = dataset_config['source_position']
    on_region = CircleSkyRegion(center=source_position, radius=on_region_radius)

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


def load_config(telescope, config_file):
    with open(config_file) as f:
        d = yaml.load(f)
        source_pos = d['source_position']
        tel_config = d['datasets'][telescope]
        n_bins_per_decade = tel_config['bins_per_decade']
        d = {
            'telescope': telescope,
            'on_radius': tel_config['on_radius'] * u.deg,
            'containment_correction': tel_config['containment_correction'],
            'stack': tel_config.get('stack', False),
            'fit_range': tel_config['fit_range'] * u.TeV,
            'e_reco_bins': np.logspace(-2, 2, (4 * n_bins_per_decade) + 1) * u.TeV,
            'e_true_bins': np.logspace(-2, 2, (4 * n_bins_per_decade) + 1) * u.TeV,
            'source_position': SkyCoord(ra=source_pos['ra'], dec=source_pos['dec']),
        }

        return d


def add_meta_information(observations, telescope, dataset_config):
    '''
    This propertiy will be added to the hdus of the *pha.fits files.
    They are read by the loop in `get_fit_settings` and then set to be meta data for 
    the observations again. I considers this an ugly workaround to gammapy behaviour which I think is a bug.
    '''
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
def main(input_dir, config_file, output_dir, telescope):
    '''
    Take the DL3 data and create OGIP observations from it.
    The background is estimated using the reflected regions method.

    CONFIG_FILE should point to the yaml file containing the
    dataset specific settings such as fit_range etc.

    INPUT_DIR is the folder containing the subfolders for
    each seperate telescope.
    '''

    telescopes = ['magic', 'fact', 'veritas', 'hess']
    for tel in telescopes:
        dataset_config = load_config(tel, config_file)

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
    main()
