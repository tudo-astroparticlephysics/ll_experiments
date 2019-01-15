import click

from pyaml import yaml
import os

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

from gammapy.data import DataStore
from gammapy.spectrum import SpectrumObservationList
from gammapy.maps import Map
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.spectrum import SpectrumExtraction

from regions import CircleSkyRegion



crab_position = SkyCoord(ra='83d37m59.0988s', dec='22d00m52.2s')
exclusion_map = Map.read(f"./data/exclusion_mask.fits.gz")


def create_data(input_dir, dataset_config):
    telescope = dataset_config['telescope']
    on_region_radius = dataset_config['on_radius']
    energy_bins = dataset_config['bins']
    containment = dataset_config['containment_correction']

    ds = DataStore.from_dir(os.path.join(input_dir, telescope))
    observations = ds.obs_list(ds.hdu_table['OBS_ID'].data)

    on_region = CircleSkyRegion(center=crab_position, radius=on_region_radius)

    print('Estimating Background')
    bkg_estimate = ReflectedRegionsBackgroundEstimator(
        obs_list=observations, on_region=on_region, exclusion_mask=exclusion_map
    )
    bkg_estimate.run()

    # for obs in observations:
    #     if 'LO_THRESH' not in obs.aeff.meta:
    #         obs.aeff.meta['LO_THRESH'] = energy_bins.min()
    #     if 'HI_THRESH' not in obs.aeff.meta:
    #         obs.aeff.meta['HI_THRESH'] = energy_bins.max()

    print('Extracting Count Spectra')
    extract = SpectrumExtraction(
        obs_list=observations,
        bkg_estimate=bkg_estimate.result,
        e_true=energy_bins,
        e_reco=energy_bins,
        containment_correction=containment,
        use_recommended_erange=False, # TODO this might have to be checked.
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


def plot_counts(output_path, extracted_data, name):
    counts = []
    for obs in extracted_data.observations:
        c = obs.on_vector.counts_in_safe_range.value
        counts.append(c)

    counts = np.sum(counts, axis=0)
    x = extracted_data.observations[0].e_reco.lower_bounds.to_value('TeV')

    plt.title(f'Count Spectrum for {name} telescope.')
    plt.suptitle(f'total counts: {counts.sum()}')
    plt.step(x, counts, where='post', lw=2)
    plt.xscale('log')
    plt.xlabel('Energy')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(output_path, 'counts.pdf'))


def add_meta_information(observations, telescope, dataset_config):
    lo, hi = dataset_config['fit_range'].to_value('TeV')
    for obs in observations:
        obs.meta['TELESCOP'] = telescope
        obs.meta['TEL'] = telescope
        obs.meta['LO_THRES'] = lo
        obs.meta['HI_THRES'] = hi

        obs.aeff.meta['TELESCOP'] = telescope
        obs.aeff.meta['TEL'] = telescope
        obs.aeff.meta['LO_THRES'] = lo
        obs.aeff.meta['HI_THRES'] = hi


@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('config_file',type=click.Path(dir_okay=False))
@click.argument('output_dir', type=click.Path(dir_okay=True))
@click.option('-t', '--tel', default=None)
def extract(input_dir, config_file, output_dir, tel):
    '''
    Provide the input_dir to the folder containing the subfolders for
    each speerate telescopes.
    '''

    for dataset_config in config(config_file):
        telescope = dataset_config['telescope']
        if tel is not None and tel != telescope:
            continue

        on_radius = dataset_config['on_radius']
        stack = dataset_config['stack']
        print(f'Extracting data for {telescope} with radius {on_radius}. Stacking: {stack}')
        extracted_data = create_data(input_dir, dataset_config)

        output_path = os.path.join(output_dir, telescope)
        os.makedirs(output_path, exist_ok=True)

        print(f'Writing data for {telescope} to {output_dir}/{telescope}')
        if stack:
            print('Stacking observations')
            obs = extracted_data.observations.stack()

            # we are writing a single observation, as for Fermi
            add_meta_information([obs], telescope, dataset_config)
            obs.write(output_path, use_sherpa=True, overwrite=True)
        else:
            add_meta_information(extracted_data.observations, telescope, dataset_config)
            extracted_data.write(output_path, ogipdir="", use_sherpa=True, overwrite=True)
        plot_counts(output_path, extracted_data, telescope)

if __name__ == '__main__':
    extract()
