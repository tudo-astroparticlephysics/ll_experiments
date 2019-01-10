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


def create_data(telescope, on_region_radius, containment, energy_bins):
    ds = DataStore.from_dir(f'./data/{telescope}')
    observations = ds.obs_list(ds.hdu_table['OBS_ID'].data)

    on_region = CircleSkyRegion(center=crab_position, radius=on_region_radius)
    
    print('Estimating Background')
    bkg_estimate = ReflectedRegionsBackgroundEstimator(
        obs_list=observations, on_region=on_region, exclusion_mask=exclusion_map
    )
    bkg_estimate.run()

    print('Extracting Count Spectra')
    extract = SpectrumExtraction(
        obs_list=observations,
        bkg_estimate=bkg_estimate.result,
        e_true=energy_bins,
        e_reco=energy_bins,
        containment_correction=containment,
        use_recommended_erange=True,
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
            telescope = tel_config['name']
            on_radius = tel_config['on_radius'] * u.deg
            containment = tel_config['containment_correction']
            bins = create_energy_bins(tel_config)
            yield (telescope, on_radius, containment, bins)

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
    

@click.command()
@click.argument('config_file',type=click.Path(dir_okay=False))
@click.argument('output_dir', type=click.Path(dir_okay=True))
def extract(config_file, output_dir):
    
    for telescope, on_radius, containment, energy_bins in config(config_file):
        print(telescope, on_radius)
        extracted_data = create_data(telescope, on_radius, containment, energy_bins)
        
        output_path = os.path.join(output_dir, telescope)
        os.makedirs(output_path, exist_ok=True)
        
        print(f'Writing data for {telescope} to {output_dir}')
        if telescope in ['fact', 'veritas']:
            # For FACT the IRFs are the same for all observations
            # So we only store a stacked spectrum and response
            # plus we add a LO_THRESHOLD keyword was missing
            obs = extracted_data.observations.stack()
#             obs.lo_threshold = energy_bins.min()
#             obs.hi_threshold = energy_bins.max()
            # we are writing a single observation, as for Fermi
            obs.write(output_path, use_sherpa=True, overwrite=True)
        else:
#             for obs in extracted_data.observations:
#                 obs.lo_threshold = energy_bins.min()
#                 obs.hi_threshold = energy_bins.max()

            extracted_data.write(output_path, ogipdir="", use_sherpa=True, overwrite=True)
    
        plot_counts(output_path, extracted_data, telescope)

if __name__ == '__main__':
    extract()