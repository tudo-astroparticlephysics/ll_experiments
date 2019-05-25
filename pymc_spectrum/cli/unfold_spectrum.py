# import theano
import theano.tensor as T

import numpy as np
import pymc3 as pm
import yaml

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

from gammapy.data import DataStore
# from gammapy.maps import Map
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.spectrum import SpectrumExtraction

from regions import CircleSkyRegion

from ..plots import plot_unfolding_result, plot_excees
import click
import os
import shutil
from scipy.ndimage import laplace
from ..utils import display_data


def create_energy_bins(n_bins_per_decade=10, fit_range=None, overflow=False):
    bins = np.logspace(-2, 2, (4 * n_bins_per_decade) + 1)
    if fit_range is not None:
        bins = apply_range(bins, fit_range=fit_range, bins=bins * u.TeV)[0]
        bins = np.append(0.01, bins)
        bins = np.append(bins, 100)
    return bins * u.TeV


def apply_range(*arr, fit_range, bins):
    '''
    Takes one or more array-like things and returns only those entries
    whose bins lie within the fit_range.
    '''
    idx = np.searchsorted(bins.to_value(u.TeV), fit_range.to_value(u.TeV))
    lo, up = idx[0], idx[1] + 1
    return [a[lo:min(up, len(a) - 1)] for a in arr]


def load_data(input_dir, dataset_config, exclusion_map=None):
    on_region_radius = dataset_config['on_radius']
    e_reco_bins = dataset_config['e_reco_bins']
    e_true_bins = dataset_config['e_true_bins']
    containment = dataset_config['containment_correction']

    ds = DataStore.from_dir(input_dir)
    observations = ds.get_observations(ds.obs_table['OBS_ID'].data)
    # observations = ds.get_observations(ds.hdu_table['OBS_ID'].data)  # this is plenty wrong
    

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
        e_true=e_true_bins,
        e_reco=e_reco_bins,
        containment_correction=containment,
        use_recommended_erange=False,
    )
    extract.run()
    if dataset_config['stack']:
        return [extract.spectrum_observations.stack()]
    else:
        return extract.spectrum_observations


def thikonov(f, normalize=False):
    if normalize:
        f = f / f.sum()

    laplace_matrix = laplace(np.eye(len(f))) // 2

    a = T.dot(laplace_matrix, f)
    a_transposed = a.T
    return T.dot(a, a_transposed)


def transform(mu_s, aeff):
    return pm.math.log(mu_s / aeff)




def prepare_output(output_dir):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print('Overwriting previous results')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def calc_mu_b(mu_s, on_data, off_data, exposure_ratio):
    '''
    Calculate the value of mu_b given all other  parameters. See
    Dissertation of johannes king or gammapy docu on WSTAT.

    https://www.imprs-hd.mpg.de/267524/thesis_king.pdf

    https://docs.gammapy.org/0.8/stats/fit_statistics.html#poisson-data-with-background-measurement

    '''
    alpha = exposure_ratio
    c = alpha * (on_data + off_data) - (alpha + 1) * mu_s
    d = pm.math.sqrt(c**2 + 4 * (alpha + 1) * alpha * off_data * mu_s)
    mu_b = (c + d) / (2 * alpha * (alpha + 1))

    # mu_b  = T.where(on_data == 0, off_data/(alpha + 1), mu_b)
    # mu_b  = T.where(off_data == 0, on_data/(alpha + 1) - mu_s/alpha, mu_b)
    return mu_b


def get_observed_counts(observations, fit_range, bins):
    on_data = []
    off_data = []
    excess = []

    for observation in observations:
        on_data.append(observation.on_vector.data.data.value)
        off_data.append(observation.off_vector.data.data.value)
        excess.append(on_data[-1] - observation.alpha * off_data[-1])

    on_data = np.sum(on_data, axis=0)
    off_data = np.sum(off_data, axis=0)
    excess = np.sum(excess, axis=0)

    if len(observations) > 1:  # obs have not been stacked
        t = np.array([o.livetime.to_value('s') for o in observations])
        w = t / t.sum()
        mean_exposure_ratio = (w * [o.alpha for o in observations]).sum()
    else:
        mean_exposure_ratio = observations[0].alpha.mean()

    on_data, off_data = apply_range(on_data, off_data, fit_range=fit_range, bins=bins)
    return on_data, off_data, excess, mean_exposure_ratio


def load_config(config_file, telescope):
    with open(config_file) as f:
        d = yaml.load(f)
        source_pos = d['source_position']
        tel_config = d['datasets'][telescope]
        fit_range = tel_config['fit_range'] * u.TeV
        d = {
            'telescope': telescope,
            'on_radius': tel_config['on_radius'] * u.deg,
            'containment_correction': tel_config['containment_correction'],
            'stack': tel_config.get('stack', False),
            'fit_range': fit_range,
            'e_reco_bins': create_energy_bins(20, ),
            'e_true_bins': create_energy_bins(tel_config['bins_per_decade'], fit_range=fit_range),
            'source_position': SkyCoord(ra=source_pos['ra'], dec=source_pos['dec']),
        }

        return d


def forward_fold(counts, observations, fit_range, area_scaling=1):
    '''
    Forward fold the spectral model through the instrument functions given in the 'observations'
    Returns the predicted counts in each energy bin.
    '''
    # counts = T.exp(counts) 
    predicted_signal_per_observation = []
    for observation in observations:
        obs_bins = observation.on_vector.energy.bins.to_value(u.TeV)
        aeff = observation.aeff.data.data.to_value(u.km**2).astype(np.float32) * area_scaling
        e_true_bin_width = observation.e_true.diff().to_value('TeV')

        c = counts * aeff * e_true_bin_width * observation.livetime.to_value(u.s)

        edisp = observation.edisp.pdf_matrix
        predicted_signal_per_observation.append(T.dot(c, edisp))

    predicted_counts = T.sum(predicted_signal_per_observation, axis=0)

    idx = np.searchsorted(obs_bins, fit_range.to_value(u.TeV))
    predicted_counts = predicted_counts[idx[0]:idx[1] + 1]

    return predicted_counts


@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('config_file', type=click.Path(dir_okay=False))
@click.argument('output_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('dataset', type=click.Choice(['hess', 'fact', 'magic', 'veritas']))
@click.option('--model_type', default='full', type=click.Choice(['full', 'profile', 'wstat']))
@click.option('--tau', default=0)
@click.option('--n_samples', default=3000)
@click.option('--n_tune', default=1500)
@click.option('--target_accept', default=0.9)
@click.option('--n_cores', default=6)
@click.option('--seed', default=80085)
@click.option('--init', default='advi+adapt_diag', help='Set pymc sampler init string.')
def main(input_dir, config_file, output_dir, dataset, model_type, tau, n_samples, n_tune, target_accept, n_cores, seed, init):
    prepare_output(output_dir)

    config = load_config(config_file, dataset)
    fit_range = config['fit_range']
    path = os.path.join(input_dir, dataset)
    observations = load_data(path, config)
    e_true_bins = config['e_true_bins']
    # exposure_ratio = observations[0].alpha
    # from IPython import embed; embed()
    on_data, off_data, excess, exposure_ratio = get_observed_counts(observations, fit_range=fit_range, bins=config['e_reco_bins'])
    print(f'Exposure ratio {exposure_ratio}')
    # from IPython import embed; embed()
    # print(f'On Data {on_data.shape}\n')
    # display_data(on_data)
    # print(f'\n\n Off Data {off_data.shape}\n')
    # display_data(off_data)
    print(f'Excess {excess.shape} \n')
    display_data(excess)
    print('--' * 30)
    print(f'Unfolding data for:  {dataset.upper()}.  ')
    # print(f'IRF with {len( config['e_true_bins'] ) - 1, len( config['e_reco_bins'] ) - 1}')
    print(f'Using {len(on_data)} bins with { on_data.sum()} counts in on region and {off_data.sum()} counts in off region.')

    area_scaling = 1
    print(observations[0].aeff.data.data.to_value(u.km**2).astype(np.float32) * area_scaling)
    model = pm.Model(theano_config={'compute_test_value': 'ignore'})
    with model:
        # mu_b = pm.TruncatedNormal('mu_b', shape=len(off_data), sd=5, mu=off_data, lower=0.01)
        # expected_counts = pm.Lognormal('expected_counts', shape=len(config['e_true_bins']) - 1, testval=10, sd=1)
        expected_counts = pm.TruncatedNormal('expected_counts', shape=len(e_true_bins) - 1, testval=0.5, mu=2, sd=50, lower=0.0001)
        # expected_counts = pm.HalfFlat('expected_counts', shape=len(e_true_bins) - 1, testval=10)
        # c = expected_counts / area_scaling
        mu_s = forward_fold(expected_counts, observations, fit_range=fit_range, area_scaling=area_scaling)

        if model_type == 'wstat':
            print('Building profiled likelihood model')
            mu_b = pm.Deterministic('mu_b', calc_mu_b(mu_s, on_data, off_data, exposure_ratio))
        else:
            print('Building full likelihood model')
            mu_b = pm.HalfFlat('mu_b', shape=len(off_data))
            # mu_b = pm.Lognormal('mu_b', shape=len(off_data), sd=5)


        pm.Poisson('background', mu=mu_b + 1E-5, observed=off_data)
        pm.Poisson('signal', mu=mu_s + exposure_ratio * mu_b + 1E-5, observed=on_data)

    print('--' * 30)
    print('Model debug information:')
    for RV in model.basic_RVs:
        print(RV.name, RV.logp(model.test_point))

    print('--' * 30)
    print('Sampling likelihood:')
    with model:
        trace = pm.sample(n_samples, cores=n_cores, tune=n_tune, init=init, seed=[seed] * n_cores)

    trace_output = os.path.join(output_dir, 'traces')
    print(f'Saving traces to {trace_output}')
    with model:
        pm.save_trace(trace, trace_output, overwrite=True)

    print('--' * 30)
    print('Plotting result')
    # print(area_scaling)

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    plot_unfolding_result(trace, bins=e_true_bins, area_scaling=area_scaling, fit_range=fit_range, ax=ax1)
    plot_excees(excess, bins=config['e_reco_bins'], ax=ax2)
    plt.savefig(os.path.join(output_dir, 'result.pdf'))


    print('--' * 30)
    print('Plotting Diagnostics')
    print(pm.summary(trace).round(2))
    # plt.figure()
    # pm.traceplot(trace)
    # plt.savefig(os.path.join(output_dir, 'traces.pdf'))

    plt.figure()
    pm.energyplot(trace)
    plt.savefig(os.path.join(output_dir, 'energy.pdf'))

    # try:
    #     plt.figure()
    #     pm.autocorrplot(trace, burn=n_tune)
    #     plt.savefig(os.path.join(output_dir, 'autocorr.pdf'))
    # except:
    #     print('Could not plot auttocorrelation')

    plt.figure()
    pm.forestplot(trace)
    plt.savefig(os.path.join(output_dir, 'forest.pdf'))
    


if __name__ == '__main__':
    main()
