# import theano
import theano.tensor as T

import numpy as np
import pymc3 as pm

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

from gammapy.data import DataStore
from gammapy.maps import Map
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.spectrum import SpectrumExtraction

from regions import CircleSkyRegion

from plots import plot_unfolding_result
import click
import os
import shutil


crab_position = SkyCoord(ra='83d37m59.0988s', dec='22d00m52.2s')
exclusion_map = Map.read('./data/exclusion_mask.fits.gz')

energy_range = {
    'fact': [0.52, 12] * u.TeV,
    'magic': [0.04, 8] * u.TeV,
    'veritas': [0.115, 6] * u.TeV,
    'hess': [0.35, 15] * u.TeV
}

on_radius = {
    'fact': 0.17 * u.deg,
    'magic': 0.142 * u.deg,
    'veritas': 0.10 * u.deg,
    'hess': 0.11 * u.deg,
}


def create_energy_bins(fit_range, n_bins_per_decade=10, overflow=False):
    bins = np.logspace(-2, 2, (4 * n_bins_per_decade) + 1)
    bins = apply_range(bins, fit_range=fit_range, bins=bins * u.TeV)[0]
    if overflow:
        bins = np.append(bins, 10000)
        bins = np.append(0, bins)
    return bins * u.TeV


def load_data(input_dir, telescope, e_true_bins=5, e_reco_bins=10):
    path = os.path.join(input_dir, telescope)
    ds = DataStore.from_dir(path)
    observations = ds.get_observations(ds.hdu_table['OBS_ID'].data)

    on_region = CircleSkyRegion(center=crab_position, radius=on_radius[telescope])
    bkg_estimate = ReflectedRegionsBackgroundEstimator(
        observations=observations,
        on_region=on_region,
        exclusion_mask=exclusion_map
    )
    bkg_estimate.run()

    extract = SpectrumExtraction(
        observations=observations,
        bkg_estimate=bkg_estimate.result,
        e_true=e_true_bins,
        e_reco=e_reco_bins,
        containment_correction=False,
        use_recommended_erange=False,
    )
    extract.run()
    return extract.spectrum_observations.stack()


def thikonov(f, normalize=False):
    if normalize:
        f = f / f.sum()
    a = T.dot(laplace_matrix, f)
    a_transposed = a.T
    return T.dot(a, a_transposed)


def response(mu_sig, edisp, fit_range=None):
    counts = T.dot(mu_sig, edisp)
    return counts


def transform(mu_s, aeff):
    return pm.math.log(mu_s / aeff)


def apply_range(*arr, fit_range, bins):
    '''
    Takes one or more array-like things and returns only those entries
    whose bins lie within the fit_range.
    '''
    idx = np.searchsorted(bins.to(u.TeV).value, fit_range.to_value(u.TeV))
    return [a[idx[0]:idx[1]] for a in arr]


def prepare_output(output_dir):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print('Overwriting previous results')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def display_data(data):
    max_chars = 50
    max_value = max(data)
    for d in data:
        n = int(d * max_chars/max_value)
        s = '█'* n
        if d == 0:
            s = '_'
        print(s + f'     {d}')
    print(data)

@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('output_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('dataset', type=click.Choice(['hess', 'fact', 'magic', 'veritas']))
@click.option('--tau', default=0)
@click.option('--n_samples', default=3000)
@click.option('--n_tune', default=1500)
@click.option('--target_accept', default=0.98)
@click.option('--n_cores', default=6)
@click.option('--seed', default=80085)
@click.option('--init', default='advi+adapt_diag', help='Set pymc sampler init string.')
def fit(input_dir, output_dir, dataset, tau, n_samples, n_tune, target_accept, n_cores, seed, init):
    prepare_output(output_dir)

    r = energy_range[dataset]
    e_reco_bins = create_energy_bins(r, n_bins_per_decade=8, overflow=False)
    e_true_bins = create_energy_bins(r, n_bins_per_decade=6, overflow=False)
    observation = load_data(input_dir, dataset, e_reco_bins=e_reco_bins, e_true_bins=e_true_bins)

    on_data = observation.on_vector.data.data.value
    off_data = observation.off_vector.data.data.value

    display_data(on_data)

    exposure_ratio = observation.alpha[0]
    aeff = observation.aeff.data.data.value

    # m = filters.gaussian_filter(observation.edisp.pdf_matrix, sigma=0)
    # edisp = m
    # edisp = filters.gaussian_filter(observation.edisp.pdf_matrix, sigma=0)
    edisp = observation.edisp.pdf_matrix
    # N_true = len(observation.edisp.e_true.lo)
    # N_reco = len(observation.edisp.e_reco.lo)

    print('--' * 30)
    print(f'Unfolding data for:  {dataset.upper()}.  ')
    print(f'IRF with {observation.edisp.pdf_matrix.shape}')
    print(f'Using {len(on_data)} bins with { on_data.sum()} counts in on region and {off_data.sum()} counts in off region.')
    # print(f'Fit range is: {(lo, hi) * u.TeV}.  ')
    model = pm.Model(theano_config={'compute_test_value': 'ignore'})
    with model:
        mu_b = pm.HalfFlat('mu_b', shape=len(off_data))
        mu_s = pm.HalfFlat('mu_s', shape=len(observation.edisp.e_true.lo))

        expected_counts = response(mu_s, edisp=edisp)
        if tau > 0.0:
            lam = thikonov(transform(mu_s, aeff))
            logp = pm.Normal.dist(mu=0, sd=1 / tau).logp(lam)
            p = pm.Potential("thikonov", logp)

        b = pm.Poisson('background', mu=mu_b, observed=off_data)
        s = pm.Poisson('signal', mu=expected_counts + exposure_ratio * mu_b, observed=on_data)


    print('--' * 30)
    print('Sampling likelihood:')
    with model:
        trace = pm.sample(n_samples, cores=n_cores, tune=n_tune, init=init, seed=[seed] * n_cores)


    print('--' * 30)
    print('Plotting result')
    plot_unfolding_result(trace, observation, fit_range=[0.05, 12] * u.TeV)
    plt.savefig(os.path.join(output_dir, 'result.pdf'))


    print('--' * 30)
    print('Plotting traces')
    plt.figure()
    pm.traceplot(trace)
    plt.savefig(os.path.join(output_dir, 'traces.pdf'))

    trace_output = os.path.join(output_dir, 'traces')
    print(f'Saving traces to {trace_output}')
    with model:
        pm.save_trace(trace, trace_output, overwrite=True)


if __name__ == '__main__':
    fit()
