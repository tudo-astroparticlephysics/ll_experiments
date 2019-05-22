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

from ..plots import plot_unfolding_result
import click
import os
import shutil
from scipy.ndimage import laplace
from ..utils import display_data


crab_position = SkyCoord(ra='83d37m59.0988s', dec='22d00m52.2s')
# exclusion_map = Map.read('./data/exclusion_mask.fits.gz')


def create_energy_bins(fit_range, n_bins_per_decade=10, overflow=False):
    bins = np.logspace(-2, 2, (4 * n_bins_per_decade) + 1)
    bins = apply_range(bins, fit_range=fit_range, bins=bins * u.TeV)[0]
    if overflow:
        bins = np.append(bins, 50000)
        bins = np.append(0, bins)
    return bins * u.TeV


def load_data(path, source_position, on_radius, e_true_bins=5, e_reco_bins=10, exclusion_map=None, ):

    ds = DataStore.from_dir(path)
    observations = ds.get_observations(ds.hdu_table['OBS_ID'].data)

    on_region = CircleSkyRegion(center=crab_position, radius=on_radius)
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

    laplace_matrix = laplace(np.eye(len(f))) // 2

    a = T.dot(laplace_matrix, f)
    a_transposed = a.T
    return T.dot(a, a_transposed)


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




@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('config_file', type=click.Path(dir_okay=False))
@click.argument('output_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('dataset', type=click.Choice(['hess', 'fact', 'magic', 'veritas']))
@click.option('--model_type', default='full', type=click.Choice(['full', 'profile', 'wstat']))
@click.option('--tau', default=0)
@click.option('--n_samples', default=3000)
@click.option('--n_tune', default=1500)
@click.option('--target_accept', default=0.99999)
@click.option('--n_cores', default=6)
@click.option('--seed', default=80085)
@click.option('--init', default='advi+adapt_diag', help='Set pymc sampler init string.')
def main(input_dir, config_file, output_dir, dataset, model_type, tau, n_samples, n_tune, target_accept, n_cores, seed, init):
    prepare_output(output_dir)

    with open(config_file) as f:
        conf = yaml.load(f)
        source_pos = conf['source_position']
        d = {k['name']: k for k in conf['datasets']}
        fit_range = d[dataset]['fit_range'] * u.TeV
        on_radius = d[dataset]['on_radius'] * u.deg
        bins_per_decade = d[dataset]['bins_per_decade']

    path = os.path.join(input_dir, dataset)

    e_reco_bins = create_energy_bins(fit_range, n_bins_per_decade=bins_per_decade + 2, overflow=False)
    e_true_bins = create_energy_bins(fit_range, n_bins_per_decade=bins_per_decade, overflow=False)
    
    observation = load_data(path, e_reco_bins=e_reco_bins, e_true_bins=e_true_bins, on_radius=on_radius, source_position=source_pos)

    on_data = observation.on_vector.data.data.value
    off_data = observation.off_vector.data.data.value
    print(f'On Data {on_data.shape}\n')
    display_data(on_data)
    print(f'\n\n Off Data {off_data.shape}\n')
    display_data(off_data)

    exposure_ratio = observation.alpha[0]
    aeff = observation.aeff.data.data.value

    edisp = observation.edisp.pdf_matrix
    plt.figure()
    plt.imshow(edisp)
    plt.xlabel('reco energy')
    plt.ylabel('true energy')
    plt.savefig(os.path.join(output_dir, 'edisp.pdf'))

    print('--' * 30)
    print(f'Unfolding data for:  {dataset.upper()}.  ')
    print(f'IRF with {edisp.shape}')
    print(f'Using {len(on_data)} bins with { on_data.sum()} counts in on region and {off_data.sum()} counts in off region.')

    model = pm.Model(theano_config={'compute_test_value': 'ignore'})
    with model:
        # BoundedNormal = pm.Bound(pm.Normal, lower=-1.0)
        # mu_b = BoundedNormal('mu_b', shape=len(off_data), mu=off_data, sd=1)
        # mu_b = pm.TruncatedNormal('mu_b', shape=len(off_data), sd=5, mu=off_data, lower=0.01)
        mu_b = pm.HalfFlat('mu_b', shape=len(off_data))
        expected_counts = pm.HalfFlat('mu_s', shape=len(observation.edisp.e_true.lo), testval=on_data.mean())
        c = expected_counts * observation.aeff.data.data.to_value('km2') * observation.livetime.to_value('s') * observation.edisp.e_true.bin_width.to_value('TeV')
        mu_s = T.dot(c, edisp)

        if tau > 0.0:
            lam = thikonov(transform(mu_s, aeff))
            logp = pm.Normal.dist(mu=0, sd=1 / tau).logp(lam)
            pm.Potential("thikonov", logp)

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


    print('--' * 30)
    print('Plotting result')
    plot_unfolding_result(trace, observation)
    plt.savefig(os.path.join(output_dir, 'result.pdf'))


    print('--' * 30)
    print('Plotting Diagnostics')
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
    

    trace_output = os.path.join(output_dir, 'traces')
    print(f'Saving traces to {trace_output}')
    with model:
        pm.save_trace(trace, trace_output, overwrite=True)


if __name__ == '__main__':
    main()
