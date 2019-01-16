from scipy.integrate import quad, trapz, fixed_quad
import theano
import theano.tensor as T
from theano.compile.ops import as_op

import numpy as np
import pymc3 as pm

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from tqdm import tqdm

from gammapy.spectrum import CountsPredictor, CountsSpectrum, SpectrumObservationList
from spectrum_io import load_spectrum_observations
from theano_ops import Integrate
from plots import plot_landscape

import click
import os
import shutil
from functools import lru_cache

def apply_range(*arr, fit_range, bins):
    idx = np.searchsorted(bins.to(u.TeV).value, fit_range.to(u.TeV).value )
    return [a[idx[0]:idx[1]] for a in arr]


@lru_cache(maxsize=5000)
def get_integrator(a, b):
    energy = T.dscalar('energy')
    amplitude_ = T.dscalar('amplitude_')
    alpha_ = T.dscalar('alpha_')
    beta_ = T.dscalar('beta_')

    func = amplitude_ * energy **(-alpha_ - beta_ * T.log10(energy))

    return Integrate(func, energy, a, b, amplitude_, alpha_, beta_)


def forward_fold_log_parabola_symbolic(amplitude, alpha, beta, observations, fit_range=None):

    amplitude *= 1e-11
    if not fit_range:
        lo = observations[0].meta['LO_THRES']
        hi = observations[0].meta['HI_THRES']
        fit_range = [lo, hi] * u.TeV

    predicted_signal_per_observation = []
    for observation in observations:
        obs_bins = observation.on_vector.energy.bins.to_value(u.TeV)

        aeff_bins = observation.aeff.energy
        e_reco_bins = observation.edisp.e_reco
        e_true_bins = observation.edisp.e_true

        lower =  e_true_bins.lo.to_value(u.TeV)
        upper = e_true_bins.hi.to_value(u.TeV)


        counts = []
        for a, b in zip(lower, upper):
            # c = Integrate(func, energy, a, b, amplitude_, alpha_, beta_)(amplitude, alpha, beta)
            c = get_integrator(a, b)(amplitude, alpha, beta)
            counts.append(c)

        counts = T.stack(counts)
        aeff = observation.aeff.data.data.to_value(u.cm**2).astype(np.float32)
        aeff = aeff


        counts *= aeff
        counts *= observation.livetime.to_value(u.s)
        edisp = observation.edisp.pdf_matrix
        edisp = edisp

        predicted_signal_per_observation.append(T.dot(counts, edisp))

    predicted_counts = T.sum(predicted_signal_per_observation, axis=0)
    if fit_range is not None:
        idx = np.searchsorted(obs_bins, fit_range.to_value(u.TeV))
        predicted_counts = predicted_counts[idx[0]:idx[1]]

    return predicted_counts


def calc_mu_b(mu_s, on_data, off_data, exposure_ratio):
    '''
    Calculate the value of mu_b given all other  parameters. See
    Dissertation of johannes king or gammapy docu on WSTAT.
    '''
    alpha = exposure_ratio
    c = alpha * (on_data + off_data) - (alpha + 1)*mu_s
    d = pm.math.sqrt(c**2 + 4 * (alpha + 1)*alpha*off_data*mu_s)
    mu_b = (c + d) / (2*alpha*(alpha + 1))

    return mu_b


def get_observed_counts(observations, fit_range=None):
    on_data = []
    off_data = []

    for observation in observations:
        lo = observation.meta['LO_THRES']
        hi = observation.meta['HI_THRES']
        fit_range = [lo, hi] * u.TeV

        on_data.append(observation.on_vector.data.data.value)
        off_data.append(observation.off_vector.data.data.value)

    on_data = np.sum(on_data, axis=0)
    off_data = np.sum(off_data, axis=0)

    energy_bins = observations[0].on_vector.energy.bins
    on_data, off_data = apply_range(on_data, off_data, fit_range=fit_range, bins=energy_bins)

    return on_data, off_data



def prepare_output(output_dir):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print('Overwriting previous results')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('output_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.option('--model_type', default='full', type=click.Choice(['full', 'profile', 'wstat']))
@click.option('--n_samples', default=1000)
@click.option('--n_tune', default=600)
@click.option('--target_accept', default=0.8)
@click.option('--n_cores', default=6)
@click.option('--joint', default=False, help='Use all datasets found in input dir')
@click.option('--init', default='auto', help='Set pymc sampler init string.')
def fit(input_dir, output_dir, model_type, n_samples, n_tune, target_accept, n_cores, joint, init,):
    observations, telescope = load_spectrum_observations(input_dir, joint=joint)

    prepare_output(output_dir)

    # todo: this has to happen for every observation independently
    exposure_ratio = observations[0].alpha[0]

    on_data, off_data = get_observed_counts(observations)


    print('--' * 30)
    print('bins, total counts in on region and off_region:')
    print(f'Fitting data for {telescope}.  ', len(on_data), on_data.sum(), off_data.sum())

    model = pm.Model(theano_config={'compute_test_value': 'ignore'})
    with model:
        amplitude = pm.TruncatedNormal('amplitude', mu=4, sd=1, lower=0.01, testval=4)
        alpha = pm.TruncatedNormal('alpha', mu=2.5, sd=1, lower=0.01, testval=2.5)
        beta = pm.TruncatedNormal('beta', mu=0.5, sd=0.5, lower=0.01, testval=0.5)

        mu_s = forward_fold_log_parabola_symbolic(amplitude, alpha, beta, observations)

        if model_type == 'wstat':
            print('Building profiled likelihood model')
            mu_b  = pm.Deterministic('mu_b', calc_mu_b(mu_s, on_data, off_data, exposure_ratio))
        else:
            print('Building full likelihood model')
            mu_b = pm.TruncatedNormal('mu_b', lower=0, shape=len(off_data), mu=off_data, sd=5)

        b = pm.Poisson('background', mu=mu_b, observed=off_data, shape=len(off_data))
        s = pm.Poisson('signal', mu=mu_s + exposure_ratio * mu_b, observed=on_data, shape=len(on_data))


    print('--' * 30)
    print('Model debug information:')
    for RV in model.basic_RVs:
        print(RV.name, RV.logp(model.test_point))

    print('--' * 30)
    print('Plotting landscape:')
    fig, _ = plot_landscape(model, off_data)
    fig.savefig(os.path.join(output_dir, 'landscape.pdf'))

    print('--' * 30)
    print('Sampling likelihood:')
    with model:
        trace = pm.sample(n_samples, cores=n_cores, tune=n_tune, init=init) # advi+adapt_diag

    print('--'*30)
    print(f'Fit results for {telescope}')
    print(trace['amplitude'].mean(), trace['alpha'].mean(), trace['beta'].mean())
    print(np.median(trace['amplitude']), np.median(trace['alpha']), np.median(trace['beta']))

    print('--' * 30)
    print('Plotting traces')
    plt.figure()
    varnames = ['amplitude', 'alpha', 'beta'] if model_type != 'full' else ['amplitude', 'alpha', 'beta', 'mu_b']
    pm.traceplot(trace, varnames=varnames)
    plt.savefig(os.path.join(output_dir, 'traces.pdf'))

    trace_output = os.path.join(output_dir, 'traces')
    print(f'Saving traces to {trace_output}')
    with model:
        pm.save_trace(trace, trace_output, overwrite=True)


if __name__ == '__main__':
    fit()
