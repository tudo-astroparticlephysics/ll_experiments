import theano.tensor as T

import numpy as np
import pymc3 as pm

import matplotlib.pyplot as plt

import astropy.units as u

from spectrum_io import load_spectrum_observations, load_joint_spectrum_observation
from theano_ops import IntegrateVectorized
from plots import plot_landscape

import click
import os
import shutil


def apply_range(*arr, fit_range, bins):
    '''
    Takes one or more array-like things and returns only those entries
    whose bins lie within the fit_range.
    '''
    idx = np.searchsorted(bins.to(u.TeV).value, fit_range.to_value(u.TeV))
    return [a[idx[0]:idx[1]] for a in arr]


def init_integrators(observations):
    '''
    Initializes the theano symbolic integrator for a LogParabola Spectrum
    with a base 10 logarithm.
    '''
    energy = T.dscalar('energy')
    amplitude_ = T.dscalar('amplitude_')
    alpha_ = T.dscalar('alpha_')
    beta_ = T.dscalar('beta_')

    # define spectrum and its gradients
    def f(self, E, phi, alpha, beta):
        return phi*E**(-alpha-beta*np.log10(E))
    def df_dphi(self, E, phi, alpha, beta):
        return E**(-alpha-beta*np.log10(E))
    def df_dalpha(self, E, phi, alpha, beta):
        return -phi*E**(-alpha-beta*np.log10(E)) * np.log(E)
    def df_dbeta(self, E, phi, alpha, beta):
        return -(phi*E**(-alpha-beta*np.log10(E)) * np.log(E)**2)/np.log(10)

    e_true_bins = observations[0].edisp.e_true
    bins = e_true_bins.bins.to_value(u.TeV)
    return IntegrateVectorized(f, [df_dphi, df_dalpha, df_dbeta], energy, bins, amplitude_, alpha_, beta_)


def forward_fold_log_parabola_symbolic(integrator, amplitude, alpha, beta, observations, fit_range=None):
    '''
    Forward fold the spectral model through the instrument functions given in the 'observations'
    Returns the predicted counts in each energy bin.
    '''
    amplitude *= 1e-11
    if not fit_range:
        lo = observations[0].meta['LO_RANGE']
        hi = observations[0].meta['HI_RANGE']
        fit_range = [lo, hi] * u.TeV

    predicted_signal_per_observation = []
    for observation in observations:
        obs_bins = observation.on_vector.energy.bins.to_value(u.TeV)
        counts = integrator(amplitude, alpha, beta)  # the integrator has been initialized with the proper energy bins before.


        aeff = observation.aeff.data.data.to_value(u.cm**2).astype(np.float32)

        counts *= aeff
        counts *= observation.livetime.to_value(u.s)
        edisp = observation.edisp.pdf_matrix
        from IPython import embed; embed()
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

    https://www.imprs-hd.mpg.de/267524/thesis_king.pdf

    https://docs.gammapy.org/0.8/stats/fit_statistics.html#poisson-data-with-background-measurement

    '''
    alpha = exposure_ratio
    c = alpha * (on_data + off_data) - (alpha + 1)*mu_s
    d = pm.math.sqrt(c**2 + 4 * (alpha + 1)*alpha*off_data*mu_s)
    mu_b = (c + d) / (2*alpha*(alpha + 1))

    # mu_b  = T.where(on_data == 0, off_data/(alpha + 1), mu_b)
    # mu_b  = T.where(off_data == 0, on_data/(alpha + 1) - mu_s/alpha, mu_b)
    return mu_b


def get_observed_counts(observations, fit_range=None):
    on_data = []
    off_data = []

    for observation in observations:
        lo = observation.meta['LO_RANGE']
        hi = observation.meta['HI_RANGE']
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
@click.argument('dataset', type=click.Choice(['hess', 'fact', 'magic', 'veritas', 'joint']))
@click.option('--model_type', default='full', type=click.Choice(['full', 'profile', 'wstat']))
@click.option('--n_samples', default=1000)
@click.option('--n_tune', default=600)
@click.option('--target_accept', default=0.8)
@click.option('--n_cores', default=6)
@click.option('--seed', default=80085)
@click.option('--init', default='advi+adapt_diag', help='Set pymc sampler init string.')
@click.option('--profile/--no-profile', default=False, help='Output profiling information')
def fit(input_dir, output_dir, dataset, model_type, n_samples, n_tune, target_accept, n_cores, seed, init, profile):
    '''Fit log-parabola model to DATASET. 
    
    Parameters
    ----------
    input_dir : [type]
        input directory containing subdirs for each instrument with dl3 data
    output_dir : [type]
        where to save the results. traces and two plots
    dataset : string
        telescope name
    model_type : string
        whetehr to use the profile likelihood ('wstat' or 'profile') or not ('full')
    n_samples : int
        number of samples to draw
    n_tune : int
        number of tuning steps
    target_accept : foat
        target accept fraction for the pymc sampler
    n_cores : int
        number of cpu cores to use
    seed : int
        random seed
    init : string
        pymc init string
    profile : bool
        whether to output debugging/profiling information to the console
    Raises
    ------
    NotImplementedError
        This does not yet work on the joint dataset. but thats good enough for me.
    '''
    np.random.seed(seed)

    if dataset == 'joint':
        #TODO need to calculate mu_b for each observation independently.
        raise NotImplementedError('This is not implemented for the joint dataset yet.')
        # observations, lo, hi = load_joint_spectrum_observation(input_dir)
    else:
        p = os.path.join(input_dir, dataset)
        observations, lo, hi = load_spectrum_observations(p)

    prepare_output(output_dir)

    #TODO: this has to happen for every observation independently
    exposure_ratio = observations[0].alpha[0]

    on_data, off_data = get_observed_counts(observations)

    integrator_dict = init_integrators(observations)

    print('--' * 30)
    print(f'Fitting data for {dataset} in {len(observations)} observations.  ')
    print(f'Using {len(on_data)} bins with { on_data.sum()} counts in on region and {off_data.sum()} counts in off region.')
    print(f'Fit range is: {(lo, hi) * u.TeV}.  ')
    model = pm.Model(theano_config={'compute_test_value': 'ignore'})
    with model:
        amplitude = pm.TruncatedNormal('amplitude', mu=4, sd=1, lower=0.01, testval=4)
        alpha = pm.TruncatedNormal('alpha', mu=2.5, sd=1, lower=0.01, testval=2.5)
        beta = pm.TruncatedNormal('beta', mu=0.5, sd=0.5, lower=0.01, testval=0.5)

        mu_s = forward_fold_log_parabola_symbolic(integrator_dict, amplitude, alpha, beta, observations)

        if model_type == 'wstat':
            print('Building profiled likelihood model')
            mu_b  = pm.Deterministic('mu_b', calc_mu_b(mu_s, on_data, off_data, exposure_ratio))
        else:
            print('Building full likelihood model')
            mu_b = pm.TruncatedNormal('mu_b', lower=0, shape=len(off_data), mu=off_data, sd=5)

        pm.Poisson('background', mu=mu_b, observed=off_data, shape=len(off_data))
        pm.Poisson('signal', mu=mu_s + exposure_ratio * mu_b, observed=on_data, shape=len(on_data))


    print('--' * 30)
    print('Model debug information:')
    for RV in model.basic_RVs:
        print(RV.name, RV.logp(model.test_point))

    if profile:
        model.profile(model.logpt).summary()

    print('--' * 30)
    print('Plotting landscape:')
    fig, _ = plot_landscape(model, off_data)
    fig.savefig(os.path.join(output_dir, 'landscape.pdf'))

    print('--' * 30)
    print('Sampling likelihood:')
    with model:
        trace = pm.sample(n_samples, cores=n_cores, tune=n_tune, init=init, seed=[seed]*n_cores)

    print('--'*30)
    print(f'Fit results for {dataset}')
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
