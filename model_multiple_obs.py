from scipy.integrate import quad, trapz, fixed_quad
import theano
import theano.tensor as T
from theano.tests.unittest_tools import verify_grad
from theano.compile.ops import as_op

import numpy as np
import pymc3 as pm

import os

import matplotlib.pyplot as plt

import astropy.units as u

from tqdm import tqdm

from gammapy.spectrum import SpectrumObservationList

from utils import plot_spectra, Log10Parabola, integrate_spectrum

import click


theano.config.floatX ='float32'
# print(theano.config)
class Integrate(theano.Op):
    def __init__(self, expr, var, lower, upper, *inputs):
        super().__init__()
        self._expr = expr
        self._var = var
        self._extra_vars = inputs
        self.lower = lower
        self.upper = upper
        self._func = theano.function(
            [var] + list(self._extra_vars),
            self._expr,
            on_unused_input='ignore'
        )
    
    def make_node(self, *inputs):
        assert len(self._extra_vars)  == len(inputs)
        return theano.Apply(self, list(inputs), [T.dscalar().type()])
    
    def perform(self, node, inputs, out):
        x = np.linspace(self.lower, self.upper, num=3)
        y = np.array([self._func(i , *inputs) for i in x])
        val = trapz(y, x)
        out[0][0] = np.array(val)
        
    def grad(self, inputs, grads):
        out, = grads
        grads = T.grad(self._expr, self._extra_vars)
        dargs = []
        for grad in grads:
            integrate = Integrate(grad, self._var, self.lower, self.upper, *self._extra_vars)
            darg = out * integrate(*inputs)
            dargs.append(darg)
            
        return dargs


def apply_range(*arr, fit_range, bins):
    idx = np.searchsorted(bins.to_value(u.TeV), fit_range.to_value(u.TeV))
    return [a[idx[0]:idx[1]] for a in arr]


def forward_fold_log_parabola_symbolic(amplitude, alpha, beta, observations, fit_range=None):
    
    amplitude *= 1e-11
    
    predicted_signal_per_observation = []
    for observation in observations:
        obs_bins = observation.on_vector.energy.bins.to_value(u.TeV)

        aeff_bins = observation.aeff.energy
        e_reco_bins = observation.edisp.e_reco
        e_true_bins = observation.edisp.e_true

        lower =  e_true_bins.lo.to_value(u.TeV)
        upper = e_true_bins.hi.to_value(u.TeV)

        energy = T.dscalar('energy')
        amplitude_ = T.dscalar('amplitude_')
        alpha_ = T.dscalar('alpha_')
        beta_ = T.dscalar('beta_')

        func = amplitude_ * energy **(-alpha_ - beta_ * T.log10(energy))

        counts = []
        for a, b in zip(lower, upper):
            c = Integrate(func, energy, a, b, amplitude_, alpha_, beta_)(amplitude, alpha, beta)
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



def get_observed_counts(observations, fit_range=None):
    on_data = []
    off_data = []
    
    for observation in observations:
        on_data.append(observation.on_vector.data.data.value)
        off_data.append(observation.off_vector.data.data.value)
    on_data = np.sum(on_data, axis=0)
    off_data = np.sum(off_data, axis=0)
    if fit_range is not None:
        energy_bins = observations[0].on_vector.energy.bins
        on_data, off_data = apply_range(on_data, off_data, fit_range=fit_range, bins=energy_bins)
    
    return on_data, off_data

def interval_transform(x, a, b):
    return np.log(x - a) - np.log(b - x)


def plot_landscape(model, output_dir, mu_seed):
    N = 25
    betas = np.linspace(0, 3, N)
    alphas = np.linspace(1.1, 4.0, N)
      
    f = model.logp
    zs = []
    a, b = np.meshgrid(alphas, betas)
    for al, be in tqdm(zip(a.ravel(), b.ravel())):

        p = f(amplitude_lowerbound__ = np.log(4.0), alpha_lowerbound__ = np.log(al), beta_lowerbound__= np.log(be), mu_b_interval__ = mu_seed)
        zs.append(p)

    zs = np.array(zs)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5.5))
    cf = ax1.contourf(a, b, zs.reshape(len(a), -1),  levels=124)
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('beta')
    plt.colorbar(cf, ax=ax1)
    plt.savefig(f'{output_dir}/landscape.pdf')



def load_spectrum_observations(input_dir, name):
    """ Load the OGIP files and return a SpectrumObservationList
        SpectrumObservationList has already a method to read from a directory
        http://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumObservationList.html
    """

    if name == 'joint':
        spec_obs_list = SpectrumObservationList()
        # extend the list adding all the other SpectrumObservationList
        for n in {'magic', 'hess', 'fact', 'veritas'}:
            spectra_path = os.path.join(input_dir, n)
            spec_obs = SpectrumObservationList.read(spectra_path)
            spec_obs_list.extend(spec_obs)
            
    else:
        spectra_path = os.path.join(input_dir, name)
        spec_obs_list = SpectrumObservationList.read(spectra_path)

    return spec_obs_list
    

fit_ranges = {'fact': [0.4, 25] * u.TeV, 'magic': [0.08, 30] * u.TeV, 'hess': [0.8, 30] * u.TeV, 'veritas': [0.16, 30] * u.TeV}

@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('telescope', type=click.Choice(['fact', 'hess', 'magic', 'veritas']))
@click.argument('output_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.option('--n_samples', default=1000)
@click.option('--n_tune', default=600)
@click.option('--target_accept', default=0.8)
@click.option('--n_cores', default=6)
def fit(input_dir, telescope, output_dir, n_samples, n_tune, target_accept, n_cores):
    
    os.makedirs(output_dir, exist_ok=True)
    
    observations = load_spectrum_observations(input_dir, telescope)
    fit_range = fit_ranges[telescope]
   
    on_data, off_data = get_observed_counts(observations, fit_range=fit_range)
    exposure_ratio = observations[0].alpha[0]   # 1/9 = 0.11111

    print(f'Fitting data in {input_dir}/{telescope} with {len(on_data)} bins.')
    
    model = pm.Model(theano_config={'compute_test_value': 'ignore'})

    with model:

        amplitude = pm.TruncatedNormal('amplitude', mu=4, sd=3, lower=0, testval=4)
        alpha = pm.TruncatedNormal('alpha', mu=2.5, sd=2, lower=0, testval=2.5)
        beta = pm.TruncatedNormal('beta', mu=0.5, sd=0.5, lower=0, testval=0.5)

        mu_s = forward_fold_log_parabola_symbolic(amplitude, alpha, beta, observations, fit_range=fit_range)

        mu_b = pm.Uniform('mu_b', shape=len(off_data), lower=0, upper=2 * np.max(off_data))

        b = pm.Poisson('background', mu=mu_b, observed=off_data)    
        s = pm.Poisson('signal', mu=mu_s + exposure_ratio * mu_b, observed=on_data)

    print('Plotting Landscape')
    plot_landscape(model, output_dir, mu_seed = interval_transform(off_data + 0.005, 0, b=2*np.max(off_data)))
    
    print('Fitting Map')
    map_estimate = pm.find_MAP(model=model, start={'amplitude': 4.0})
    print(map_estimate)

    print('Sampling Posterior')
    with model:
        trace = pm.sample(n_samples, chains=n_cores, cores=n_cores, init='auto', target_accept=target_accept, tune=n_tune)


    print(trace['amplitude'].mean(), trace['alpha'].mean(), trace['beta'].mean())
    print(np.median(trace['amplitude']), np.median(trace['alpha']), np.median(trace['beta']))

    print('Plotting traces')
    plt.figure()
    pm.traceplot(trace)
    plt.savefig(f'{output_dir}/traces.pdf')

    print('Saving traces')
    with model:
        pm.save_trace(trace, os.path.join(output_dir, 'traces'), overwrite=True)
        
if __name__ == '__main__':
    fit()
