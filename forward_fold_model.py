from gammapy.spectrum import SpectrumObservationList
from gammapy.spectrum.models import SpectralModel
from astropy import units as u
from gammapy.utils.fitting import Parameter, Parameters
import numpy as np
from astropy.units import Quantity

import yaml
import numpy as np
# from gammapy.stats import wstat
from gammapy.spectrum import SpectrumFit, CrabSpectrum
from gammapy.spectrum.models import SpectralModel

from scipy.optimize import minimize
from gammapy.spectrum import CountsPredictor


class Log10Parabola(SpectralModel):
    """Gammapy log parabola model matching Sherpa parametrisation.

    The difference to the `LogParabola` in Gammapy is that here
    `log10` is used, whereas in Gammapy natural `log` is used.

    We're doing this to make the comparison / debugging easier.

    * Sherpa: http://cxc.harvard.edu/sherpa/ahelp/logparabola.html
    * Gammapy: http://docs.gammapy.org/dev/api/gammapy.spectrum.models.LogParabola.html
    """

    def __init__(self, amplitude=1E-12 * u.Unit('cm-2 s-1 TeV-1'), reference=10 * u.TeV,
                 alpha=2, beta=1):
        self.parameters = Parameters([
            Parameter('amplitude', amplitude),
            Parameter('reference', reference, frozen=True),
            Parameter('alpha', alpha),
            Parameter('beta', beta)
        ])

    @staticmethod
    def evaluate(energy, amplitude, reference, alpha, beta):
        """Evaluate the model (static function)."""
        try:
            xx = energy / reference
            exponent = -alpha - beta * np.log10(xx)
        except AttributeError:
            from uncertainties.unumpy import log10
            xx = energy / reference
            exponent = -alpha - beta * log10(xx)

        return amplitude * np.power(xx, exponent)



def model_signal_prediction(model, observation):
    predictor = CountsPredictor(model=model, aeff=observation.aeff, edisp=observation.edisp, livetime=observation.livetime)
    predictor.run()
    counts = predictor.npred.data.data
    counts *= observation.on_vector.areascal
    return counts.value

def ln_profile_likelihood(mu_sig, n_on, n_off, alpha):
    c = alpha * (n_on + n_off) - (alpha + 1)*mu_sig
    k = alpha * (alpha + 1)
    d = np.sqrt(c**2 + 4*k*mu_sig*n_off)
    mu_bkg = np.nan_to_num((c+d)/(2*k))
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = n_on * np.log(mu_sig + alpha * mu_bkg)  - mu_sig - (alpha+1) * mu_bkg
        likelihood_value = np.where(mu_bkg == 0, ll, ll + n_off * np.log(mu_bkg))
    
    return np.where((mu_sig == 0) & (n_on == 0) & (n_off == 0), 0, -likelihood_value)
    

def ln_prior(theta):
    '''
    Uninformative poisson prior as far as I know. 
    
    See https://en.wikipedia.org/wiki/Jeffreys_prior
    '''
    mu_sig = theta
    prior = np.zeros_like(mu_sig)

    m = mu_sig < 0
    prior[m] = -np.inf
    return prior



def ln_prob(theta, n_on, n_off, alpha, return_posterior=False):
    
    prob = 0.5*ln_profile_likelihood(theta, n_on, n_off, alpha)

    if return_posterior:
        lp = ln_prior(theta)
        prob = prob + lp
    
    return np.where(np.isfinite(prob), prob, 0)


def apply_range(*arr, fit_range, bins):
    idx = np.searchsorted(bins.to(u.TeV).value, fit_range.to(u.TeV).value )
    return [a[idx[0]:idx[1]] for a in arr]


def model_probability(theta, observations, return_posterior=False, fit_range=None):
    amplitude, alpha, beta = theta

    if (alpha < 0) or ( alpha > 5) or (beta > 5):
        return -np.inf

    model = Log10Parabola(
        amplitude=amplitude * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=alpha * u.Unit(''),
        beta=beta * u.Unit(''),
    )
       
    obs_probabilities = []
    for obs in observations:
        mu_sig = model_signal_prediction(model, obs)
        n_on=obs.on_vector.data.data.value
        n_off=obs.off_vector.data.data.value
        obs_alpha=obs.alpha

        if fit_range is not None:
            bins = obs.on_vector.energy.bins
            mu_sig, n_on, n_off, obs_alpha = apply_range(mu_sig, n_on, n_off, obs_alpha, bins=bins, fit_range=fit_range)

        prob = ln_prob(mu_sig, n_on, n_off, obs_alpha, return_posterior=return_posterior).sum()            
        obs_probabilities.append(prob)
    return -sum(obs_probabilities), mu_sig
    