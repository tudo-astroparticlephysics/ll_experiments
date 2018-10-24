from gammapy.spectrum import SpectrumObservationList
from gammapy.spectrum.models import SpectralModel
from gammapy.spectrum import CrabSpectrum
from astropy import units as u
from gammapy.utils.fitting import Parameter, Parameters
import numpy as np


fit_range = {
    'veritas': [0.15, 30] * u.TeV,
    'magic':  [0.08, 30] * u.TeV,
    'hess':  [0.5, 30] * u.TeV,
    'fact':  [0.4, 30] * u.TeV,
    'fermi':  [0.03, 2] * u.TeV,
    'joint':  [0.03, 30] * u.TeV,
}


def wstat_profile(mu_sig, n_on, n_off, alpha):
    if mu_sig == 0 and n_on == 0 and n_off == 0:
        return 0
    c = alpha * (n_on + n_off) - (alpha + 1)*mu_sig
    k = alpha * (alpha + 1)
    d = np.sqrt(c**2 + 4*k*mu_sig*n_off)
    mu_bkg = np.nan_to_num((c+d)/(2*k))
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = n_on * np.log(mu_sig + alpha * mu_bkg)  - mu_sig - (alpha+1) * mu_bkg
        likelihood_value = np.where(mu_bkg == 0, ll, ll + n_off * np.log(mu_bkg))
    return -likelihood_value


def load_spectrum_observations(name):
    """ Load the OGIP files and return a SpectrumObservationList
        SpectrumObservationList has already a method to read from a directory
        http://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumObservationList.html
    """
    if name == 'joint':
        spec_obs_list = SpectrumObservationList()
        # extend the list adding all the other SpectrumObservationList
        for n in {'fermi', 'magic', 'hess', 'fact', 'veritas'}:
            spectra_path = f'spectra/{n}'
            spec_obs = SpectrumObservationList.read(spectra_path)
            spec_obs_list.extend(spec_obs)
    else:
        spectra_path = f'spectra/{name}'
        spec_obs_list = SpectrumObservationList.read(spectra_path)
    return spec_obs_list, fit_range[name]



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


    
def plot_spectra(sampler, mle_result, fit_range=[0.03, 30] * u.TeV, min_sample=50):
    joint_model = Log10Parabola(
        amplitude=3.78 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=2.49 * u.Unit(''),
        beta=0.22 * u.Unit(''),
    )
    joint_model.plot(energy_range=fit_range, energy_power=2, color='black', label='joint')

    r = np.median(sampler.chain[:, min_sample:, :3], axis=(0, 1))
    fitted_model = Log10Parabola(
        amplitude=r[0] * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=r[1] * u.Unit(''),
        beta=r[2] * u.Unit(''),
    )
    fitted_model.plot(energy_range=fit_range, energy_power=2, color='crimson', label='mcmc')
                     
    mle_model = Log10Parabola(
        amplitude=mle_result.x[0] * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=mle_result.x[1] * u.Unit(''),
        beta=mle_result.x[2] * u.Unit(''),
    )
    mle_model.plot(energy_range=fit_range, energy_power=2, color='orange', ls='--', label='mle')


    fact_model = Log10Parabola(
        amplitude=3.47 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=2.56 * u.Unit(''),
        beta=0.4 * u.Unit(''),
    )
    fact_model.plot(energy_range=fit_range, energy_power=2, color='gray', label='fact')


    
    magic_model = Log10Parabola(
        amplitude=4.20 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=2.58 * u.Unit(''),
        beta=0.43 * u.Unit(''),
    )
    magic_model.plot(energy_range=fit_range, energy_power=2, color='gray', ls='--', label='magic')





    CrabSpectrum(reference='meyer').model.plot(energy_range=[0.01, 100]*u.TeV, energy_power=2, color='black', ls=':')
