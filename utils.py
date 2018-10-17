from gammapy.spectrum import SpectrumObservationList
from gammapy.spectrum.models import SpectralModel
from astropy import units as u
from gammapy.utils.fitting import Parameter, Parameters
import numpy as np


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
        for name in {'fermi', 'magic', 'hess', 'fact', 'veritas'}:
            spectra_path = f'spectra/{name}'
            spec_obs = SpectrumObservationList.read(spectra_path)
            spec_obs_list.extend(spec_obs)
    else:
        spectra_path = f'spectra/{name}'
        spec_obs_list = SpectrumObservationList.read(spectra_path)

    return spec_obs_list



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

