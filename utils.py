from gammapy.spectrum import SpectrumObservationList
from gammapy.spectrum.models import SpectralModel
from gammapy.spectrum import CrabSpectrum
from astropy import units as u
from gammapy.utils.fitting import Parameter, Parameters
import numpy as np
from astropy.units import Quantity


fit_range = {
    'veritas': [0.15, 30] * u.TeV,
    'magic':  [0.08, 30] * u.TeV,
    'hess':  [0.5, 30] * u.TeV,
    'fact':  [0.4, 30] * u.TeV,
    'fermi':  [0.03, 2] * u.TeV,
    'joint':  [0.03, 30] * u.TeV,
}


def integrate_spectrum(func, xmin, xmax, ndecade=100, intervals=False):
    """
    Integrate 1d function using the log-log trapezoidal rule. If scalar values
    for xmin and xmax are passed an oversampled grid is generated using the
    ``ndecade`` keyword argument. If xmin and xmax arrays are passed, no
    oversampling is performed and the integral is computed in the provided
    grid.
    Parameters
    ----------
    func : callable
        Function to integrate.
    xmin : `~astropy.units.Quantity` or array-like
        Integration range minimum
    xmax : `~astropy.units.Quantity` or array-like
        Integration range minimum
    ndecade : int, optional
        Number of grid points per decade used for the integration.
        Default : 100.
    intervals : bool, optional
        Return integrals in the grid not the sum, default: False
    """
    is_quantity = False
    if isinstance(xmin, Quantity):
        unit = xmin.unit
        xmin = xmin.value
        xmax = xmax.to_value(unit)
        is_quantity = True

    if np.isscalar(xmin):
        logmin = np.log10(xmin)
        logmax = np.log10(xmax)
        n = (logmax - logmin) * ndecade
        x = np.logspace(logmin, logmax, n)
    else:
        x = np.append(xmin, xmax[-1])

    if is_quantity:
        x = x * unit

    y = func(x)

    val = _trapz_loglog(y, x, intervals=intervals)

    return val


# This function is copied over from https://github.com/zblz/naima/blob/master/naima/utils.py#L261
# and slightly modified to allow use with the uncertainties package


def _trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.
    Integrate `y` (`x`) along given axis in loglog space.
    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.
    intervals : bool, optional
        Return array of shape x not the total integral, default: False
    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    log10 = np.log10

    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)
    slice1, slice2 = tuple(slice1), tuple(slice2)

    # arrays with uncertainties contain objects
    if y.dtype == "O":
        from uncertainties.unumpy import log10

        # uncertainties.unumpy.log10 can't deal with tiny values see
        # https://github.com/gammapy/gammapy/issues/687, so we filter out the values
        # here. As the values are so small it doesn't affect the final result.
        # the sqrt is taken to create a margin, because of the later division
        # y[slice2] / y[slice1]
        valid = y > np.sqrt(np.finfo(float).tiny)
        x, y = x[valid], y[valid]

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with np.errstate(invalid="ignore", divide="ignore"):
        # Compute the power law indices in each integration bin
        b = log10(y[slice2] / y[slice1]) / log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use normal
        # powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.0) > 1e-10,
            (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
            / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]),
        )

    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret


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



def apply_range(*arr, fit_range, bins):
    idx = np.searchsorted(bins.to(u.TeV).value, fit_range.to(u.TeV).value )
    return [a[idx[0]:idx[1]] for a in arr]


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

    for obs in spec_obs_list:
        obs.hi_threshold = fit_range[name][1]
        obs.lo_threshold = fit_range[name][0]

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
