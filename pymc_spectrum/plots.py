import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gammapy.spectrum.models import SpectralModel
import astropy.units as u
from gammapy.utils.fitting import Parameter, Parameters




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


magic_model = Log10Parabola(
    amplitude=4.20 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
    reference=1 * u.Unit('TeV'),
    alpha=2.58 * u.Unit(''),
    beta=0.43 * u.Unit(''),
)

fact_model = Log10Parabola(
    amplitude=3.5 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
    reference=1 * u.Unit('TeV'),
    alpha=2.56 * u.Unit(''),
    beta=0.4 * u.Unit(''),
)

hess_model = Log10Parabola(
    amplitude=4.47 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
    reference=1 * u.Unit('TeV'),
    alpha=2.39 * u.Unit(''),
    beta=0.37 * u.Unit(''),
)

veritas_model = Log10Parabola(
    amplitude=3.76 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
    reference=1 * u.Unit('TeV'),
    alpha=2.44 * u.Unit(''),
    beta=0.26 * u.Unit(''),
)


def plot_landscape(model, off_data, N=60):
    '''
    Plot the likelihood landscape of the given pymc model
    '''
    betas = np.linspace(0, 3, N)
    alphas = np.linspace(1.1, 4.0, N)
    f = model.logp
    zs = []
    a, b = np.meshgrid(alphas, betas)
    # from IPython import embed; embed()
    for al, be in tqdm(zip(a.ravel(), b.ravel())):

        try:
            p = f(
                amplitude_lowerbound__=np.log(4),
                alpha_lowerbound__=np.log(al),
                beta_lowerbound__=np.log(be),
                mu_b_lowerbound__=np.log(off_data + 0.1)
            )
        except TypeError:
            p = f(
                amplitude_log__=np.log(4),
                alpha_log__=np.log(al),
                beta_log__=np.log(be),
                mu_b_log__=np.log(off_data + 0.1)
            )
        zs.append(p)
    zs = np.array(zs)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
    cf = ax1.contourf(a, b, zs.reshape(len(a), -1), levels=100)
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('beta')
    fig.colorbar(cf, ax=ax1)

    cf = ax2.contourf(a, b, np.log10(-zs.reshape(len(a), -1)), levels=255, )
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('beta')
    fig.colorbar(cf, ax=ax2)

    return fig, [ax1, ax2]


def plot_excees(excess, bins, ax=None):
    if not ax:
        ax = plt.gca()

    ax.step(bins[:-1], excess, where='post')


def plot_unfolding_result(trace, bins, fit_range=[0.01, 20] * u.TeV, area_scaling=1, ax=None):
    if not ax:
        ax = plt.gca()

    bins = bins.to_value('TeV')
    transformed_samples = trace['expected_counts'][:, :]
    # norm = 1 / stacked_observation.aeff.data.data / stacked_observation.livetime / stacked_observation.edisp.e_true.bin_width
    norm = 1 * u.Unit('km-2 s-1 TeV-1')
    flux = (transformed_samples * norm).to_value(1 / (u.TeV * u.s * u.cm**2)) * area_scaling

    bin_center = np.sqrt(bins[0:-1] * bins[1:])
    # bin_width = np.diff(bins)
    
    lower, mean_flux, upper = np.nanpercentile(flux, [16, 50, 84], axis=0)
    lower_95, upper_95 = np.nanpercentile(flux, [5, 95], axis=0)
    print(mean_flux)
    line_range = [0.01, 20] * u.TeV
    magic_model.plot(energy_range=line_range, ls='--', color='gray', label='magic', ax=ax)
    fact_model.plot(energy_range=line_range, ls=':', color='silver', label='fact', ax=ax)
    hess_model.plot(energy_range=line_range, ls='-.', color='darkgray', label='hess', ax=ax)
    veritas_model.plot(energy_range=line_range, ls='-', color='lightgray', label='veritas', ax=ax)

    xl = bin_center - bins[:-1]
    xu = bins[1:] - bin_center

    dl = mean_flux - lower_95
    du = upper_95 - mean_flux
    ax.errorbar(bin_center, mean_flux, yerr=[dl, du], linestyle='', color='gray')

    dl = mean_flux - lower
    du = upper - mean_flux

    ax.errorbar(bin_center, mean_flux, yerr=[dl, du], xerr=[xl, xu], linestyle='')

    if fit_range is not None:
        idx = np.searchsorted(bins, fit_range.to_value(u.TeV))
        # bin_center = bin_center[idx[0]:idx[1]]
        # bin_width = bin_width[idx[0]:idx[1]]
        
        mean_flux_range = mean_flux[idx[0]:idx[1]]
        x = bin_center[idx[0]:idx[1]]
        ax.scatter(x, mean_flux_range, color='blue', s=30, )
        # lower, upper = lower[idx[0]:idx[1]], upper[idx[0]:idx[1]]
        # lower_95, upper_95 = lower_95[idx[0]:idx[1]], upper_95[idx[0]:idx[1]]
        ax.axvspan(*fit_range.to_value('TeV'), color='blue', alpha=0.2)


    # ax1.legend()
    



def plot_counts(output_path, extracted_data, name):
    '''
    Plot the signal counts i the extraced data.
    '''
    signal_counts = [obs.on_vector.counts_in_safe_range.value for obs in extracted_data.spectrum_observations]
    signal_counts = np.sum(signal_counts, axis=0)

    bkg_counts = [obs.off_vector.counts_in_safe_range.value for obs in extracted_data.spectrum_observations]
    bkg_counts = np.sum(bkg_counts, axis=0)

    x = extracted_data.spectrum_observations[0].e_reco.lower_bounds.to_value('TeV')

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Count Spectrum for {name.upper()} telescope.')
    ax.set_title(f'total signal, background counts: {(signal_counts.sum(), bkg_counts.sum())}')
    ax.step(x, signal_counts, where='post', lw=2, label='signal counts', color='crimson')
    ax.step(x, bkg_counts, where='post', lw=2, label='background counts', color='gray')
    ax.set_xscale('log')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Counts')
    ax.legend()
    return fig, ax
