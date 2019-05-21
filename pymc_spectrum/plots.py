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


def plot_landscape(model, off_data, N=60):
    '''
    Plot the likelihood landscape of the given pymc model
    '''
    betas = np.linspace(0, 3, N)
    alphas = np.linspace(1.1, 4.0, N)
    f = model.logp
    zs = []
    a, b = np.meshgrid(alphas, betas)
    for al, be in tqdm(zip(a.ravel(), b.ravel())):

        p = f(
            amplitude_lowerbound__=np.log(4),
            alpha_lowerbound__=np.log(al),
            beta_lowerbound__=np.log(be),
            mu_b_lowerbound__=np.log(off_data + 0.1)
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


def plot_unfolding_result(trace, stacked_observation, fit_range):
    magic_model = Log10Parabola(
        amplitude=4.20 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=2.58 * u.Unit(''),
        beta=0.43 * u.Unit(''),
    )

    fact_model = Log10Parabola(
        amplitude=3.5 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=2.56* u.Unit(''),
        beta=0.4 * u.Unit(''),
    )

    hess_model = Log10Parabola(
        amplitude=4.47 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=2.39* u.Unit(''),
        beta=0.37 * u.Unit(''),
    )

    veritas_model = Log10Parabola(
        amplitude=3.76 * 1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.Unit('TeV'),
        alpha=2.44 * u.Unit(''),
        beta=0.26 * u.Unit(''),
    )


    e_center = stacked_observation.edisp.e_true.log_center().to_value(u.TeV)[1:-1]
    bin_width = stacked_observation.edisp.e_true.bin_width.to_value(u.TeV)[1:-1]

    norm = 1 / stacked_observation.aeff.data.data / stacked_observation.livetime / stacked_observation.edisp.e_true.bin_width
    norm = norm[1:-1]
    flux = trace['mu_s'][:, 1:-1] * norm
    mean_flux = np.median(flux, axis=0).to_value(1 / (u.TeV * u.s * u.cm**2))
    lower, upper = np.percentile(flux, [25, 75], axis=0)
    lower_95, upper_95 = np.percentile(flux, [5, 95], axis=0)

    dl = mean_flux - lower
    du = upper - mean_flux

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    magic_model.plot(energy_range=fit_range, ls='--', color='gray', label='magic', ax=ax)
    fact_model.plot(energy_range=fit_range, ls=':', color='silver', label='fact', ax=ax)
    hess_model.plot(energy_range=fit_range, ls='-.', color='darkgray', label='hess', ax=ax)
    veritas_model.plot(energy_range=fit_range, ls='-', color='lightgray', label='veritas', ax=ax)

    dl = mean_flux - lower_95
    du = upper_95 - mean_flux
    ax.errorbar(e_center, mean_flux, yerr=[dl, du], linestyle='', color='lightgray')

    dl = mean_flux - lower
    du = upper - mean_flux
    ax.errorbar(e_center, mean_flux, yerr=[dl, du], xerr=bin_width / 2, linestyle='')



    ax.legend()
    return fig, ax



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