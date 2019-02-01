import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LogNorm

def plot_landscape(model, off_data, N = 40):
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
            amplitude_lowerbound__ = np.log(4),
            alpha_lowerbound__ = np.log(al),
            beta_lowerbound__= np.log(be),
            mu_b_lowerbound__=np.log(off_data + 0.1)
        )
        zs.append(p)

    zs = np.array(zs)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
    cf = ax1.contourf(a, b, zs.reshape(len(a), -1),  levels=100)
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('beta')
    fig.colorbar(cf, ax=ax1)

    cf = ax2.contourf(a, b, np.log10(-zs.reshape(len(a), -1)),  levels=255, )
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('beta')
    fig.colorbar(cf, ax=ax2)

    return fig, [ax1, ax2]


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
