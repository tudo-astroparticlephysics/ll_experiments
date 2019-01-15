import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_landscape(model, off_data):
    N = 25
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

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    cf = ax.contourf(a, b, zs.reshape(len(a), -1),  levels=124)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    fig.colorbar(cf, ax=ax)
    return fig, ax
