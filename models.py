import numpy as np
from symbolic_gradients import dbeta, dalpha, dphi, logpar_integral, logpar
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import time
    import sys

    # maxima_solution =  0.6504851386166309
    # np.testing.assert_allclose(maxima_solution,logpar_log10(2, 4, 2.5, 0.4))
    N = 50
    if len(sys.argv) == 2:
        N = sys.argv[1]

    from IPython import embed; embed()
    xs = np.logspace(-2, 2, N)

    print('Calculating integral analyticaly.')
    res_an = []
    t0 = time.time()
    for l, u in zip(xs[0:-1], xs[1:]):
        res_an.append(logpar_integral(l, u, 4, 2.5, 0.4))
    t1 = time.time()
    res_an = np.array(res_an)
    print(f'Finished in {t1-t0}. Thats {(t1-t0) / N} seconds per call')

    print('Calculating integral using trapz.')
    res_trapz = []
    t0 = time.time()
    for l, u in zip(xs[0:-1], xs[1:]):
        x = np.linspace(l, u, 25)
        y = logpar(x, 4, 2.5, 0.4)
        res_trapz.append(np.trapz(y, x))
    t1 = time.time()
    res_trapz = np.array(res_trapz)
    print(f'Finished in {t1-t0}. Thats {(t1-t0) / N} seconds per call')
    print('Quad solution:')
    print(res_trapz)

    print('maxima solution:')
    print(res_an)
    # from IPython import embed; embed()
    np.testing.assert_allclose(res_an, res_trapz, rtol=0.01,)


    print('Calculating dalpha.')
    t0 = time.time()
    for l, u in zip(xs[0:-1], xs[1:]):
        dalpha(l, u, 4E-11, 2.5, 0.4)
    t1 = time.time()
    print(f'Finished in {t1-t0}. Thats {(t1-t0) / N} seconds per call')


    print('Calculating dbeta.')
    t0 = time.time()
    for l, u in zip(xs[0:-1], xs[1:]):
        dbeta(l, u, 4E-11, 2.5, 0.4)
    t1 = time.time()
    print(f'Finished in {t1-t0}. Thats {(t1-t0) / N} seconds per call')


    print('Calculating dphi.')
    t0 = time.time()
    for l, u in zip(xs[0:-1], xs[1:]):
        dphi(l, u, 4E-11, 2.5, 0.4)
    t1 = time.time()
    print(f'Finished in {t1-t0}. Thats {(t1-t0) / N} seconds per call')
#     plot_gradients()
#
# def plot_gradients():

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)

    dbs = []
    dbs_ana = []
    xs=np.linspace(1.5, 3.5, 100)
    for alpha in xs:
        xk = [1, 2, 4, alpha,  0.4]
        dbs.append(approx_fprime(xk, lambda x: logpar_integral(*x), 0.001)[-1])
        dbs_ana.append(dbeta(*xk))

    ax1.plot(xs, dbs, label='numeric')
    ax1.plot(xs, dbs_ana, label='symbolic')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('dbeta')
    ax1.legend()

    dbs = []
    dbs_ana = []
    xs=np.linspace(0.1, 0.7, 100)
    for beta in xs:
        xk = [1, 2, 4, 2.5,  beta]
        dbs.append(approx_fprime(xk, lambda x: logpar_integral(*x), 0.001)[-2])
        dbs_ana.append(dalpha(*xk))

    ax2.plot(xs, dbs, label='numeric')
    ax2.plot(xs, dbs_ana, label='symbolic')
    ax2.set_xlabel('beta')
    ax2.set_ylabel('dalpha')
    ax2.legend()

    dbs = []
    dbs_ana = []
    xs=np.linspace(0.1, 0.7, 100)
    for beta in xs:
        xk = [1, 2, 4, 2.5,  beta]
        dbs.append(approx_fprime(xk, lambda x: logpar_integral(*x), 0.001)[-3])
        dbs_ana.append(dphi(*xk))

    ax3.plot(xs, dbs, label='numeric')
    ax3.plot(xs, dbs_ana, label='symbolic')
    ax3.set_xlabel('beta')
    ax3.set_ylabel('dphi')
    ax3.legend()


    dbs = []
    dbs_ana = []
    xs=np.linspace(1.5, 3.5, 100)
    for alpha in xs:
        xk = [1, 2, 4, alpha,  0.4]
        dbs.append(approx_fprime(xk, lambda x: logpar_integral(*x), 0.001)[-3])
        dbs_ana.append(dphi(*xk))

    ax4.plot(xs, dbs, label='numeric')
    ax4.plot(xs, dbs_ana, label='symbolic')
    ax4.set_xlabel('alpha')
    ax4.set_ylabel('dphi')
    ax4.legend()


    plt.show()
    # from IPython import embed; embed()
