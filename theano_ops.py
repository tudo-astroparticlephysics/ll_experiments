import theano
import theano.tensor as T
import numpy as np
from scipy.integrate import quad
from scipy.special import erf


class IntegrateVectorizedGeneralized(theano.Op):
    '''
    A numerical integration routine using theano.
    This works without specifying the gradients explicetly.
    It requires setting the integration var to T.dvector
    instead of T.dscalar. This operation seems to be faster
    in standalone mode. However using pymc it gets horribly slow.
    I have no clue why that is.
    '''
    def __init__(self, expr, var, bins, *inputs):
        super().__init__()
        self._expr = expr
        self._var = var
        self._extra_vars = inputs
        self._func = theano.function(
            [var] + list(self._extra_vars),
            self._expr,
            on_unused_input='ignore'
        )
        self.lower = bins[0:-1]
        self.upper = bins[1:]
        self.bins = bins
        self.xs = np.array([np.linspace(a, b, num=5) for a, b in zip(self.lower, self.upper)])
        self.delta_xs = np.array([np.diff(x) for x in self.xs])


    def make_node(self, *inputs):
        # assert len(self._extra_vars)  == len(inputs)
        return theano.Apply(self, list(inputs), [T.dvector().type()])

    def perform(self, node, inputs, out):
        vals = []
        t = self._func(self.xs.ravel(), *inputs).reshape(self.xs.shape)
        # t = np.array(ts)
        vals = 0.5 * np.sum((t[:, 0:-1] + t[:, 1:]) * self.delta_xs, axis=1)
        out[0][0] = vals

    def L_op(self, inputs, output, grads):
        # from IPython import embed; embed()
        if not hasattr(self, 'precomputed_grads'):
            grad_integrators = T.jacobian(self._expr, self._extra_vars)
            self.precomputed_grads = [IntegrateVectorizedGeneralized(gi, self._var, self.bins, *self._extra_vars) for gi in grad_integrators]

        out, = grads
        dargs = []
        for integrate in self.precomputed_grads:
            darg = T.dot(out, integrate(*inputs))
            # print(darg)
            dargs.append(darg)
        return dargs


class IntegrateVectorized(theano.Op):
    '''
    A numerical integration routine using theano.
    you need to pass the function 'f' to integrate and its gradients 'gradients_of_f'
    w.r.t to the vartiables that you're not integrating over.

    These functions should be numpy friendly and vectorized to work on multiple
    values of 'var'

    '''
    def __init__(self, f, gradients_of_f, var, bins, *inputs, num_nodes=5):
        super().__init__()
        self.f = f
        self.gradients_of_f = gradients_of_f

        self._var = var
        self._extra_vars = inputs
        num_bins = len(bins) - 1
        bin_widths = np.diff(bins)

        d = np.tile(np.linspace(0, 1, num=num_nodes), num_bins).reshape(num_bins, -1) 
        self.xs = (d * bin_widths[:, None]) + bins[0:-1, None] 
        self.delta_xs = np.diff(self.xs)

    def make_node(self, *inputs):
        return theano.Apply(self, list(inputs), [T.dvector().type()])

    def perform(self, node, inputs, out):
        t = self.f(self.xs, *inputs)
        out[0][0] = self.trapz(t)

    def trapz(self, t):
        return 0.5 * np.sum((t[:, 0:-1] + t[:, 1:]) * self.delta_xs, axis=1)

    def trapz_theano(self, t):
        return 0.5 * T.sum((t[:, 0:-1] + t[:, 1:]) * self.delta_xs, axis=1)

    def L_op(self, inputs, output, output_grads):
        out, = output_grads

        dargs = []

        for gradient in self.gradients_of_f:
            t = gradient(self.xs, *inputs)
            darg = T.dot(out, self.trapz_theano(t))
            dargs.append(darg)

        return dargs


def _int(alpha, beta, lower, upper):
    erf_arg = np.sqrt(beta) + (alpha - 1) / (2 * np.sqrt(beta))
    exp_arg = ((alpha - 1)**2) / (4 * beta)
    # print((erf(upper*erf_arg) - erf(lower*erf_arg)))
    r = 0.5 * np.sqrt(np.pi / beta) * np.exp(exp_arg) * (erf(upper*erf_arg) - erf(lower*erf_arg))
    return r


def log_par_integral(bins, N, alpha, beta):
    u = np.log(bins)
    b = beta / np.log(10)
    return N * _int(alpha, b, u[0:-1], u[1:])


def _int_theano(alpha, beta, lower, upper):
    erf_arg = T.sqrt(beta) + (alpha - 1) / (2 * T.sqrt(beta))
    exp_arg = ((alpha - 1)**2) / (4 * beta)
    r = 0.5 * T.sqrt(np.pi / beta) * T.exp(exp_arg) * (T.erf(upper*erf_arg) - T.erf(lower*erf_arg))
    return r


def log_par_integral_theano(bins, N, alpha, beta):
    u = T.log(bins)
    b = beta / T.log(10)
    return N * _int_theano(alpha, b, u[0:-1], u[1:])



if __name__ == '__main__':
    import time
    import sys
    import matplotlib.pyplot as plt

    N = 100
    if len(sys.argv) == 2:
        N = int(sys.argv[1])


    def f(E, phi, alpha, beta):
        return phi * E**(-alpha - beta * np.log10(E))

    def df_dphi(E, phi, alpha, beta):
        return E**(-alpha - beta * np.log10(E))

    def df_dalpha(E, phi, alpha, beta):
        return -phi * E**(-alpha - beta * np.log10(E)) * np.log(E)

    def df_dbeta(E, phi, alpha, beta):
        return -(phi * E**(-alpha - beta * np.log10(E)) * np.log(E)**2) / np.log(10)


    amplitude_ = T.dscalar('amplitude_')
    alpha_ = T.dscalar('alpha_')
    beta_ = T.dscalar('beta_')

    bins = np.logspace(-2, 2, 50)

    amplitude = T.dscalar('amplitude')
    alpha = T.dscalar('alpha')
    beta = T.dscalar('beta')
    
    from tqdm import tqdm
    print('Testing equality')
    print(np.log(bins))
    invalid_coords = []
    valid_coords = []
    zero_coords = []
    nan_coords = []
    for b_param in tqdm(np.linspace(0.01, 7, 100)):
        for a_param in np.linspace(0.01, 7, 100):
            reference_result = np.array([quad(lambda e: f(e, 4.0, a_param, b_param), a=a, b=b)[0] for a, b in zip(bins[:-1], bins[1:])])
            analytical_result = log_par_integral(bins, 4.0, a_param, b_param)
            if np.allclose(reference_result, analytical_result):
                valid_coords.append([a_param, b_param])
            else:
                invalid_coords.append([a_param, b_param])
            if np.isnan(analytical_result).any():
                nan_coords.append([a_param, b_param])
            if (analytical_result == 0).all():
                # print('Had zero!')
                zero_coords.append([a_param, b_param])
            
    # print(reference_result)
    # print(analytical_result)
    invalid_coords = np.array(invalid_coords)
    valid_coords = np.array(valid_coords)
    zero_coords = np.array(zero_coords)
    nan_coords = np.array(nan_coords)
    _, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    ax1.scatter(invalid_coords[:, 0], invalid_coords[:, 1], color='yellow', label='result != truth')
    ax1.scatter(zero_coords[:, 0], zero_coords[:, 1], color='orange', label='Zero Result')
    ax1.scatter(nan_coords[:, 0], nan_coords[:, 1], color='crimson', label='NaN result')
    ax1.scatter(valid_coords[:, 0], valid_coords[:, 1], color='green', label='Valid result')
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.legend()
    # ax2.hist2d(invalid_coords[:, 0], invalid_coords[:, 1], bins=20)
    plt.savefig('validity.pdf')

    1/0
    # analytical_result = log_par_integral(bins, 0.1 * 1E-11, .1, 0.1)
    # print(analytical_result)

    print('--' * 30)
    print('Integrating Analytically')
    energy = T.dscalar('energy')
    integration_result = log_par_integral_theano(bins, amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # integration_result = integrator(amplitude, alpha, beta)
    # print(integration_result)
    print(f'Measuring {N} calls of eval')
    t0 = time.time()
    for i in range(N):
        log_par_integral_theano(bins, amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    # test_result = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})

    print(f'Measuring {N} calls of jacobi')
    # integrator = IntegrateVectorized(f, [df_dphi, df_dalpha, df_dbeta], energy, bins, amplitude_, alpha_, beta_)
    T.jacobian(log_par_integral_theano(bins, amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t0 = time.time()
    for i in range(N):
        T.jacobian(log_par_integral_theano(bins, amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (for {len(bins)} bins)')
    # test_result_jacobian = T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})


    print('--' * 30)
    print('Integrating Vectorized')
    energy = T.dscalar('energy')
    integrator = IntegrateVectorized(f, [df_dphi, df_dalpha, df_dbeta], energy, bins, amplitude_, alpha_, beta_)
    integration_result = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # print(integration_result)
    print(f'Measuring {N} calls of eval')

    t0 = time.time()
    for i in range(N):
        integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    test_result = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})

    print(f'Measuring {N} calls of jacobi')
    integrator = IntegrateVectorized(f, [df_dphi, df_dalpha, df_dbeta], energy, bins, amplitude_, alpha_, beta_)
    T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t0 = time.time()
    for i in range(N):
        T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (for {len(bins)} bins)')
    test_result_jacobian = T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})


        # print('--'*30)
    # print('Integrating Vectorized Slow (non vectorized)')
    # energy = T.dscalar('energy')

    # integrators = [IntegrateVectorized(f, [df_dphi, df_dalpha, df_dbeta], energy, np.array(b), amplitude_, alpha_, beta_) for b in zip(bins[:-1], bins[1:])]

    # t0 = time.time()
    # for i in range(N):
    #     for integrator in integrators:
    #         integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # t1 = time.time()
    # print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    
    # print(f'Measuring {N} calls of jacobi')
    # integrators = [IntegrateVectorized(f, [df_dphi, df_dalpha, df_dbeta], energy, np.array(b), amplitude_, alpha_, beta_) for b in zip(bins[: -1], bins[1:])]
    # T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # t0 = time.time()
    # for i in range(N):
    #     for integrator in integrators:
    #         T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # t1 = time.time()

    # print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (for {len(bins)} bins)')


    print('--'*30)
    print('Integrating Vectorized Generalized')
    print(f'Measuring {N} calls of eval')
    energy = T.dvector('energy')
    func = amplitude_ * energy **(-alpha_ - beta_ * T.log10(energy))

    integrator = IntegrateVectorizedGeneralized(func, energy, bins, amplitude_, alpha_, beta_)
    integration_result_generalized = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t0 = time.time()
    for i in range(N):
        integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    old_test_result = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    print(f'Measuring {N} calls of jacobi')
    integrator = IntegrateVectorizedGeneralized(func, energy, bins, amplitude_, alpha_, beta_)
    T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t0 = time.time()
    for i in range(N):
        T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (for {len(bins)} bins)')
    old_test_result_jacobian = T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})

    assert np.allclose(test_result, old_test_result)
    assert np.allclose(test_result_jacobian, old_test_result_jacobian)

    f, [ax1, ax2] = plt.subplots(2, 1, figsize=(7, 7))
    ax1.plot((bins[:-1] + bins[1:]) / 2, reference_result / integration_result, color='gray', lw=1)
    ax1.plot((bins[:-1] + bins[1:]) / 2, reference_result / integration_result, '.', color='crimson')
    ax1.set_xlabel('energy')
    ax2.plot((bins[:-1] + bins[1:]) / 2, (integration_result_generalized - reference_result) / reference_result, color='gray', lw=1)
    ax2.plot((bins[:-1] + bins[1:]) / 2, (integration_result_generalized - reference_result) / reference_result, '.', color='crimson')
    ax2.set_xscale('log')
    ax2.set_xlabel('energy')
    plt.savefig('integration_results.pdf')