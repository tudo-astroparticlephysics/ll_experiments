import theano
import theano.tensor as T
import numpy as np


class IntegrateVectorizedGeneralized(theano.Op):
    '''
    A numerical integration routine using theano.
    This works without specifying the gradients explicetly.
    It requires setting the integration var to T.dvector
    instead of T.dscalar. This operation seems to be faster
    in standalone mode. However using pymc it gets horribly slow.
    I have no clue why that is.
    '''
    def __init__(self, expr, var, bins,  *inputs):
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
        self.xs = np.array([np.linspace(a, b, num=4) for a, b in zip(self.lower, self.upper)])
        self.delta_xs = np.array([np.diff(x) for x in self.xs])


    def make_node(self, *inputs):
        # assert len(self._extra_vars)  == len(inputs)
        return theano.Apply(self, list(inputs), [T.dvector().type()])

    def perform(self, node, inputs, out):
        vals = []
        t = self._func(self.xs.ravel(), *inputs).reshape(self.xs.shape)
        # t = np.array(ts)
        vals = 0.5 * np.sum((t[:, 0:-1] + t[:, 1:])*self.delta_xs, axis=1)
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
    def __init__(self, f, gradients_of_f, var, bins, *inputs, num_nodes=4):
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
        t = self.f(self, self.xs, *inputs)
        out[0][0] = self.trapz(t)

    def trapz(self, t):
        return 0.5 * np.sum((t[:, 0:-1] + t[:, 1:]) * self.delta_xs, axis=1)

    def trapz_theano(self, t):
        return 0.5 * T.sum((t[:, 0:-1] + t[:, 1:]) * self.delta_xs, axis=1)


    def L_op(self, inputs, output, output_grads):
        out, = output_grads

        dargs = []

        for gradient in self.gradients_of_f:
            t = gradient(self, self.xs, *inputs)
            darg = T.dot(out, self.trapz_theano(t))
            dargs.append(darg)

        return dargs



if __name__ == '__main__':
    import time
    import sys
    N = 100
    if len(sys.argv) == 2:
        N = int(sys.argv[1])


    def f(self, E, phi, alpha, beta):
        return phi*E**(-alpha-beta*np.log10(E))

    def df_dphi(self, E, phi, alpha, beta):
        return E**(-alpha-beta*np.log10(E))

    def df_dalpha(self, E, phi, alpha, beta):
        return -phi*E**(-alpha-beta*np.log10(E)) * np.log(E)

    def df_dbeta(self, E, phi, alpha, beta):
        return -(phi*E**(-alpha-beta*np.log10(E)) * np.log(E)**2)/np.log(10)


    amplitude_ = T.dscalar('amplitude_')
    alpha_ = T.dscalar('alpha_')
    beta_ = T.dscalar('beta_')


    bins = np.logspace(-2, 2, 20)

    amplitude = T.dscalar('amplitude')
    alpha = T.dscalar('alpha')
    beta = T.dscalar('beta')

    print('--'*30)
    print('Integrating Vectorized')
    print(f'Measuring {N} calls of eval')
    energy = T.dscalar('energy')


    integrator = IntegrateVectorized(f,[df_dphi, df_dalpha, df_dbeta], energy, bins, amplitude_, alpha_, beta_)

    t0 = time.time()
    for i in range(N):
        integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    test_result = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})

    print(f'Measuring {N} calls of jacobi')
    integrator = IntegrateVectorized(f,[df_dphi, df_dalpha, df_dbeta], energy, bins, amplitude_, alpha_, beta_)
    T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t0 = time.time()
    for i in range(N):
        T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (for {len(bins)} bins)')
    test_result_jacobian = T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})


    print('--'*30)
    print('Integrating Vectorized Slow (non vectorized)')
    energy = T.dscalar('energy')

    integrators = [IntegrateVectorized(f, [df_dphi, df_dalpha, df_dbeta], energy, np.array(b), amplitude_, alpha_, beta_) for b in zip(bins[:-1], bins[1:])]

    t0 = time.time()
    for i in range(N):
        for integrator in integrators:
            integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    
    print(f'Measuring {N} calls of jacobi')
    integrators = [IntegrateVectorized(f, [df_dphi, df_dalpha, df_dbeta], energy, np.array(b), amplitude_, alpha_, beta_) for b in zip(bins[: -1], bins[1:])]
    T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t0 = time.time()
    for i in range(N):
        for integrator in integrators:
            T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (for {len(bins)} bins)')


    print('--'*30)
    print('Integrating Vectorized Generalized')
    print(f'Measuring {N} calls of eval')
    energy = T.dvector('energy')
    func = amplitude_ * energy **(-alpha_ - beta_ * T.log10(energy))

    integrator = IntegrateVectorizedGeneralized(func, energy, bins, amplitude_, alpha_, beta_)
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
