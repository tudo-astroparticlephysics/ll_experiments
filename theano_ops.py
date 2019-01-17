import theano
import theano.tensor as T
import numpy as np
from scipy.integrate import quad, trapz, fixed_quad


class IntegrateVectorized(theano.Op):
    '''
    A numerical integration routine using theano. This is very fragile code.
    Not only because theano is in low power maintenance mode.
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
        self.xs = [np.linspace(a, b, num=4) for a, b in zip(self.lower, self.upper)]


    def make_node(self, *inputs):
        # assert len(self._extra_vars)  == len(inputs)
        return theano.Apply(self, list(inputs), [T.dvector().type()])

    def perform(self, node, inputs, out):

        vals = []
        for i, x in enumerate(self.xs):
            y = [self._func(i , *inputs) for i in x]
            vals.append(trapz(y, x))

        # vals = theano.map(lambda x: trapz([self._func(i , *inputs) for i in x], x), sequences=[self.xs, ])
        out[0][0] = np.array(vals)

    def grad(self, inputs, grads):
        if not hasattr(self, 'precomputed_grads'):
            self.precomputed_grads = T.grad(self._expr, self._extra_vars)
        out, = grads
        dargs = []
        for grad in self.precomputed_grads:
            integrate = IntegrateVectorized(grad, self._var, self.bins, *self._extra_vars)
            darg = T.dot(out,  integrate(*inputs))
            dargs.append(darg)
        return dargs



class Integrate(theano.Op):
    '''
    A numerical integration routine using theano. This is very fragile code.
    Not only because theano is in low power maintenance mode.
    '''
    def __init__(self, expr, var, lower, upper, *inputs):
        super().__init__()
        self._expr = expr
        self._var = var
        self._extra_vars = inputs
        self.lower = lower
        self.upper = upper
        self._func = theano.function(
            [var] + list(self._extra_vars),
            self._expr,
            on_unused_input='ignore'
        )
        self.xs =  np.linspace(lower, upper, num=4)

    def make_node(self, *inputs):
        # assert len(self._extra_vars)  == len(inputs)
        return theano.Apply(self, list(inputs), [T.dscalar().type()])

    def perform(self, node, inputs, out):
        x = self.xs
        y = [self._func(i , *inputs) for i in x]
        val = trapz(y, x)
        out[0][0] = np.array(val)

    def grad(self, inputs, grads):
        out, = grads
        grads = T.grad(self._expr, self._extra_vars)
        dargs = []
        for grad in grads:
            integrate = Integrate(grad, self._var, self.lower, self.upper, *self._extra_vars)
            darg = out * integrate(*inputs)
            dargs.append(darg)

        return dargs


if __name__ == '__main__':
    import time

    energy = T.dscalar('energy')
    amplitude_ = T.dscalar('amplitude_')
    alpha_ = T.dscalar('alpha_')
    beta_ = T.dscalar('beta_')

    func = amplitude_ * energy **(-alpha_ - beta_ * T.log10(energy))

    bins = np.logspace(-2, 2, 20)

    amplitude = T.dscalar('amplitude')
    alpha = T.dscalar('alpha')
    beta = T.dscalar('beta')

    print('Calculating Integral piecewise.')
    t0 = time.time()
    result_elemetwise = []
    for a, b, in zip(bins[0:-1], bins[1:]):
        integrator = Integrate(func, energy, a, b, amplitude_, alpha_, beta_)
        result_elemetwise.append(integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5}))
        T.grad(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})

    t1 = time.time()
    print(f'Finished in {t1-t0}')

    N = 100
    print(f'Measuring {N} calls of eval')
    t0 = time.time()
    integrator = Integrate(func, energy, bins[0], bins[1], amplitude_, alpha_, beta_)
    for i in range(N):
        integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')

    print(f'Measuring {N} calls of grad')
    t0 = time.time()
    integrator = Integrate(func, energy, bins[0], bins[1], amplitude_, alpha_, beta_)
    for i in range(N):
        T.grad(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')

    print('--'*30)
    energy = T.dscalar('energy')
    amplitude_ = T.dscalar('amplitude_')
    alpha_ = T.dscalar('alpha_')
    beta_ = T.dscalar('beta_')

    func = amplitude_ * energy **(-alpha_ - beta_ * T.log10(energy))

    bins = np.logspace(-2, 2, 20)

    amplitude = T.dscalar('amplitude')
    alpha = T.dscalar('alpha')
    beta = T.dscalar('beta')
    print('Integrating Vecotrized')
    t0 = time.time()

    integrator = IntegrateVectorized(func, energy, bins, amplitude_, alpha_, beta_)
    result = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    r = T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})

    t1 = time.time()
    print(f'Finished in {t1-t0}')

    print(f'Measuring {N} calls of eval')
    t0 = time.time()
    integrator = IntegrateVectorized(func, energy, bins, amplitude_, alpha_, beta_)
    for i in range(N):
        integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')

    print(f'Measuring {N} calls of jacobi')
    t0 = time.time()
    integrator = IntegrateVectorized(func, energy, bins, amplitude_, alpha_, beta_)
    for i in range(N):
        T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    np.testing.assert_allclose(result, result_elemetwise)

    # f = theano.function([amplitude, alpha, beta], integrator(amplitude, alpha, beta), profile=True)
    # f.profile.summary()
    # model.profile(model.logpt).summary()
    # print(result)
