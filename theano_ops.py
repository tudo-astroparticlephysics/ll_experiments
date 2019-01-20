import theano
import theano.tensor as T
import numpy as np
from scipy.integrate import trapz
from symbolic_gradients import logpar_integral, dphi, dalpha, dbeta


class IntegrateVectorizedOld(theano.Op):
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
        self.xs = np.array([np.linspace(a, b, num=4) for a, b in zip(self.lower, self.upper)])
        self.delta_xs = np.array([np.diff(x) for x in self.xs])


    def make_node(self, *inputs):
        # assert len(self._extra_vars)  == len(inputs)
        return theano.Apply(self, list(inputs), [T.dvector().type()])


    def perform(self, node, inputs, out):
        vals = []
        for i, (x, delta_x) in enumerate(zip(self.xs, self.delta_xs)):
            y = np.array([self._func(i , *inputs) for i in x])

            v = 0.5 * np.sum((y[0:-1] + y[1:])*delta_x)
            # v = trapz(y, x)
            # print(v, trapz(y, x))
            vals.append(v)
        #
        # # print(np.array(inputs), self.lower, self.upper)
        vals  = np.array(vals)
        # from IPython import embed; embed()
        # res = logpar_integral(self.lower, self.upper, *np.array(inputs))
        # print(res.shape, vals.shape)
        # vals = theano.map(lambda x: trapz([self._func(i , *inputs) for i in x], x), sequences=[self.xs, ])
        out[0][0] = vals

    def L_op(self, inputs, output,  grads):
        if not hasattr(self, 'precomputed_grads'):
            self.precomputed_grads = T.grad(self._expr, self._extra_vars)
        out, = grads
        dargs = []
        for grad in self.precomputed_grads:
            integrate = IntegrateVectorized(grad, self._var, self.bins, *self._extra_vars)
            darg = T.dot(out,  integrate(*inputs))
            dargs.append(darg)
        return dargs


class IntegrateVectorized(theano.Op):
    '''
    A numerical integration routine using theano. This is very fragile code.
    Not only because theano is in low power maintenance mode.
    '''
    def __init__(self, expr, var, bins,  *inputs, name=None):
        super().__init__()
        self._expr = expr
        self._var = var
        self._extra_vars = inputs
        self._func = theano.function(
            [var] + list(self._extra_vars),
            self._expr,
            on_unused_input='ignore'
        )
        self._name = name
        self.lower = bins[0:-1]
        self.upper = bins[1:]
        self.bins = bins
        self.xs = np.array([np.linspace(a, b, num=4) for a, b in zip(self.lower, self.upper)])
        self.delta_xs = np.array([np.diff(x) for x in self.xs])


    def make_node(self, *inputs):
        # assert len(self._extra_vars)  == len(inputs)
        return theano.Apply(self, list(inputs), [T.dvector().type()])

    def f(self, E, phi, alpha, beta):
        return phi*E**(-alpha-beta*np.log10(E))

    def df_dphi(self, E, phi, alpha, beta):
        return E**(-alpha-beta*np.log10(E))

    def df_dalpha(self, E, phi, alpha, beta):
        return -phi*E**(-alpha-beta*np.log10(E)) * np.log(E)

    def df_dbeta(self, E, phi, alpha, beta):
        return -(phi*E**(-alpha-beta*np.log10(E)) * np.log(E)**2)/np.log(10)


    def perform(self, node, inputs, out):
        if not self._name:
            t = self.f(self.xs, *inputs)
        elif self._name == 'dphi':
            t = self.df_dphi(self.xs, *inputs)
        elif self._name == 'dalpha':
            t = self.df_dalpha(self.xs, *inputs)
        elif self._name == 'dbeta':
            t = self.df_dalpha(self.xs, *inputs)
        # from IPython import embed; embed()
        v_vec = 0.5 * np.sum((t[:, 0:-1] + t[:, 1:])*self.delta_xs, axis=1)
        out[0][0] = v_vec

    def L_op(self, inputs, output,  output_grads):
        if not hasattr(self, 'precomputed_grads'):
            self.precomputed_grads = T.grad(self._expr, self._extra_vars)
        out, = output_grads

        dargs = []
        for grad, name in zip(self.precomputed_grads, ['dphi', 'dalpha', 'dbeta']):
            integrate = IntegrateVectorized(grad, self._var, self.bins, *self._extra_vars, name=name)
            darg = T.dot(out,  integrate(*inputs))
            dargs.append(darg)
        return dargs


# class Integrate(theano.Op):
#     '''
#     A numerical integration routine using theano. This is very fragile code.
#     Not only because theano is in low power maintenance mode.
#     '''
#     def __init__(self, expr, var, lower, upper, *inputs):
#         super().__init__()
#         self._expr = expr
#         self._var = var
#         self._extra_vars = inputs
#         self.lower = lower
#         self.upper = upper
#         self._func = theano.function(
#             [var] + list(self._extra_vars),
#             self._expr,
#             on_unused_input='ignore'
#         )
#         self.xs =  np.linspace(lower, upper, num=4)
#
#     def make_node(self, *inputs):
#         # assert len(self._extra_vars)  == len(inputs)
#         return theano.Apply(self, list(inputs), [T.dscalar().type()])
#
#     def perform(self, node, inputs, out):
#         x = self.xs
#         y = [self._func(i , *inputs) for i in x]
#         val = trapz(y, x)
#         out[0][0] = np.array(val)
#
#     def grad(self, inputs, grads):
#         out, = grads
#         grads = T.grad(self._expr, self._extra_vars)
#         dargs = []
#         for grad in grads:
#             integrate = Integrate(grad, self._var, self.lower, self.upper, *self._extra_vars)
#             darg = out * integrate(*inputs)
#             dargs.append(darg)
#
#         return dargs


if __name__ == '__main__':
    import time
    import sys
    N = 100
    if len(sys.argv) == 2:
        N = int(sys.argv[1])


    amplitude_ = T.dscalar('amplitude_')
    alpha_ = T.dscalar('alpha_')
    beta_ = T.dscalar('beta_')


    bins = np.logspace(-2, 2, 20)

    amplitude = T.dscalar('amplitude')
    alpha = T.dscalar('alpha')
    beta = T.dscalar('beta')
    print('GradOp test')

    # f = GradientOp(lower=1, upper=2)
    # r = f(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.5, beta: 0.5})
    # print(r)
    #
    # print('Calculating Integral analytically. (single bin)')
    # integrator = IntegrateTest(func, energy, bins[0], bins[1], amplitude_, alpha_, beta_)
    # print(f'Measuring {N} calls of eval. (single bin)')
    # t0 = time.time()
    # for i in np.linspace(1, 4, N):
    #     result_analytical = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: i, beta: 0.5})
    # t1 = time.time()
    # print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (single bin)')
    #
    # print(f'Measuring {N} calls of grad. (single bin)')
    # integrator = IntegrateTest(func, energy, bins[0], bins[1], amplitude_, alpha_, beta_)
    # t0 = time.time()
    # for i in range(N):
    #     T.grad(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # t1 = time.time()
    # print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    # print('--'*30)

    # sys.exit()

    # print('Calculating Integral analytically.')
    # integrator = IntegrateLogParabolaAnalytically(energy, bins, amplitude_, alpha_, beta_)
    #
    # print(f'Measuring {N} calls of eval')
    # t0 = time.time()
    # for i in np.linspace(1, 4, N):
    #     result_analytical = integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: i, beta: 0.5})
    # t1 = time.time()
    # print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    #
    # print(f'Measuring {N} calls of grad')
    # integrator =  IntegrateLogParabolaAnalytically(energy, bins, amplitude_, alpha_, beta_)
    # T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # t0 = time.time()
    # for i in range(N):
    #     T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # t1 = time.time()
    # print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    # print('--'*30)

    # print('Calculating Integral piecewise.')
    # integrator = Integrate(func, energy, bins[0], bins[1], amplitude_, alpha_, beta_)
    # print(f'Measuring {N} calls of eval (a single bin)')
    # t0 = time.time()
    # for i in range(N):
    #     integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # t1 = time.time()
    # print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    #
    # print(f'Measuring {N} calls of grad (a single bin)')
    # integrator = Integrate(func, energy, bins[0], bins[1], amplitude_, alpha_, beta_)
    # t0 = time.time()
    # for i in range(N):
    #     T.grad(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    # t1 = time.time()
    # print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')
    #

        #
    print('--'*30)
    print('Integrating Vectorized')
    print(f'Measuring {N} calls of eval')
    energy = T.dscalar('energy')
    func = amplitude_ * energy **(-alpha_ - beta_ * T.log10(energy))

    integrator = IntegrateVectorized(func, energy, bins, amplitude_, alpha_, beta_)
    t0 = time.time()
    for i in range(N):
        integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')

    print(f'Measuring {N} calls of jacobi')
    integrator = IntegrateVectorized(func, energy, bins, amplitude_, alpha_, beta_)
    T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t0 = time.time()
    for i in range(N):
        T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (for {len(bins)} bins)')



    print('--'*30)
    print('Integrating Vectorized Old')
    print(f'Measuring {N} calls of eval')
    energy = T.dscalar('energy')
    func = amplitude_ * energy **(-alpha_ - beta_ * T.log10(energy))

    integrator = IntegrateVectorizedOld(func, energy, bins, amplitude_, alpha_, beta_)
    t0 = time.time()
    for i in range(N):
        integrator(amplitude, alpha, beta).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()
    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total')

    print(f'Measuring {N} calls of jacobi')
    integrator = IntegrateVectorizedOld(func, energy, bins, amplitude_, alpha_, beta_)
    T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t0 = time.time()
    for i in range(N):
        T.jacobian(integrator(amplitude, alpha, beta), amplitude).eval({amplitude: 4.0, alpha: 2.0, beta: 0.5})
    t1 = time.time()

    print(f'Takes approximately  {(t1-t0) / N} seconds per iteration, {(t1-t0)} seconds in total (for {len(bins)} bins)')
