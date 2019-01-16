import theano
import theano.tensor as T
import numpy as np
from scipy.integrate import quad, trapz, fixed_quad

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
        # self.xs = [np.linspace(a, b, num=3)]
        self.xs =  np.linspace(lower, upper, num=3)

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
