import numpy
import theano
import theano.tensor as T

class Zae(object):
    def __init__(self, rng, numvis, numhid, vistype, numsteps, init_features, selectionthreshold=1.0):
        self.numvis = numvis
        self.numhid  = numhid
        self.vistype = vistype
        self.numsteps = numsteps
        self.rng = rng
        self.selectionthreshold = theano.shared(value=selectionthreshold, name='selectionthreshold')

        self.W_init = init_features.astype(theano.config.floatX)
        self.W = theano.shared(value = self.W_init, name='W')
        self.Wfeedback = theano.shared(value = self.W_init+0.01*self.rng.randn(self.numvis, self.numhid).astype(theano.config.floatX), name='Wfeedback')
        self.bvis = theano.shared(value=numpy.zeros(numvis, dtype=theano.config.floatX), name='bvis')
        self.inputs = T.matrix(name = 'inputs') 
        #self.params = [self.W, self.bvis]
        self.params = [self.W, self.Wfeedback]

        self._canvas = [T.zeros((self.inputs.shape))] + [None] * (self.numsteps-1)
        for t in range(1, numsteps):
            self._prehiddens = T.dot(self.inputs, self.W) 
            self._hiddens = (self._prehiddens > self.selectionthreshold) * self._prehiddens + T.dot(self._canvas[t-1], self.Wfeedback)
            self._canvas[t] = self._canvas[t-1] + T.dot(self._hiddens, self.W.T) + self.bvis 

        if self.vistype == 'binary':
            self._canvas = T.nnet.sigmoid(self._canvas)
            costpercase = -T.sum(self.inputs*T.log(self._canvas) + (1-self.inputs)*T.log(1-self._canvas), axis=1) 
        elif self.vistype == 'real':
            costpercase = T.sum(0.5 * ((self.inputs - self._canvas)**2), axis=1) 

        self._cost = T.mean(costpercase)
        self._grads = T.grad(self._cost, self.params)

        self.cost = theano.function([self.inputs], self._cost)
        self.grad = theano.function([self.inputs], T.grad(self._cost, self.params))
        self.prehiddens = theano.function([self.inputs], self._prehiddens)
        self.hiddens = theano.function([self.inputs], self._hiddens)
        self.recons_from_prehiddens = theano.function([self._prehiddens, self._canvas], self._canvas)
        #self.recons_from_inputs = theano.function([self.inputs], self._canvas)
        self.selection = theano.function([self.inputs], (self._prehiddens > self.selectionthreshold))

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))

