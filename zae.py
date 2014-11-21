import numpy
import theano
import theano.tensor as T

class Zae(object):
    """Zero-bias autoencoder"""

    def __init__(self, numvis, numhid, vistype, init_features, selectionthreshold=1.0):
        self.numvis = numvis
        self.numhid  = numhid
        self.vistype = vistype
        self.selectionthreshold = theano.shared(value=selectionthreshold, name='selectionthreshold')
        self.W_init = init_features.astype(theano.config.floatX)
        self.W = theano.shared(value = self.W_init, name='W')
        self.bvis = theano.shared(value=numpy.zeros(numvis, dtype=theano.config.floatX), name='bvis')
        self.inputs = T.matrix(name = 'inputs') 
        self.params = [self.W, self.bvis]

        self._prehiddens = T.dot(self.inputs, self.W) 
        self._hiddens = (self._prehiddens > self.selectionthreshold) * self._prehiddens
        if self.vistype == 'binary':
            self._outputs = T.nnet.sigmoid(T.dot(self._hiddens, self.W.T) + self.bvis)
            costpercase = -T.sum(self.inputs*T.log(self._outputs) + (1-self.inputs)*T.log(1-self._outputs), axis=1) 
        elif self.vistype == 'real':
            self._outputs = T.dot(self._hiddens, self.W.T) + self.bvis 
            costpercase = T.sum(0.5 * ((self.inputs - self._outputs)**2), axis=1) 

        self._cost = T.mean(costpercase)
        self._grads = T.grad(self._cost, self.params)

        self.cost = theano.function([self.inputs], self._cost)
        self.grad = theano.function([self.inputs], T.grad(self._cost, self.params))
        self.prehiddens = theano.function([self.inputs], self._prehiddens)
        self.hiddens = theano.function([self.inputs], self._hiddens)

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

