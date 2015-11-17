import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Zae(object):
    """Zero-bias autoencoder"""

    def __init__(self, numvis, numhid, vistype, init_features, selectionthreshold=1.0, weightcost=0.0, dropout=0.5):
        self.numvis = numvis
        self.numhid  = numhid
        self.dropout = dropout
        self.vistype = vistype
        self.weightcost = weightcost
        self.selectionthreshold = theano.shared(value=selectionthreshold, name='selectionthreshold')
        self.W_init = init_features.astype(theano.config.floatX)
        self.W = theano.shared(value = self.W_init, name='W')
        self.bvis = theano.shared(value=numpy.zeros(numvis, dtype=theano.config.floatX), name='bvis')
        self.inputs = T.matrix(name = 'inputs') 
        self.params = [self.W, self.bvis]

        self.theano_rng = RandomStreams(1)

        self._prehiddens = T.dot(self.inputs, self.W) 
        self._hiddens = (self._prehiddens > self.selectionthreshold) * self._prehiddens

        self._dropoutmask = self.theano_rng.binomial(size=(self.inputs.shape[0], self.numhid), n=1, p=self.dropout, dtype=theano.config.floatX) 

        self._outputs = T.dot(self._dropoutmask*self._hiddens, (1/self.dropout)*self.W.T) + self.bvis 

        if self.vistype == 'binary':
            self._outputs = T.nnet.sigmoid(self._outputs)
            costpercase = -T.sum(self.inputs*T.log(self._outputs) + (1-self.inputs)*T.log(1-self._outputs), axis=1) 
        elif self.vistype == 'real':
            costpercase = T.sum(0.5 * ((self.inputs - self._outputs)**2), axis=1) 

        self._cost = T.mean(costpercase)
        self._cost += self.weightcost * T.sum(self.W**2)
        self._grads = T.grad(self._cost, self.params)

        self.cost = theano.function([self.inputs], self._cost)
        self.grad = theano.function([self.inputs], T.grad(self._cost, self.params))
        self.prehiddens = theano.function([self.inputs], self._prehiddens)
        self.hiddens = theano.function([self.inputs], self._hiddens)
        #self.recons_from_prehiddens = theano.function([self._prehiddens], self._outputs)
        self.recons_from_inputs = theano.function([self.inputs], self._outputs)

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

