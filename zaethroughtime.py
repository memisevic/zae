import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Zae(object):
    def __init__(self, rng, numvis, numhid, vistype, numsteps, init_features, selectionthreshold=1.0):
        self.numvis = numvis
        self.numhid  = numhid
        self.vistype = vistype
        self.numsteps = numsteps
        self.rng = rng
        self.theano_rng = RandomStreams(1)
        self.selectionthreshold = theano.shared(value=selectionthreshold, name='selectionthreshold')
        self.corruption_level = 0.5

        self.Wz = theano.shared(value=init_features.astype(theano.config.floatX), name='Wz')
        #self.Wz = theano.shared(value=1.0*self.rng.randn(self.numvis, self.numhid).astype(theano.config.floatX), name='Wz')
        self.Wlin = theano.shared(value=0.01*self.rng.randn(self.numhid, self.numvis).astype(theano.config.floatX), name='Wlin')
        self.Wfeedbackdecenc = theano.shared(value=0.01*self.rng.randn(self.numhid, self.numhid).astype(theano.config.floatX), name='Wfeedbackdecenc')
        self.Wfeedbackcanenc = theano.shared(value=0.01*self.rng.randn(self.numvis, self.numhid).astype(theano.config.floatX), name='Wfeedbackcanenc')
        self.Wfeedbackdecdec = theano.shared(value=0.01*self.rng.randn(self.numhid, self.numhid).astype(theano.config.floatX), name='Wfeedbackdecdec')
        self.Wfeedbackencenc = theano.shared(value=0.01*self.rng.randn(self.numhid, self.numhid).astype(theano.config.floatX), name='Wfeedbackencenc')
        self.bvis = theano.shared(value=numpy.zeros(numvis, dtype=theano.config.floatX), name='bvis')
        self.bhidZ = theano.shared(value=numpy.zeros(numhid, dtype=theano.config.floatX), name='bhidZ')
        self.bhidZrecons = theano.shared(value=numpy.zeros(numhid, dtype=theano.config.floatX), name='bhidZrecons')
        self.inputs = T.matrix(name = 'inputs') 
        self.params = [self.Wz, self.Wlin, self.Wfeedbackdecenc, self.Wfeedbackcanenc, self.Wfeedbackdecdec, self.Wfeedbackencenc, self.bvis, self.bhidZrecons, self.bhidZ]

        self._canvas    = T.zeros(self.inputs.shape)
        self._preZ      = [T.zeros((self.inputs.shape[0], self.numhid))] + [None] * (self.numsteps-1)
        self._Z         = [T.zeros((self.inputs.shape[0], self.numhid))] + [None] * (self.numsteps-1)
        self._preZrecons= [T.zeros((self.inputs.shape[0], self.numhid))] + [None] * (self.numsteps-1)
        self._Zrecons   = [T.zeros((self.inputs.shape[0], self.numhid))] + [None] * (self.numsteps-1)
        self._L         = [T.zeros((self.inputs.shape[0], self.numvis))] + [None] * (self.numsteps-1)
        for t in range (1, numsteps):
            #self._corruptedinputs = self.theano_rng.binomial(size=self.inputs.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputs
            self._corruptedinputs = self.inputs
            #self._corruptedinputs = self.theano_rng.normal(size=self.inputs.shape, avg=0.0, std=1.0, dtype=theano.config.floatX) + self.inputs
            self._preZ[t] = T.dot(self._corruptedinputs, self.Wz) + T.dot(self._preZrecons[t-1], self.Wfeedbackdecenc) + T.dot(self._preZ[t-1], self.Wfeedbackencenc) + T.dot(self._canvas[t-1]-self.inputs, self.Wfeedbackcanenc) + self.bhidZ
            self._Z[t] = (self._preZ[t] > self.selectionthreshold) * self._preZ[t]
            #self._Z[t] = T.nnet.sigmoid(self._preZ[t])
            self._L[t] = T.dot(self._Z[t], self.Wlin) 
            self._preZrecons[t] = T.dot(self._L[t], self.Wlin.T) + T.dot(self._preZrecons[t-1], self.Wfeedbackdecdec) + self.bhidZrecons
            #self._Zrecons[t] = (self._preZrecons[t] > self.selectionthreshold) * self._preZrecons[t]
            self._Zrecons[t] = (self._preZrecons[t] > 0.0) * self._preZrecons[t]
            #self._Zrecons[t] = T.nnet.sigmoid(self._preZrecons[t])
            self._canvas += T.dot(self._Zrecons[t], self.Wz.T) 
        self._canvas += self.bvis 

        if self.vistype == 'binary':
            self._canvas = T.nnet.sigmoid(self._canvas)
            costpercase = -T.sum(self.inputs*T.log(self._canvas) + (1-self.inputs)*T.log(1-self._canvas), axis=1) 
        elif self.vistype == 'real':
            costpercase = T.sum(0.5 * ((self.inputs - self._canvas)**2), axis=1) 

        self._cost = T.mean(costpercase)
        self._grads = T.grad(self._cost, self.params)

        self.cost = theano.function([self.inputs], self._cost)
        self.grad = theano.function([self.inputs], T.grad(self._cost, self.params))
        #self.prehiddens = theano.function([self.inputs], self._prehiddens)
        #self.hiddens = theano.function([self.inputs], self._hiddens)
        #self.recons_from_prehiddens = theano.function([self.inputs, self._prehiddens], self._canvas)
        self.recons_from_inputs = theano.function([self.inputs], self._canvas)
        self.selection = theano.function([self.inputs], (self._Z[-1] > self.selectionthreshold))

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

