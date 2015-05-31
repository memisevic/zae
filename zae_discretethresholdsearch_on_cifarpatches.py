import numpy 
import numpy.random
import pylab
from dispims_color import dispims_color
import zae 
#import switchingzae as zae
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

rng = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
SMALL = 0.001
patchsize = 20
numfeatures = 400


import os
HOME = os.environ['HOME']
CIFARDATADIR = HOME+'/research/data/cifar/cifar-10-batches-py'


def crop_patches_color(image, keypoints, patchsize):
    patches = numpy.zeros((len(keypoints), 3*patchsize**2))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[k[0]-patchsize/2:k[0]+patchsize/2, k[1]-patchsize/2:k[1]+patchsize/2,:].flatten()
    return patches


def pca(data, var_fraction, whiten=True):
    """ principal components analysis of data (columnwise in array data), retaining as many components as required to retain var_fraction of the variance 
    """
    from numpy.linalg import eigh
    u, v = eigh(numpy.cov(data, rowvar=0, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<=(u.sum()*var_fraction)]
    numprincomps = u.shape[0]
    u[u<SMALL] = SMALL
    if whiten: 
        backward_mapping = ((u**(-0.5))[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]).T
        forward_mapping = (u**0.5)[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]
    else: 
        backward_mapping = v[:,:numprincomps].T
        forward_mapping = v[:,:numprincomps]
    return backward_mapping, forward_mapping, numpy.dot(v[:,:numprincomps], backward_mapping), numpy.dot(forward_mapping, v[:,:numprincomps].T)


class GraddescentMinibatchDiscreteThresholdSearch(object):

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data_numpy    = data
        self.data_theano   = theano.shared(self.data_numpy)
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data_theano.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.inputs:self.data_theano[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        def inplaceclip(x):
            x[:,:] *= x>0.0
            return x

        def inplacemask(x, mask):
            x[:,:] *= mask
            return x

        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)
            for i in range(10):
                oldcost = model.cost(self.data_numpy[batch_index*self.batchsize:(batch_index+1)*self.batchsize])
                oldselectionthreshold = model.selectionthreshold.get_value()
                newselectionthreshold = oldselectionthreshold + self.rng.randn(*oldselectionthreshold.shape).astype("float32")*0.0001
                newselectionthreshold *= newselectionthreshold > 0.0
                self.model.selectionthreshold.set_value(newselectionthreshold)
                newcost = model.cost(self.data_numpy[batch_index*self.batchsize:(batch_index+1)*self.batchsize])
                if newcost > oldcost:
                    self.model.selectionthreshold.set_value(oldselectionthreshold)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)



#GET SOME CIFAR IMAGES 
trainimages = (numpy.concatenate([(numpy.load(CIFARDATADIR+'/data_batch_'+b)['data']) for b in ["1"]], 0).reshape(-1,3,32,32)/255.).astype("float32")[:1000]

#CROP PATCHES
print "cropping patches"
trainpatches = numpy.concatenate([crop_patches_color(im.reshape(3, 32, 32).transpose(1,2,0), numpy.array([numpy.random.randint(patchsize/2, 32-patchsize/2, 40), numpy.random.randint(patchsize/2, 32-patchsize/2, 40)]).T, patchsize) for im in trainimages])
R = rng.permutation(trainpatches.shape[0])
trainpatches = trainpatches[R, :]
print "numpatches: ", trainpatches.shape[0]
print "done"

#LEARN WHITENING MATRICES 
print "whitening"
meanstd = trainpatches.std()
trainpatches -= trainpatches.mean(1)[:,None]
trainpatches /= trainpatches.std(1)[:,None] + 0.1 * meanstd
trainpatches_mean = trainpatches.mean(0)[None,:]
trainpatches_std = trainpatches.std(0)[None,:] 
trainpatches -= trainpatches_mean
trainpatches /= trainpatches_std + 0.1 * meanstd
pca_backward, pca_forward, zca_backward, zca_forward = pca(trainpatches, 0.9, whiten=True)
trainpatches_whitened = numpy.dot(trainpatches, pca_backward.T).astype("float32")
print "done"

#INSTANTIATE THE ZERO-BIAS AUTOENCODER
model = zae.Zae(numvis=trainpatches_whitened.shape[1], numhid=numfeatures, vistype="real", init_features=0.1*trainpatches_whitened[:numfeatures].T, selectionthreshold=1.0*numpy.ones(numfeatures, dtype="float32"))

assert False, "preprocessing is done, may train now"


#DO SOME STEPS WITH SMALL LEARNING RATE TO MAKE SURE THE INITIALIZATION IS IN A REASONABLE RANGE
trainer = GraddescentMinibatchDiscreteThresholdSearch(model, trainpatches_whitened, 100, learningrate=0.0001, momentum=0.9)
trainer.step(); trainer.step(); trainer.step() 

#TRAIN THE MODEL FOR REAL, AND SHOW FILTERS 
trainer = GraddescentMinibatchDiscreteThresholdSearch(model, trainpatches_whitened, 100, learningrate=0.01, momentum=0.9)
#trainer = graddescent_rewrite.SGD_Trainer(model, trainpatches_whitened, batchsize=128, learningrate=0.1, momentum=0.9, loadsize=30000, gradient_clip_threshold=5.0)                                          


for epoch in xrange(100):
    trainer.step()
    if epoch % 10 == 0 and epoch > 0:
        #trainer.set_learningrate(trainer.learningrate*0.8)
        dispims_color(numpy.dot(model.W.get_value().T, pca_forward.T).reshape(-1, patchsize, patchsize, 3), 1)
        pylab.draw(); pylab.show()



