import numpy 
import numpy.random
import pylab
from dispims_color import dispims_color
import zaethroughtime
import train
import theano
from theano.tensor.shared_randomstreams import RandomStreams

rng = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
SMALL = 0.001
patchsize = 16 
numfeatures = 225 


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




#GET SOME CIFAR IMAGES 
trainimages = (numpy.concatenate([(numpy.load(CIFARDATADIR+'/data_batch_'+b)['data']) for b in ["1"]], 0).reshape(-1,3,32,32)/255.).astype("float32")[:1000]

#CROP PATCHES
print "cropping patches"
trainpatches = numpy.concatenate([crop_patches_color(im.reshape(3, 32, 32).transpose(1,2,0), numpy.array([numpy.random.randint(patchsize/2, 32-patchsize/2, 200), numpy.random.randint(patchsize/2, 32-patchsize/2, 200)]).T, patchsize) for im in trainimages])
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
pca_backward, pca_forward, zca_backward, zca_forward = pca(trainpatches, 0.95, whiten=True)
trainpatches_whitened = numpy.dot(trainpatches, pca_backward.T).astype("float32")
trainpatches_theano = theano.shared(trainpatches_whitened)
print "done"

#INSTANTIATE THE ZERO-BIAS AUTOENCODER
#model = zae.Zae(numvis=trainpatches_whitened.shape[1], numhid=numfeatures, vistype="real", init_features=0.1*trainpatches_whitened[:numfeatures].T, selectionthreshold=1.0)
model = zaethroughtime.Zae(rng,numvis=trainpatches_whitened.shape[1], numhid=numfeatures, vistype="real", numsteps=8, init_features=0.1*trainpatches_whitened[:numfeatures].T, selectionthreshold=1.0)



##DO SOME STEPS WITH SMALL LEARNING RATE TO MAKE SURE THE INITIALIZATION IS IN A REASONABLE RANGE
#trainer = train.GraddescentMinibatch(model, trainpatches_theano, 100, learningrate=0.0001, momentum=0.9)
##trainer = graddescent_rewrite.SGD_Trainer(model, trainpatches_whitened, batchsize=128, learningrate=0.001, momentum=0.9, loadsize=30000, gradient_clip_threshold=5.0)
#trainer.step(); trainer.step(); trainer.step() 


#TRAIN THE MODEL FOR REAL, AND SHOW FILTERS 
import graddescent_rewrite
#trainer = train.GraddescentMinibatch(model, trainpatches_theano, 100, learningrate=0.01, momentum=0.9)
trainer = graddescent_rewrite.SGD_Trainer(model, trainpatches_whitened, batchsize=128, learningrate=0.1, momentum=0.9, loadsize=30000, gradient_clip_threshold=5.0)                                          

assert False, "preprocessing is done, may train now"

for epoch in xrange(100):
    trainer.step()
    if epoch % 10 == 0 and epoch > 0:
        #trainer.set_learningrate(trainer.learningrate*0.8)
        dispims_color(numpy.dot(model.W.get_value().T, pca_forward.T).reshape(-1, patchsize, patchsize, 3), 1)
        pylab.draw(); pylab.show()



#SHOW SOME EXAMPLES 
subplot(1,2,2); dispims_color(numpy.dot(model.recons_from_prehiddens(trainpatches_whitened[:25], 0.5*randn(25,model.numhid).astype("float32")), pca_forward.T).reshape(-1, patchsize, patchsize, 3), 1)
#AND SOME SAMPLED RECONSTRUCTIONS
subplot(1,2,2); dispims_color(numpy.dot(model.recons_from_prehiddens(trainpatches_whitened[:25], 0.5*randn(25,model.numhid).astype("float32")), pca_forward.T).reshape(-1, patchsize, patchsize, 3), 1)
#OR SOME RANDOM SAMPLES 
#subplot(1,2,2); dispims_color(numpy.dot(model.recons_from_prehiddens(1.0*randn(100,92).astype("float32"), 0.24*randn(100,model.numhid).astype("float32")), pca_forward.T).reshape(-1, patchsize, patchsize, 3), 1)




