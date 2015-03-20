import matplotlib
matplotlib.use('Agg')

import numpy 
import numpy.random
import pylab
from dispims_color import dispims_color
import zae
import train
import theano
import theano.tensor as T
#import graddescent_rewrite

rng = numpy.random.RandomState(1)
numfeatures = 4096

datafile = "/data/tinyimages/FirstMioTiny.npy"


def pca_nowhite(data, dimstokeep):
    """ principal components analysis of data (columnwise in array data), retaining as many components as required to retain var_fraction of the variance 
    """
    from numpy.linalg import eigh
    u, v = eigh(numpy.cov(data, rowvar=0, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    backward_mapping = v[:,:dimstokeep].T
    forward_mapping = v[:,:dimstokeep]
    return backward_mapping.astype("float32"), forward_mapping.astype("float32"), numpy.dot(v[:,:dimstokeep].astype("float32"), backward_mapping), numpy.dot(forward_mapping, v[:,:dimstokeep].T.astype("float32"))


print "loading data"
trainimages = numpy.load(datafile ).reshape(-1,3,32,32).transpose(0,3,2,1).reshape(-1,32*32*3)
trainimages = trainimages[rng.permutation(trainimages.shape[0])].astype("float32")/255.
meanstd = trainimages.std()
trainimages -= trainimages.mean(1)[:,None]
trainimages /= trainimages.std(1)[:,None] + meanstd
trainimages_mean = trainimages.mean(0)[None,:]
trainimages_std = trainimages.std(0)[None,:] 
meanstd0 = trainimages_std.mean()
trainimages -= trainimages_mean
trainimages /= trainimages_std + meanstd0
print "done"


print "doing pca"
#pca_backward, pca_forward, zca_backward, zca_forward = pca_nowhite(trainimages, 0.9999999)
pca_backward, pca_forward, zca_backward, zca_forward = pca_nowhite(trainimages, 2000)
print "done"
print "dimensions retained:", pca_backward.shape[0]
trainimages = numpy.dot(trainimages, pca_backward.T)


print "instantiating model" 
model = zae.Zae(numvis=trainimages.shape[1], numhid=numfeatures, vistype="real", init_features=0.01*trainimages[:numfeatures].T, selectionthreshold=1.0) #, normpenalty=0.01)
print "done"
print "instantiating trainer"
#trainer = graddescent_rewrite.SGD_Trainer(model=model, inputs=trainimages, batchsize=128, learningrate=0.01, gradient_clip_threshold=5.0, loadsize=300000)
trainer = train.GraddescentMinibatch(model, trainpatches_theano, 100, learningrate=0.01, momentum=0.9)
print "done"


for epoch in xrange(500):
    trainer.step()
    if True: #$epoch % 10 == 0:
        pylab.clf()
        dispims_color(numpy.dot(model.W.get_value().T, pca_forward.T).reshape(-1, 32, 32, 3), 1)
        pylab.draw(); pylab.show()
    if epoch == 400:
        trainer.set_learningrate(trainer.learningrate*0.5)
    if epoch == 450:
        trainer.set_learningrate(trainer.learningrate*0.5)

#pylab.savefig("2015_03_03_numfeatures"+str(numfeatures)+"threshold"+str(threshold)+"_trainoncifar.png")
#model.save("2015_03_03_numfeatures"+str(numfeatures)+"threshold"+str(threshold)+"_trainoncifar")


