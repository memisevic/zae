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
import logreg 
import LogisticRegression_theano 
from theano.tensor.shared_randomstreams import RandomStreams
import graddescent_rewrite

rng = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
SMALL = 0.001
numfeatures = 4096

def pca_nowhite(data, dimstokeep):
    """ principal components analysis of data (columnwise in array data), retaining as many components as required to retain var_fraction of the variance 
    """
    from numpy.linalg import eigh
    u, v = eigh(numpy.cov(data, rowvar=0, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    backward_mapping = v[:,:dimstokeep].T
    forward_mapping = v[:,:dimstokeep]
    return backward_mapping.astype("float32"), forward_mapping.astype("float32"), numpy.dot(v[:,:dimstokeep].astype("float32"), backward_mapping), numpy.dot(forward_mapping, v[:,:dimstokeep].T.astype("float32"))

pretrainimages = numpy.load("/data/tinyimages/FirstMioTiny.npy").reshape(-1,3,32,32).transpose(0,3,2,1).reshape(-1,32*32*3)
pretrainimages = pretrainimages[rng.permutation(pretrainimages.shape[0])].astype("float32")/255.
print "done"


meanstd1 = alltrainimages.std()
alltrainimages -= alltrainimages.mean(1)[:,None]
alltrainimages /= alltrainimages.std(1)[:,None] + meanstd1
pretrainimages -= pretrainimages.mean(1)[:,None]
pretrainimages /= pretrainimages.std(1)[:,None] + meanstd1
alltrainimages_mean = alltrainimages.mean(0)[None,:]
alltrainimages_std = alltrainimages.std(0)[None,:] 
meanstd0 = alltrainimages_std.mean()
alltrainimages -= alltrainimages_mean
alltrainimages /= alltrainimages_std + meanstd0
pretrainimages -= alltrainimages_mean
pretrainimages /= alltrainimages_std + meanstd0
print "doing pca"
#pca_backward, pca_forward, zca_backward, zca_forward = pca_nowhite(alltrainimages, 0.9999999)
pca_backward, pca_forward, zca_backward, zca_forward = pca_nowhite(alltrainimages, 2000)
print "done"
print "dimensions retained:", pca_backward.shape[0]
alltrainimages = numpy.dot(alltrainimages, pca_backward.T)
pretrainimages = numpy.dot(pretrainimages, pca_backward.T)


print "instantiating model" 
model = zae.Zae(numvis=alltrainimages.shape[1], numhid=numfeatures, vistype="real", init_features=0.01*alltrainimages[:numfeatures].T, selectionthreshold=1.0) #, normpenalty=0.01)
print "done"
print "instantiating trainer"
#trainer = graddescent_rewrite.SGD_Trainer(model=model, inputs=pretrainimages, batchsize=128, learningrate=0.01, gradient_clip_threshold=5.0, loadsize=300000)
trainer = train.GraddescentMinibatch(model, trainpatches_theano, 100, learningrate=0.01, momentum=0.9)
print "done"


if TRAINMODEL:
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

    pylab.savefig("2015_03_03_numfeatures"+str(numfeatures)+"threshold"+str(threshold)+"_trainoncifar.png")
    model.save("2015_03_03_numfeatures"+str(numfeatures)+"threshold"+str(threshold)+"_trainoncifar")
else:
    model.load("2015_03_03_numfeatures"+str(numfeatures)+"threshold"+str(threshold)+"_trainoncifar.npy")




