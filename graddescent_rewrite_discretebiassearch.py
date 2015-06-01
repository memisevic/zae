import collections
import cPickle as pickle
import os
import shutil
import warnings

import numpy as np
import theano
import theano.tensor as T
import tables
#theano.config.compute_test_value = 'warn'


class SGD_Trainer(object):
    """Implementation of a stochastic gradient descent trainer
    """

#{{{ Properties

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, val):
        #FIXME: make this work for other input types
        if not isinstance(val, np.ndarray):
            raise TypeError('Resetting trainer inputs currently only works for '
                        'ndarray inputs!')
        self._inputs = val
        self._inputs_theano = theano.shared(
            self._inputs[:self._loadsize],
            name='inputs')
        self._numcases = self._inputs.shape[0]
        self._numloads = self._numcases // self._loadsize
        print 'recompiling trainer functions...'
        self._compile_functions()


    @property
    def gradient_clip_threshold(self):
        return self._gradient_clip_threshold.get_value()

    @property
    def learningrate_decay_factor(self):
        return self._learningrate_decay_factor.get_value()

    @learningrate_decay_factor.setter
    def learningrate_decay_factor(self, val):
        self._learningrate_decay_factor.set_value(np.float32(val))

    @property
    def learningrate_decay_interval(self):
        return self._learningrate_decay_interval.get_value()

    @learningrate_decay_interval.setter
    def learningrate_decay_interval(self, val):
        self._learningrate_decay_interval.set_value(np.int64(val))

    @gradient_clip_threshold.setter
    def gradient_clip_threshold(self, val):
        self._gradient_clip_threshold.set_value(np.float32(val))

    @property
    def learningrate(self):
        return self._learningrate.get_value()

    @learningrate.setter
    def learningrate(self, value):
        self._learningrate.set_value(np.float32(value))

    @property
    def momentum(self):
        return self._momentum.get_value()

    @momentum.setter
    def momentum(self, val):
        self._momentum.set_value(np.float32(val))

    @property
    def batchsize(self):
        return self._batchsize

    @property
    def loadsize(self):
        return self._loadsize

    @property
    def numcases(self):
        return self._numcases

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        self._verbose = bool(val)

    @property
    def epochcount(self):
        return self._epochcount

    @epochcount.setter
    def epochcount(self, val):
        self._epochcount = int(val)

    @property
    def momentum_batchcounter(self):
        return self._momentum_batchcounter

    @property
    def nparams(self):
        return self._nparams
#}}}

    def __init__(self, model=None, inputs=None, batchsize=100, learningrate=.01,
                 momentum=0.9, loadsize=None,
                 rng=None, verbose=True,
                 numcases=None, gradient_clip_threshold=1000,
                 numepochs_per_load=1,
                 rmsprop=False, cost=None, params=None, inputvar=None,
                 grads=None, monitor_update_weight_norm_ratio=False,
                 auto_reset_on_naninf=True):

    # loadsize = ??
    # numcases = ??

        self.mutationrate = 0.0001

#{{{ Initialization of Properties
        assert model is not None or (
            cost is not None and params is not None and
            inputvar is not None and grads is not None), (
                "either a model instance or cost, params and inputvar "
                "have to be passed to the SGD_Trainer constructor")

        self.auto_reset_on_naninf = auto_reset_on_naninf
        self.monitor_update_weight_norm_ratio = monitor_update_weight_norm_ratio
        print 'monitor_update_weight_norm_ratio: {0}'.format(monitor_update_weight_norm_ratio, )

        if model is not None:
            self._model = model
            self._params = model.params
            self._cost = model._cost
            self._inputvar = model.inputs
            self._grads = model._grads
        else:
            self._params = params
            self._cost = cost
            self._inputvar = inputvar
            self._grads = grads

        # compute total number of params
        self._nparams = 0
        for p in self._params:
            try:
                self._nparams += p.get_value().size
            except AttributeError:
                # handles scalar params
                self._nparams += 1
        print 'number of params: {0}'.format(self._nparams)

        if monitor_update_weight_norm_ratio:
            self._update_weight_norm_ratios_log = dict(
                [(p, []) for p in self._params])

        self._learningrate = theano.shared(np.float32(learningrate),
                                           name='learningrate')
        self.numepochs_per_load = numepochs_per_load

        self._momentum = theano.shared(np.float32(momentum),
                                       name='momentum')
        self._total_stepcount = 0

        self._gradient_clip_threshold = theano.shared(
                np.float32(gradient_clip_threshold),
                name='gradient_clip_threshold')
        self._avg_gradnorm = theano.shared(np.float32(0.), name='avg_gradnorm')

        self._learningrate_decay_factor = theano.shared(
            np.float32,
            name='learningrate_decay_factor')

        self._learningrate_decay_interval = theano.shared(
            np.int64,
            name='learningrate_decay_interval')

        if isinstance(inputs, str):
            self._inputs_type = 'h5'
            self._inputsfile = tables.openFile(inputs, 'r')
            self._inputs = self._inputsfile.root.inputs_white
        elif hasattr(inputs, '__call__'):
            self._inputs_type = 'function'
            self._inputs_fn = inputs
        else:
            self._inputs_type = 'numpy'
            self._inputs = inputs

        self._model = model

        self._numparams = reduce(lambda x,y: x+y,
            [p.get_value().size for p in self._params])

        if self._inputs_type == 'function':
            numcases = loadsize
        else:
            if numcases is None or numcases > self._inputs.shape[0]:
                numcases = self._inputs.shape[0]
        self._numcases = numcases

        self._batchsize = batchsize
        self._loadsize = loadsize
        self._verbose       = verbose
        if self._batchsize > self._numcases:
            self._batchsize = self._numcases
        if self._loadsize == None:
            self._loadsize = self._batchsize * 100
        if self._loadsize > self._numcases:
            self._loadsize = self._numcases
        self._numloads      = self._numcases // self._loadsize
        self._numbatches    = self._loadsize // self._batchsize

        if self._inputs_type == 'h5':
            self._inputs_theano = theano.shared(
                self._inputs.read(stop=self._loadsize))
        elif self._inputs_type == 'function':
            # TODO: generate inputs for first load
            print "generating first load..."
            inp = np.empty((self._loadsize, ) + (self._inputs_fn().shape),
                           dtype=np.float32)
            for i in xrange(self._loadsize):
                inp[i] = self._inputs_fn()
                if (i + 1) % 100 == 0:
                    print '{0}/{1}'.format(i + 1, self.loadsize)

            self._inputs_theano = theano.shared(
                inp)
        else:
            self._inputs_theano = theano.shared(
                self._inputs[:self._loadsize],
                name='inputs')
        #self._inputs_theano.tag.test_value = np.random.randn(100, model.n_vis*4)

        self._momentum_batchcounter = 0

        if rng is None:
            self._rng = np.random.RandomState(1)
        else:
            self._rng = rng

        self._epochcount = 0
        self._index = T.lscalar()
        self._incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='inc_'+p.name))
                            for p in self._params])
        self._inc_updates = collections.OrderedDict()
        self.rmsprop = rmsprop
        if self.rmsprop:
            self.averaging_coeff=0.95
            self.stabilizer=1e-2
            self._avg_grad_sqrs = \
              dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                                dtype=theano.config.floatX), name='avg_grad_sqr_'+p.name))
                                for p in self._params])
        self._avg_grad_sqrs_updates = collections.OrderedDict()
        self._updates_nomomentum = collections.OrderedDict()
        self._updates = collections.OrderedDict()
        self._n = T.lscalar('n')
        self._n.tag.test_value = 0.
        self._noop = 0.0 * self._n
        self._batch_idx = theano.shared(
            value=np.array(0, dtype=np.int64), name='batch_idx')

        self.costs = []
        self._compile_functions()

#}}}

    def __del__(self):
        if self._inputs_type == 'h5':
            self._inputsfile.close()

    def save(self, filename):
        """Saves the trainers parameters to a file
        Params:
            filename: path to the file
        """
        ext = os.path.splitext(filename)[1]
        if ext == '.pkl':
            print 'saving trainer params to a pkl file'
            self.save_pkl(filename)
        else:
            print 'saving trainer params to a hdf5 file'
            self.save_h5(filename)

    def save_h5(self, filename):
        """Saves a HDF5 file containing the trainers parameters
        Params:
            filename: path to the file
        """
        try:
            shutil.copyfile(filename, '{0}_bak'.format(filename))
        except IOError:
            print 'could not make backup of trainer param file (which is \
                    normal if we haven\'t saved one until now)'
        paramfile = tables.openFile(filename, 'w')
        paramfile.createArray(paramfile.root, 'learningrate',
                              self.learningrate)
        paramfile.createArray(paramfile.root, 'verbose', self.verbose)
        paramfile.createArray(paramfile.root, 'loadsize', self.loadsize)
        paramfile.createArray(paramfile.root, 'batchsize', self.batchsize)
        paramfile.createArray(paramfile.root, 'momentum',
                              self.momentum)
        paramfile.createArray(paramfile.root, 'epochcount',
                              self.epochcount)
        paramfile.createArray(paramfile.root, 'momentum_batchcounter',
                              self.momentum_batchcounter)
        incsgrp = paramfile.createGroup(paramfile.root, 'incs', 'increments')
        for p in self._params:
            paramfile.createArray(incsgrp, p.name, self._incs[p].get_value())
        if self.rmsprop:
            avg_grad_sqrs_grp = paramfile.createGroup(paramfile.root, 'avg_grad_sqrs')
            for p in self._params:
                paramfile.createArray(avg_grad_sqrs_grp, p.name, self._avg_grad_sqrs[p].get_value())
        paramfile.close()

    def save_pkl(self, filename):
        """Saves a pickled dictionary containing the parameters to a file
        Params:
            filename: path to the file
        """
        param_dict = {}
        param_dict['learningrate'] = self.learningrate
        param_dict['verbose'] = self.verbose
        param_dict['loadsize'] = self.loadsize
        param_dict['batchsize'] = self.batchsize
        param_dict['momentum'] = self.momentum
        param_dict['epochcount'] = self.epochcount
        param_dict['momentum_batchcounter'] = self.momentum_batchcounter
        param_dict['incs'] = dict(
            [(p.name, self._incs[p].get_value()) for p in self._params])
        if self.rmsprop:
            param_dict['avg_grad_sqrs'] = dict(
                [(p.name, self._avg_grad_sqrs[p].get_value()) for p in self._params])
        pickle.dump(param_dict, open(filename, 'wb'))

    def load(self, filename):
        """Loads pickled dictionary containing parameters from a file
        Params:
            filename: path to the file
        """
        param_dict = pickle.load(open('%s' % filename, 'rb'))
        self.learningrate = param_dict['learningrate']
        self.verbose = param_dict['verbose']
        self._loadsize = param_dict['loadsize']
        self._batchsize = param_dict['batchsize']
        self.momentum = param_dict['momentum']
        self.epochcount = param_dict['epochcount']
        self._momentum_batchcounter = param_dict['momentum_batchcounter']
        for param_name in param_dict['incs'].keys():
            for p in self._params:
                if p.name == param_name:
                    self._incs[p].set_value(param_dict['incs'][param_name])
        if self.rmsprop:
            for param_name in param_dict['avg_grad_sqrs'].keys():
                for p in self._params:
                    if p.name == param_name:
                        self._avg_grad_sqrs[p].set_value(param_dict['avg_grad_sqrs'][param_name])
        self._numbatches = self._loadsize // self._batchsize
        if self._inputs_type != 'function':
            self._numloads = self._inputs.shape[0] // self._loadsize
        if self._inputs_type == 'h5':
            self._inputs_theano.set_value(
                self._inputs.read(stop=self._loadsize))
        else:
            self._inputs_theano.set_value(self._inputs[:self._loadsize])

    def reset_incs(self):
        for p in self._params:
            self._incs[p].set_value(
                np.zeros(p.get_value().shape, dtype=theano.config.floatX))

    def reset_avg_grad_sqrs(self):
        if self.rmsprop:
            for p in self._params:
                self._avg_grad_sqrs[p].set_value(
                    np.zeros(p.get_value().shape, dtype=theano.config.floatX))

    def _compile_functions(self):
        self._gradnorm = T.zeros([])
        for _param, _grad in zip(self._params, self._grads):
            # apply rmsprop to before clipping gradients
            if self.rmsprop:
                avg_grad_sqr = self._avg_grad_sqrs[_param]
                new_avg_grad_sqr =  self.averaging_coeff * avg_grad_sqr + \
                    (1 - self.averaging_coeff) * T.sqr(_grad)
                self._avg_grad_sqrs_updates[avg_grad_sqr] = new_avg_grad_sqr
                rms_grad_t = T.sqrt(new_avg_grad_sqr)
                rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
                _grad = _grad / rms_grad_t
            self._gradnorm += T.sum(_grad**2) # calculated on the rmsprop 'grad'
        self._gradnorm = T.sqrt(self._gradnorm)
        self.gradnorm = theano.function(
            inputs=[],
            outputs=self._gradnorm,
            givens={
                self._inputvar:
                self._inputs_theano[
                    self._batch_idx*self.batchsize:
                    (self._batch_idx+1)*self.batchsize]})

        avg_gradnorm_update = {
            self._avg_gradnorm: self._avg_gradnorm * .8 + self._gradnorm * .2}

        self._update_weight_norm_ratios = []
        for _param, _grad in zip(self._params, self._grads):
            if hasattr(self._model, 'skip_params'):
                if _param.name in self._model.skip_params:
                    continue

            _clip_grad = T.switch(
                T.gt(self._gradnorm, self._gradient_clip_threshold),
                _grad * self._gradient_clip_threshold / self._gradnorm, _grad)

            try: # ... to apply learningrate_modifiers
                # Cliphid version:
                self._inc_updates[self._incs[_param]] = \
                        self._momentum * self._incs[_param] - \
                        self._learningrate * \
                        self._model.layer.learningrate_modifiers[
                            _param.name] * _clip_grad

                self._updates[_param] = _param + self._incs[_param]
                self._updates_nomomentum[_param] = _param - \
                    self._learningrate * \
                    self._model.layer.learningrate_modifiers[_param.name] * \
                        _clip_grad

            except AttributeError:
                self._inc_updates[self._incs[_param]] = self._momentum * \
                        self._incs[_param] - self._learningrate * _clip_grad
                self._updates[_param] = _param + self._incs[_param]
                self._updates_nomomentum[_param] = _param - \
                        self._learningrate * _clip_grad

            if self.monitor_update_weight_norm_ratio:
                print 'building update weight norm ratio graph for ', _param.name
                self._update_weight_norm_ratios.append(
                    T.mean(self._incs[_param]**2) / T.mean(
                        _param**2))


        # compute function to get update_weight_norm_ratios (returned in same
        # order as params list)
        print 'compiling update weight norm ratio function'
        self.get_update_weight_norm_ratios = theano.function(
            [], self._update_weight_norm_ratios)
        print 'done'

        # first update gradient norm running avg
        ordered_updates = collections.OrderedDict(avg_gradnorm_update)
        # so that it is considered in the parameter update computations
        ordered_updates.update(self._inc_updates)
        self._updateincs = theano.function(
            [], [self._cost, self._avg_gradnorm], updates = ordered_updates,
            givens = {self._inputvar:self._inputs_theano[
                self._batch_idx*self._batchsize:(self._batch_idx+1)* \
                self._batchsize]})

        self._trainmodel = theano.function(
            [self._n], self._noop, updates = self._updates)

        self._trainmodel_nomomentum = theano.function(
            [self._n], self._noop, updates = self._updates_nomomentum,
            givens = {self._inputvar:self._inputs_theano[
                self._batch_idx*self._batchsize:(self._batch_idx+1)* \
                self._batchsize]})

        self._momentum_batchcounter = 0


    def _trainsubstep(self, batchidx):
        self._batch_idx.set_value(batchidx)
        stepcost, avg_gradnorm = self._updateincs()
        # catch NaN, before updating params
        try:
            if np.isnan(stepcost):
                raise ValueError, 'Cost function returned nan!'
            elif np.isinf(stepcost):
                raise ValueError, 'Cost function returned infinity!'
        except ValueError:
            if self.auto_reset_on_naninf:
                print 'nan or inf detected, resetting...'
                self.reset_incs()
                self.reset_avg_grad_sqrs()
                self._avg_gradnorm.set_value(0.0)
            else:
                print ('nan or inf detected, auto_reset_on_naninf is set to '
                       'False. Set it to True to automagically reset the '
                       'trainer and continue training.')
            raise

        if self._momentum_batchcounter < 10:
            self._momentum_batchcounter += 1
            self._trainmodel_nomomentum(0)
        else:
            self._momentum_batchcounter = 10
            self._trainmodel(0)
        return stepcost, avg_gradnorm

    def get_avg_gradnorm(self):
        avg_gradnorm = 0.0
        print self.gradnorm()
        for batch_idx in range(self._numbatches):
            self._batch_idx.set_value(batch_idx)
            tmp = self.gradnorm()
            avg_gradnorm += tmp / self._numbatches
        print avg_gradnorm
        return avg_gradnorm

    def step(self):
        total_cost = 0.0
        cost = 0.0
        stepcount = 0.0

        self._epochcount += 1

        for load_index in range(self._numloads):
            indices = np.random.permutation(self._loadsize)
            if self._inputs_type == 'h5':
                self._inputs_theano.set_value(
                    self._inputs.read(
                        start=load_index * self._loadsize,
                        stop=(load_index + 1) * self._loadsize)[indices])
            elif self._inputs_type == 'function':
                # if load has been used n times, gen new load
                if self._epochcount % self.numepochs_per_load == 0:
                    print 'using data function to generate new load...'
                    inp = np.empty((self._loadsize, ) + (self._inputs_fn().shape),
                                dtype=np.float32)
                    for i in xrange(self._loadsize):
                        inp[i] = self._inputs_fn()
                        if (i + 1) % 100 == 0:
                            print '{0}/{1}'.format(i + 1, self.loadsize)
                    self._inputs_theano.set_value(inp)
                    print 'done'
            else:
                self._inputs_theano.set_value(
                    self._inputs[load_index * self._loadsize + indices])
            for batch_index in self._rng.permutation(self._numbatches):
                stepcount += 1.0
                self._total_stepcount += 1.0
                stepcost, avg_gradnorm = self._trainsubstep(batch_index)
                cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)* \
                    stepcost

            for i in range(10):
                oldcost = self._model.cost(self.inputs)
                oldselectionthreshold = self._model.selectionthreshold.get_value()
                newselectionthreshold = oldselectionthreshold + self._rng.randn(*oldselectionthreshold.shape).astype("float32")*self.mutationrate
                newselectionthreshold *= newselectionthreshold > 0.0
                self._model.selectionthreshold.set_value(newselectionthreshold)
                newcost = self._model.cost(self.inputs)
                if newcost > oldcost:
                    self._model.selectionthreshold.set_value(oldselectionthreshold)

            if self._verbose:
                print '> epoch {0:d}, load {1:d}/{2:d}, cost: {3:f}, avg. gradnorm: {4}'.format(
                    self._epochcount, load_index + 1, self._numloads, cost, avg_gradnorm)
                if hasattr(self._model, 'monitor'):
                    self._model.monitor()
            if self.monitor_update_weight_norm_ratio:
                print 'computing update weight norm ratios of last random batch'
                ratios = self.get_update_weight_norm_ratios()
                print 'len(ratios): {0}'.format(len(ratios), )
                for p, ratio in zip(self._params, ratios):
                    print p.name
                    self._update_weight_norm_ratios_log[p].append(ratio)
                    if self._verbose:
                        print p.name, 'update/weight norm ratio: ', ratio
        self.costs.append(cost)
        return cost
