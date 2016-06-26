# ann_train.py
#
# Trains a neural net on a toy problem to generate a grid of zeros and
# ones. Training data is located in the tdata.txt file and is arranged
# as a 50x50 grid. For training, inputs are normalized on the interval
# [-1.0, 1.0]. The number of training pairs is 50 * 50 = 2500. There is
# a single output node.
import sys
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.callbacks import Callback


def read_tfile(tfname):
    """read_tfile()

    :param tfname: a text file "tdata.txt" containing a 50x50 grid of training data.
    :return: (xtrain, ytrain), where xtrain is a numpy array with shape (2500, 2)
        containing the x1,x2 coordinates of the training data, and ytrain is a numpy
        array of shape (2500, 1) containing values 0.0 or 1.0 corresponding to the
        x1,x2 coordinates in xtrain.
    """
    sdat_ll = list()
    with open(tfname) as fin:
        for rlin in fin:
            lin = rlin.strip()
            if len(lin) < 20:
                continue  # line too small, so ignore it
            sdat_ll.append(lin)
    xdim = len(sdat_ll[0])
    ydim = len(sdat_ll)
    xincr = 2.0 / float(xdim - 1)
    yincr = 2.0 / float(ydim - 1)
    in_ll = list()
    ot_ll = list()
    y = -1.0
    for iy, dlin in enumerate(sdat_ll):
        x = -1.0
        for ix, ch in enumerate(dlin):
            if '.' == ch:
                ot_ll.append((0.0,))
            else:
                ot_ll.append((1.0,))
            in_ll.append((x + xincr * float(ix), y + yincr * float(iy)))
    xtrain = np.array(in_ll)
    ytrain = np.array(ot_ll)
    return xtrain, ytrain


class EpochInstrospector(Callback):
    """Used to save weights after each epoch."""

    def __init__(self, model, optimizer, progress_fname, weights_dir):
        super(EpochInstrospector, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.progress_fname = progress_fname
        self.weights_dir = weights_dir
        self.best_loss = 10000.0

    def _save_weights(self, epoch, loss, acc):
        wgts_np = self.model.get_weights()
        fpname = os.path.join(self.weights_dir, '%05d' % epoch)
        np.save(fpname, wgts_np)
        with open(self.progress_fname, 'a') as fout:
            fout.write("%d\t%.5f\t%5f\n" % (epoch, loss, acc))

    def on_train_begin(self, logs=None):
        self._save_weights(1, 0.5, 0.5)

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        edx = epoch + 2
        self._save_weights(edx, loss, logs['acc'])


class Toy01:
    """Sets up and trains the ANN using keras"""

    def __init__(self, xtrain, ytrain, results_dir):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.results_dir = results_dir
        # ANN model
        self.optimizer = optimizers.RMSprop(lr=0.005)
        self.loss = 'mse'
        self.metrics = 'accuracy'
        self.model = Sequential()
        self.model.add(Dense(output_dim=30, input_dim=2, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_dim=30, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_dim=20, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_dim=1, init='uniform'))
        self.model.add(Activation('sigmoid'))
        # Compile the model
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=[self.metrics]
        )

    def process(self, nepochs):
        progress_fname = os.path.join(self.results_dir, 'progress.txt')
        model_fname = os.path.join(self.results_dir, 'ann_model.txt')
        weights_dir = os.path.join(self.results_dir, 'weights')
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir, 0o755)
        step = EpochInstrospector(self.model, self.optimizer, progress_fname, weights_dir)
        fout = open(progress_fname, 'w')
        fout.close()
        with open(model_fname, 'w') as fout:
            jmodel = self.model.to_json()
            fout.write(jmodel)
        self.model.fit(
            self.xtrain,
            self.ytrain,
            batch_size=30,
            nb_epoch=nepochs,
            verbose=1,
            callbacks=[step],
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None
        )


def run(tfname, results_dir, nepochs):
    xtrain, ytrain = read_tfile(tfname)
    toy = Toy01(xtrain, ytrain, results_dir)
    toy.process(nepochs)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("USE: ann_train <tfile> <resultsDir> <nepochs>")
        sys.exit()
    g_tfname = sys.argv[1]
    g_results_dir = sys.argv[2]
    g_nepochs = int(sys.argv[3])
    run(g_tfname, g_results_dir, g_nepochs)
