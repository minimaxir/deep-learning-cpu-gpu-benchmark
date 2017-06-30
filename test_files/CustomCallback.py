import sys
import csv
import os
import time
from keras import backend as K
from keras.callbacks import Callback


class EpochStatsLogger(Callback):

    def on_train_begin(self, logs={}):
        filename = os.path.basename(sys.argv[0])[:-3]
        backend = K.backend()
        self.f = open('logs/{}_{}.csv'.format(filename, backend), 'w')
        self.log_writer = csv.writer(self.f)
        self.log_writer.writerow(['epoch', 'elapsed', 'loss',
                                  'acc', 'val_loss', 'val_acc'])

    def on_train_end(self, logs={}):
        self.f.close()

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.log_writer.writerow([epoch, time.time() - self.start_time,
                                  logs.get('loss'),
                                  logs.get('acc'),
                                  logs.get('val_loss'),
                                  logs.get('val_acc')])
