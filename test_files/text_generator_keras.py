from keras.models import Model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Input, Bidirectional
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import plot_model
import numpy as np
import random
import sys
import csv
import os
import h5py
import time
from keras import backend as K
from keras.callbacks import Callback


class EpochStatsLogger(Callback):

    def on_train_begin(self, logs={}):
        filename = os.path.basename(sys.argv[0])[:-3]
        backend = K.backend()
        self.f = open('logs/{}_{}.csv'.format(filename, backend), 'w')
        self.log_writer = csv.writer(self.f)
        self.log_writer.writerow(['epoch', 'elapsed', 'loss'])

    def on_train_end(self, logs={}):
        self.f.close()

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.log_writer.writerow([epoch, time.time() - self.start_time,
                                  logs.get('main_out_loss')])

logger = EpochStatsLogger()

embeddings_path = "glove.840B.300d-char.txt"
embedding_dim = 300
batch_size = 128
use_pca = False
lr = 0.001
lr_decay = 1e-4
maxlen = 40
consume_less = 2   # 0 for cpu, 2 for gpu

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters

step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))


print('Vectorization...')
X = np.zeros((len(sentences), maxlen), dtype=np.int)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_indices[char]
    y[i, char_indices[next_chars[i]]] = 1


print('Build model...')
main_input = Input(shape=(maxlen,))
embedding_layer = Embedding(
    len(chars), embedding_dim, input_length=maxlen)

embedded = embedding_layer(main_input)

# RNN Layer
rnn = LSTM(128, implementation=consume_less)(embedded)

aux_output = Dense(len(chars))(rnn)
aux_output = Activation('softmax', name='aux_out')(aux_output)

# Hidden Layers
hidden_1 = Dense(512, use_bias=False)(rnn)
hidden_1 = BatchNormalization()(hidden_1)
hidden_1 = Activation('relu')(hidden_1)

hidden_2 = Dense(256, use_bias=False)(hidden_1)
hidden_2 = BatchNormalization()(hidden_2)
hidden_2 = Activation('relu')(hidden_2)

main_output = Dense(len(chars))(hidden_2)
main_output = Activation('softmax', name='main_out')(main_output)

model = Model(inputs=main_input, outputs=[main_output, aux_output])

optimizer = Adam(lr=lr, decay=lr_decay)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, loss_weights=[1., 0.2])
model.summary()

# plot_model(model, to_file='model.png', show_shapes=True)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


for iteration in [1]:
    print()
    print('-' * 50)
    print('Iteration', iteration)

    model.fit(X, [y, y], batch_size=batch_size,
                        epochs=10, callbacks=[logger])

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.5]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(1200):
            x = np.zeros((1, maxlen), dtype=np.int)
            for t, char in enumerate(sentence):
                x[0, t] = char_indices[char]

            preds = model.predict(x, verbose=0)[0][0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

