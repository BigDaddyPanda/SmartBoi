"""

From:
https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras

Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf
For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling

Activation Functions (ohne Advanced):
- elu           elu(x, alpha=1.0)
- selu          selu(x)
- softplus      softplus(x)
- softsign      softsign(x)
- relu          relu(x, alpha=0.0, max_value=None)
- tanh          tanh(x)
- sigmoid       sigmoid(x)
- hard_sigmoid  hard_sigmoid(x)
- linear        linear(x)
- softmax       softmax(x, axis=-1)

"""

import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence

from keras.models import model_from_json
import os
import sys
import pickle as pickle
#import _pickle as cPickle
import warnings
import logging
import matplotlib.pyplot as plt
import timeit
try: import simplejson as json
except ImportError: import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
warnings.filterwarnings("ignore")

startT=timeit.default_timer()
np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static
# CNN-non-static bedeuted: geshuffelte Testdaten

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 300         #org 50 --> britz 128 --> kim 300
filter_sizes = (3, 4, 5)    #org 3,8 --> kim & britz 3,4,5
#filter_sizes = (3, 5, 7)   #?
#num_filters = 10           #org 10
num_filters = 128           #org 10 --> britz 128
#num_filters = 1200         #?
dropout_prob = (0.5, 0.8)   #org 0.5, 0.8 --> kim  0.5 --> britz 0.5
hidden_dims = 200           # org 50 --> kim 100,2 --> britz 200

# Training parameters
batch_size = 64             #org 64 --> kim 50 --> britz 64
num_epochs = 22             #org 20 --> kim 25 --> britz 200

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------


def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
    else:
        x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        y = y.argmax(axis=1)

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        train_len = int(len(x) * 0.9)   # 10% aller zufaelligen Datensaetze sind Testdaten
        x_train = x[:train_len]
        y_train = y[:train_len]
        x_test = x[train_len:]
        y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv, vocabulary


# Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv, vocabulary = load_data(data_source)
# print(len(x_train))
# print(len(x_train[0]))
# print(x_train)

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        print("x_train static shape:", x_train.shape)
        print("x_test static shape:", x_test.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")

# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,) # Anzahl der Trainingsdaten

model_input = Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    # Anzahl Token in dict, Anzahl Dimensionen, anzahl Trainingsdaten
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)    # acc: 74.85%
#model_output = Dense(1, activation="softmax")(z)   # acc: 50.90%

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=2)

stopT=timeit.default_timer()
print('Prepare duration: {}'.format(stopT - startT))

# summarize history for accuracy
plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
#plt.show()


# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
#plt.show()

# ###############
# ###############
# evaluate the model
scores = model.evaluate(x_train, y_train, verbose=0)
print("Evaluate with train data")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(x_test, y_test, verbose=0)
print("Evaluate with test data")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Save model and weights: Serialize them to JSON
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("Saved model to disk")
# Save Dictionary
pickle.dump(vocabulary_inv, open("models/vocabulary.p", "wb"))
print( "Saved vocabulary to disk")

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
print("Graph of the Model saved to disk")


for sentence in x_test:
    sen_list = map(lambda x: vocabulary_inv.get(x), sentence)
#    print(sen_list)
    print(json.dumps(sen_list, indent=4))
    print(' ')
