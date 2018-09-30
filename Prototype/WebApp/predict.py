"""
Some Text


"""
import os
import sys
import json
import numpy as np
import data_helpers
import timeit
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.datasets import imdb
from keras.preprocessing import sequence, text
from os.path import join, exists, split
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import pickle as pickle
import tensorflow as tf


# ---------------------- Parameters section -------------------
#

# Log Level
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("we coool")
# Prepossessing parameters
sequence_length = 400
max_words = 5000

w2v = ""

#
# ---------------------- Parameters end -----------------------

def load_data(data_source):
    #global sequence_length
    assert data_source in ["keras_data_set", "local_dir", "pickle"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None, oov_char=None, index_from=None)
        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test  = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"


    # elif data_source == "pickle":
    #     vocabulary_inv = pickle.load(open(".models/vocabulary.p","rb"))
    #     return "","","","",vocabulary_inv
    else:
        x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        y = y.argmax(axis=1)

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        train_len = int(len(x) * 0.9)
        x_train = x[:train_len]
        y_train = y[:train_len]
        x_test = x[train_len:]
        y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv

# Data Preparation
def load_dict(data_source):
    print("Load data...")
    x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source)
    vocabulary = dict((v, k) for k, v in vocabulary_inv.items())
    return vocabulary

def load_model():
    print("Load model...")

    graph = tf.get_default_graph()
    json_file = open('./models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./models/model.h5")
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("Model loaded...")
    loaded_model._make_predict_function()
    return loaded_model

def load_w2v():
    model_dir = '../data'
    #model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = "GoogleNews-vectors-negative300.bin"
    # model_name = "Test.bin"
    model_name = join(model_dir, model_name)
    global w2v
    if exists(model_name):
        #embedding_model = word2vec.Word2Vec.load(model_name)
        w2v = KeyedVectors.load_word2vec_format(model_name, binary=True)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    return

def predict_Problem(loaded_model, vocabulary, sentence):
    print("sentence Before",sentence)
    sentence = sentence.split(" ")
    #print(json.dumps(vocabulary , indent=4))
    sentence = map(lambda x: data_helpers.clean_str(x), sentence)
    print("sentence after",str(sentence))
    sequence_length=49
    # Schauen ob word in dictionary vorhanden ist. Falls ja dessen Index im dict einer Liste hinzufuegen

    startT=timeit.default_timer()
    pred=[]
    for word in sentence:
        print("Predecting",word)
        if not word:
            continue
        x = vocabulary.get(word)
        if x != None:
            pred.append(x)
        else:   # Testen ob in w2v ein aehnliches wort gefunden werden kann und ob dieses im Dictionary ist
            print('Word {} is not in dict'.format(word))
            #w2vList=w2v.wv.most_similar(word)
            try:
                w2vList=w2v.most_similar(word)
            except Exception:
                pred.append(0)
                continue
            for w in w2vList:
                w=w[0]
                print(w)
                print('similarity: {} in dict: {}'.format(w, vocabulary.get(w)))
                if vocabulary.get(w) != None:
                    pred.append(vocabulary.get(w))
                    break
    stopT=timeit.default_timer()
    print('Prepare duration: {}'.format(stopT - startT),"Result:\n",pred)
    # <PAD/> einfuegen falls word nicht im dict?
    #	else:
    #		pred.append(0)
    #		print(str(word) + " " + str(x))
    #print("test1")
    #print(json.dumps(pred, indent=4))
    # Die eben erzeugte Liste mit 0 --> <PAD/> auf die Laenge der sequence_length (400) auffuellen
    
    empty_spaces = sequence_length - len(pred)
    for x in range(0, empty_spaces):
        pred.append(0)

    #print("test2")
    #print(json.dumps(pred, indent=4))
    # Liste zu Numpy array konvertieren
    pred = np.array(pred)
    pred = pred[None, :]
    # print("Liste zu Numpy",json.dumps(pred, indent=4))
    pred.T

    #print("test3")
    #print(pred)
    #print(json.dumps(pred, indent=4))

    print("Pred before: ")
    startT=timeit.default_timer()
    prediction = loaded_model.predict(pred, verbose=1)
    stopT=timeit.default_timer()
    print('Prediction duration: {}'.format(stopT - startT))
    #print("Pred#: ")
    #print(prediction)
    return [prediction, sequence_length, pred]

def predict_Problem_List(loaded_model, vocabulary, sentence_list):
    ret_p = []
    ret_c = []
    sequence_length=49
    for line in sentence_list:
        sentence = line.split(" ")
        pred=[]
        #print("Len: %d \n" %len(sentence))
        if 1 < len(sentence):
            for word in sentence:
                x = vocabulary.get(word)
                if x != None:
                    pred.append(x)

            iter = sequence_length - len(pred)
            for x in range(0, iter):
                pred.append(0)

            pred = np.array(pred)
            pred = pred[None, :]
            pred.T
            prediction = loaded_model.predict(pred, verbose=0)
            prediction = prediction.tolist()[0][0]
            prediction = prediction * 100 if prediction else None
            if prediction <= 40:
                form_class = 'bg-danger'
            elif prediction > 40 and prediction < 60:
                form_class = 'bg-warning'
            else:
                form_class = 'bg-success'
            ret_c.append(form_class)
            ret_p.append(round(prediction, 2))


    return [ret_c, ret_p, sequence_length, pred]
