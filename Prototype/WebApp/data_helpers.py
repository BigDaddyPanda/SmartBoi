import numpy as np
import re
import itertools
import os
from collections import Counter

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def add_Problem(problem):
    #print(problem)
    problem_words = problem.split()
    pw_len = len(problem_words)
    check_success = True            # New problem should be added
    errors = []
    positive_examples_filename = "../data/rt-polarity.pos"
    negative_examples_filename = "../data/rt-polarity.neg"
    negative_pool_filename = "../data/rt-polarity.pool"          # Take negative_examples from pool
    negative_pool_tmp_filename = "../data/rt-polarity.pooltmp"

    f2check=[positive_examples_filename, negative_examples_filename, negative_pool_filename]
    # check if files exists
    for f in f2check:
        if not(os.path.exists(f)):
            check_success = False
            errors.append("File %s does not exist" %f)

    # Step1: Check if problem is already a problem in our data
    positive_examples = list(open(positive_examples_filename).readlines())
    for sentence in positive_examples:
        pos_ex_word_list = sentence.split()
        ewl_len = len(pos_ex_word_list)
        # print("problem_len: %d sentence_len: %d \n" % (pw_len, ewl_len))
        if ewl_len == pw_len:           # if word length is equal, check if problem is new
            matches = 0
            for word in pos_ex_word_list:
                if word in problem_words:
                    matches = matches + 1
            # print("problem_len: %d matches: %d \n" % (pw_len, matches))
            if pw_len < (matches + 2):   # new problems should be slightly different from old ones
                errors.append("New problems should be different from old ones")
                check_success = False

    # Step2: Check if Pool size is big enough
    negative_pool = list(open(negative_pool_filename).readlines())
    if len(negative_pool) < 1: # Not enough negative_examples
        errors.append("Not enough negative_examples in negative_pool")
        check_success = False

    if check_success:
        # Step3: write new problem to positive_examples file
        with open(positive_examples_filename, "a") as f:
            f.write(problem + '\n')

        # Step4: get one none problem from pool file
        n = 1   # read n lines
        nfirstlines = []
        with open(negative_pool_filename) as f, open(negative_pool_tmp_filename, "w") as out:
            for x in xrange(n):                 # read first n lines
                nfirstlines.append(next(f))
            for line in f:                      # stream data to tmp file
                out.write(line)
        # remove old pool file and make tmp to old
        os.remove(negative_pool_filename)
        os.rename(negative_pool_tmp_filename, negative_pool_filename)

        # Step5: add non problem to negative_examples file
        with open(negative_examples_filename, "a") as f:
            for line in nfirstlines:
                f.write(line)
        errors.append("Problem successfully added")
    return (check_success, errors)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("../data/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("../data/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
