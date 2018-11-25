import gzip
import gensim
import logging
import os

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if i == 1:
                break

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


def train_prototype_model():
    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "../reviews_data.txt.gz")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=150,
        window=10,
        min_count=2,
        workers=10)
    # model.wv.load(os.path.join(abspath, "./vectors/default"))
    model.train(documents, total_examples=len(documents), epochs=2)
    return model


def evaluate_sentence(sentence, ref_model):
    sentence = list(sentence)
    result = []
    for word in sentence:
        ls = [i for i in sentence if i != word and len(i) > 2]
        try:
            sims = ref_model.wv.similarity(ls)
        except Exception as E:
            sims = [('', 0)]
        result.append(sum([t[1] for t in sims]) / len(sims))

    return sum(result) / len(result)


