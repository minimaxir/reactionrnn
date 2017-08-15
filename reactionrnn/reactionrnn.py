from keras.layers import Input, Embedding, Dense, GRU
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import json
import h5py
from pkg_resources import resource_filename


class reactionrnn:
    MAXLEN = 140

    def __init__(self, model_path=None,
                 vocab_path=None):

        if model_path is None:
            model_path = resource_filename(__name__,
                                           'reactionrnn_model.hdf5')

        if vocab_path is None:
            vocab_path = resource_filename(__name__,
                                           'reactionrnn_vocab.json')

        with open(vocab_path, 'r') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer(filters='', char_level=True)
        self.tokenizer.word_index = self.vocab
        #self.num_classes = len(self.vocab) + 1
        self.model = load_model(model_path)
        self.model_enc = Model(inputs=self.model.input,
                               outputs=self.model.get_layer('rnn').output)
        #self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

    def generate(self, n=1, return_as_list=False, **kwargs):
        gen_texts = []
        for _ in range(n):
            gen_text = textgenrnn_generate(self.model,
                                           self.vocab,
                                           self.indices_char,
                                           **kwargs)
            if not return_as_list:
                print("{}\n".format(gen_text))
            gen_texts.append(gen_text)
        if return_as_list:
            return gen_texts


def textgenrnn_generate(model, vocab,
                        indices_char, prefix=None, temperature=0.2,
                        maxlen=40, meta_token='<s>',
                        max_gen_length=200):
    '''
    Generates and returns a single text.
    '''

    text = [meta_token] + list(prefix) if prefix else [meta_token]
    next_char = ''

    while next_char != meta_token and len(text) < max_gen_length:
        encoded_text = textgenrnn_encode_sequence(text[-maxlen:],
                                                  vocab, maxlen)
        next_index = textgenrnn_sample(
            model.predict(encoded_text, batch_size=1)[0],
            temperature)
        next_char = indices_char[next_index]
        text += [next_char]
    return ''.join(text[1:-1])


def textgenrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def textgenrnn_encode_training(text, meta_token='<s>', maxlen=40):
    '''
    Encodes a list of texts into a list of texts, and the next character
    in those texts.
    '''

    text_aug = [meta_token] + list(text) + [meta_token]
    chars = []
    next_char = []

    for i in range(len(text_aug) - 1):
        chars.append(text_aug[0:i + 1][-maxlen:])
        next_char.append(text_aug[i + 1])

    return chars, next_char


def textgenrnn_texts_from_file(file_path, header=True, delim='\n'):
    '''
    Retrieves texts from a newline-delimited file and returns as a list.
    '''

    with open(file_path, 'r', encoding="utf-8") as f:
        if header:
            f.readline()
        texts = [line.rstrip(delim) for line in f]
    return texts


def textgenrnn_encode_cat(chars, vocab):
    '''
    One-hot encodes values at given chars efficiently by preallocating
    a zeros matrix.
    '''

    a = np.float32(np.zeros((len(chars), len(vocab) + 1)))
    rows, cols = zip(*[(i, vocab.get(char, 0))
                       for i, char in enumerate(chars)])
    a[rows, cols] = 1
    return a
