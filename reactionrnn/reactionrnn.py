from keras.layers import Input, Embedding, Dense, GRU
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import json
import h5py
from pkg_resources import resource_filename
from collections import OrderedDict


class reactionrnn:
    MAXLEN = 140
    REACTIONS = ['love', 'wow', 'haha', 'sad', 'angry']

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
        self.model = load_model(model_path)
        self.model_enc = Model(inputs=self.model.input,
                               outputs=self.model.get_layer('rnn').output)

    def predict(self, text, **kwargs):
        text_enc = reactionrnn_encode_sequence(text, self.vocab)
        predicts = np.around(self.model.predict(text_enc)[-1], decimals=2)
        predicts_dict = {react: predicts[i]
                         for i, react in enumerate(self.REACTIONS)}
        predicts_dict = OrderedDict(sorted(predicts_dict.items(),
                                           key=lambda t: -t[1]))
        return predicts_dict

    def predict_label(self, text, **kwargs):
        text_enc = reactionrnn_encode_sequence(text, self.vocab)
        predicts = np.around(self.model.predict(text_enc)[-1], decimals=2)
        return self.REACTIONS[np.argmax(predicts)]

    def encode(self, text, **kwargs):
        text_enc = reactionrnn_encode_sequence(text, self.vocab)
        predicts = self.model_enc.predict(text_enc)[-1]
        return predicts


def reactionrnn_encode_sequence(text, vocab, maxlen=140):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)
