from keras.layers import Input, Embedding, Dense, GRU
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import json
import h5py
from pkg_resources import resource_filename
from collections import OrderedDict
from keras import backend as K


class reactionrnn:
    MAXLEN = 140
    REACTIONS = ['love', 'wow', 'haha', 'sad', 'angry']

    def __init__(self, weights_path=None,
                 vocab_path=None):

        if weights_path is None:
            weights_path = resource_filename(__name__,
                                             'reactionrnn_weights.hdf5')

        if vocab_path is None:
            vocab_path = resource_filename(__name__,
                                           'reactionrnn_vocab.json')

        with open(vocab_path, 'r') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer(filters='', char_level=True)
        self.tokenizer.word_index = self.vocab
        self.num_classes = len(self.vocab) + 1
        self.model = reactionrnn_model(weights_path, self.num_classes)
        self.model_enc = Model(inputs=self.model.input,
                               outputs=self.model.get_layer('rnn').output)

    def predict(self, text, **kwargs):
        text_enc = reactionrnn_encode_sequence(text, self.vocab)
        predicts = self.model.predict(text_enc)[-1]
        predicts_dict = {react: round(float(predicts[i]), 2)
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


def reactionrnn_model(weights_path, num_classes, maxlen=140):
    '''
    Builds the model architecture for textgenrnn and
    loads the pretrained weights for the model.
    '''

    input = Input(shape=(maxlen,), name='input')
    embedded = Embedding(num_classes, 100, input_length=maxlen,
                         name='embedding')(input)
    rnn = GRU(256, return_sequences=False, name='rnn')(embedded)
    output = Dense(5, name='output',
                   activation=lambda x: K.clip(x, 0., 1.))(rnn)

    model = Model(inputs=[input], outputs=[output])
    model.load_weights(weights_path, by_name=True)
    model.compile(loss='mse', optimizer='nadam')
    return model


def reactionrnn_encode_sequence(text, vocab, maxlen=140):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)
