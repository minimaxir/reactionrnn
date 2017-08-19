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

    def predict(self, texts, **kwargs):
        texts_enc = reactionrnn_encode_sequences(texts, self.tokenizer)
        predicts = self.model.predict(texts_enc, batch_size=1)
        if len(texts_enc) == 1:
            predicts_dict = {react: round(float(predicts[0][i]), 4)
                             for i, react in enumerate(self.REACTIONS)}
            predicts_dict = OrderedDict(sorted(predicts_dict.items(),
                                               key=lambda t: -t[1]))
            return predicts_dict
        return predicts

    def predict_label(self, texts, **kwargs):
        texts_enc = reactionrnn_encode_sequences(texts, self.tokenizer)
        predicts = self.model.predict(texts_enc, batch_size=1)
        return list(np.array(self.REACTIONS)[np.argmax(predicts, axis=1)])

    def encode(self, texts, **kwargs):
        text_enc = reactionrnn_encode_sequences(texts, self.tokenizer)
        predicts = self.model_enc.predict(text_enc)
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
                   activation=lambda x: K.relu(x) / K.sum(K.relu(x),
                                                          axis=-1))(rnn)

    model = Model(inputs=[input], outputs=[output])
    model.load_weights(weights_path, by_name=True)
    model.compile(loss='mse', optimizer='nadam')
    return model


def reactionrnn_encode_sequences(texts, tokenizer, maxlen=140):
    '''
    Encodes text(s) into the corresponding encoding(s) for prediction(s) with
    the model.
    '''

    texts = texts if isinstance(texts, list) else [texts]
    texts_enc = tokenizer.texts_to_sequences(texts)
    texts_enc = sequence.pad_sequences(texts_enc, maxlen=maxlen)
    return texts_enc
