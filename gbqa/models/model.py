import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Lambda, LSTM, GRU, Bidirectional, Dense, Embedding, Attention, Concatenate, TimeDistributed, Multiply, RNN, Conv1D, UpSampling1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras import backend as K

from gbqa.configs import DEFAULT_NMT_MODEL_CONFIG, DATASET_CONFIG


class CustomLSTMCell(tf.keras.layers.LSTMCell):
    def __init__(self, units, encoder_sequence_shape, **kwargs):
        super().__init__(units, **kwargs)
        self.state_size = [[units], [units], encoder_sequence_shape[1:]]

    def build(self, input_shape):
        input_shape = (*input_shape[:-1], input_shape[-1]*2)
        super().build(input_shape)
        self.attention = Attention()
    
    def call(self, input, states, training=None):
        encoder_outputs = states[2]
        states = states[:2]
        state_h = tf.expand_dims(states[0], axis=1)
        input = tf.expand_dims(input, axis=1)

        attention_outputs = self.attention([state_h, encoder_outputs])
        # print(input.shape, encoder_outputs.shape, attention_outputs.shape)
        input = tf.squeeze(Concatenate()([input, attention_outputs]), axis=1)
        # print(input.shape, self.kernel.shape, states)

        output, states = super().call(input, states, training)
        return output, [*states, encoder_outputs]


class NMTModel:
    def __init__(self, 
                 weight_path=None,
                 embedding_dim=DEFAULT_NMT_MODEL_CONFIG["EMBEDDING_DIM"],
                 laten_dim=DEFAULT_NMT_MODEL_CONFIG["LATENT_DIM"]):
        tf.keras.backend.clear_session()

        self.embedding_dim = embedding_dim
        self.latent_dim = laten_dim

        self.__create_model()

        if weight_path != None:
            self.model.load_weights(weight_path)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
            experimental_steps_per_execution = 2, 
            loss='sparse_categorical_crossentropy'
        )

        self.callbacks = []

    def __create_model(self):
        max_input_len = DATASET_CONFIG["MAX_INPUT_LENGTH"]
        max_output_len = DATASET_CONFIG["MAX_DECODER_LENGTH"]
        input_vocab_size = DATASET_CONFIG["INPUT_VOCABULARY_SIZE"]
        output_vocab_size = DATASET_CONFIG["SPARQL_VOCABULARY_SIZE"]
        # ENCODER
        enc_input = Input(shape=(max_input_len), name="encoder_input")
        enc_embedding = Embedding(input_vocab_size+1, self.embedding_dim, mask_zero=True, name="encoder_embedding")(enc_input)
        enc_forward = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.8, name="encoder_forward")
        enc_backward = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.8, go_backwards=True, name="encoder_backward")
        forward_outputs, forward_h, forward_c = enc_forward(enc_embedding)
        backward_outputs, backward_h, backward_c = enc_backward(enc_embedding)

        enc_outputs = Concatenate(name="encoder_out_concat")([forward_outputs, backward_outputs])
        enc_state_h = Concatenate(name="encoder_state-h_concat")([forward_h, backward_h])
        enc_state_c = Concatenate(name="encoder_state-c_concat")([forward_c, backward_c])
        
        # TEMPLATE CLASSIFIER

        # DECODER 
        constants = K.ones((K.shape(enc_input)[0], max_output_len, self.latent_dim*2))

        decoder = tf.keras.layers.RNN(CustomLSTMCell(self.latent_dim*2, enc_outputs.shape, dropout=0.8), return_sequences=True, name="decoder")
        dec_outputs = decoder(constants, initial_state=[enc_state_h, enc_state_c, enc_outputs])

        # Time-distributed classifier with tanh layer connecting to softmax layer
        timedist_tanh = TimeDistributed(Dense(self.latent_dim*8, activation="tanh"))
        timedist_softmax = TimeDistributed(Dense(output_vocab_size+1, activation="softmax"))
        outputs = timedist_softmax(timedist_tanh(dec_outputs))
        
        # nmt_model = Model([enc_input, dec_input], outputs)
        nmt_model = Model(enc_input, outputs)
        
        self.model = nmt_model

    def add_train_callback(self, callback):
        self.callbacks.append(callback)

    def fit(self, x, y, 
            batch_size=32, epochs=100, 
            validation_split=0.1, 
            workers=4, verbose=1):
        self.model.fit(x, y, 
                       batch_size=batch_size, 
                       epochs=epochs, callbacks=self.callbacks, 
                       validation_split=validation_split, 
                       workers=workers, verbose=verbose)
    
    def evaluate(self, x, y_true):
        cross_entropy = self.model.evaluate(x, y_true)
        print("Perplexity on test set:", np.exp(cross_entropy))

    def predict(self, x):
        return self.model.predict([x])

class TemplateBasedModel:
    def __init__(self,
                 classifier_weight_path=None, recognizor_weight_path=None,
                 embedding_dim=DEFAULT_NMT_MODEL_CONFIG["EMBEDDING_DIM"],
                 laten_dim=DEFAULT_NMT_MODEL_CONFIG["LATENT_DIM"]):
        tf.keras.backend.clear_session()

        self.embedding_dim = embedding_dim
        self.latent_dim = laten_dim

        self.__create_model()

        if classifier_weight_path != None and recognizor_weight_path != None:
            self.classifier.load_weights(classifier_weight_path)
            self.recognizer.load_weights(recognizor_weight_path)

        self.template_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
            experimental_steps_per_execution = 2, 
            loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']
        )
        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
            experimental_steps_per_execution = 2,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.recognizer.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
            experimental_steps_per_execution = 2,
            loss='sparse_categorical_crossentropy'
        )

        self.callbacks = []

    def __create_model(self):
        max_input_len = DATASET_CONFIG["MAX_INPUT_LENGTH"]
        max_output_len = DATASET_CONFIG["MAX_DECODER_LENGTH"]
        input_vocab_size = DATASET_CONFIG["INPUT_VOCABULARY_SIZE"]
        output_vocab_size = DATASET_CONFIG["SPARQL_VOCABULARY_SIZE"]
        template_count = DATASET_CONFIG["TEMPLATE_COUNT"]
        max_entity_number = DATASET_CONFIG["MAX_ENTITY_NUMBER"]
        entity_vocab_size = DATASET_CONFIG["ENTITY_VOCABULARY_SIZE"]
        dropout_rate = 0.4
        # BiLSTM ENCODER
        enc_input = Input(shape=(max_input_len), name="encoder_input")
        enc_embedding = Embedding(input_vocab_size+1, self.embedding_dim, mask_zero=True, name="encoder_embedding")(enc_input)
        enc_forward = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=dropout_rate, name="encoder_forward")
        enc_backward = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=dropout_rate, go_backwards=True, name="encoder_backward")
        forward_outputs, forward_h, forward_c = enc_forward(enc_embedding)
        backward_outputs, backward_h, backward_c = enc_backward(enc_embedding)

        enc_outputs = Concatenate(name="encoder_out_concat")([forward_outputs, backward_outputs]) # (None, 40, 1024)
        enc_state_h = Concatenate(name="encoder_state-h_concat")([forward_h, backward_h]) # (None, 1024)
        enc_state_c = Concatenate(name="encoder_state-c_concat")([forward_c, backward_c]) # (None, 1024)

        # Attention please!!!
        self_attention = Attention(use_scale=True)
        att_vectors = self_attention([enc_outputs, enc_outputs])
        att_merged = Concatenate()([enc_outputs, att_vectors])
        att_outputs = TimeDistributed(Dense(self.latent_dim*2, activation='relu'))(att_merged)

        # TEMPLATE CLASSIFIER
        temp_output = Dense(template_count, activation='softmax', name="template")(att_outputs[:, 0]) # CLS Token
        
        # DECODER 
        temp_embedding = Dense(self.latent_dim*2, activation='relu')(temp_output)
        constants = K.ones((K.shape(enc_input)[0], max_entity_number, self.latent_dim*2))
        
        dec = tf.keras.layers.RNN(CustomLSTMCell(self.latent_dim*2, att_outputs.shape, dropout=dropout_rate), return_sequences=True, name="decoder")
        dec_outputs = dec(constants, initial_state=[temp_embedding, enc_state_c, enc_outputs])

        # Time-distributed classifier with tanh layer connecting to softmax layer
        timedist_relu= TimeDistributed(Dense(self.latent_dim*4, activation="relu"))
        timedist_softmax = TimeDistributed(Dense(entity_vocab_size+1, activation="softmax", name="entities"))
        ent_outputs = timedist_softmax(timedist_relu(dec_outputs))

        # Finalize the model:
        template_classifier = Model(enc_input, temp_output)
        entity_recognizer = Model(enc_input, ent_outputs)
        template_model = Model(enc_input, [temp_output, ent_outputs])
        
        self.template_model = template_model
        self.classifier = template_classifier
        self.recognizer = entity_recognizer

    def add_train_callback(self, callback):
        self.callbacks.append(callback)

    def fit(self, x, y, 
            batch_size=32, epochs=100, 
            validation_split=0.1, 
            workers=4, verbose=1):
        self.template_model.fit(x, y, 
                       batch_size=batch_size, 
                       epochs=epochs, callbacks=self.callbacks, 
                       validation_split=validation_split, 
                       workers=workers, verbose=verbose)
    
    def evaluate(self, x, y_true):
        temp_loss, temp_acc = self.classifier.evaluate(x, y_true[0])
        print("Template Classification Loss:", temp_loss)
        print("Template Classification Accuracy:", temp_acc)

        ent_loss = self.recognizer.evaluate(x, y_true[1])
        print("Entity Recognition Loss:", ent_loss)
        print("Entity Recognition Perplexity:", np.exp(ent_loss))

    def predict(self, x):
        return [self.classifier.predict(x), self.recognizer.predict(x)]