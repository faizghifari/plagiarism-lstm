import os
import datetime
import numpy as np
import tensorflow as tf
import keras.backend as K

from time import time
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda, Dropout, Dense, Bidirectional, GlobalMaxPool1D, GlobalMaxPool2D, Activation
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import load_model

from evaluation import precision_m, recall_m, f1_m, confusion_matrix_m, rescale_predictions, eval_summary

class SiameseBiLSTM():
    def __init__(self, embeddings, embedding_dim, max_seq_len, num_lstm, num_hidden, epochs, batch_size):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_lstm = num_lstm
        self.num_hidden = num_hidden
        self.epochs = epochs
        self.batch_size = batch_size
    
    def __build_front_layers(self, input_layer):
        encoded_layer = Embedding(len(self.embeddings), self.embedding_dim, weights=[self.embeddings], 
                                      input_length=self.max_seq_len, trainable=False)(input_layer)
        lstm_layer = Bidirectional(LSTM(self.num_lstm, return_sequences=True))(encoded_layer)
        output_layer = GlobalMaxPool1D()(lstm_layer)

        return output_layer

    def __build_model(self, activation='relu'):
        left_input = Input(shape=(self.max_seq_len,), dtype='int32')
        right_input = Input(shape=(self.max_seq_len,), dtype='int32')

        # Add front layers
        left_output = self.__build_front_layers(left_input)
        right_output = self.__build_front_layers(right_input)

        # Merge layers
        merged = concatenate([left_output, right_output])
        merged = BatchNormalization()(merged)
        merged = Dropout(0.5)(merged)
        merged = Dense(self.num_hidden)(merged)
        merged = BatchNormalization()(merged)
        merged = Activation('relu')(merged)
        merged = Dropout(0.5)(merged)
        merged = Dense(self.num_hidden * 2)(merged)
        merged = BatchNormalization()(merged)
        merged = Activation('relu')(merged)
        merged = Dropout(0.5)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[left_input, right_input], outputs=preds)

        model.summary()

        return model
    
    def train(self, X_train, X_val, X_test, Y_train, Y_val, Y_test, optimizer='nadam', loss='binary_crossentropy'):
        model = self.__build_model()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', f1_m, precision_m, recall_m])

        STAMP = 'lstm_%d_%d' % (self.num_lstm, self.num_hidden)
        checkpoint_dir = './checkpoints/' + str(int(time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        print('START TRAINING ... ')
        training_start_time = time()

        model_trained = model.fit([X_train['left'], X_train['right']], Y_train, batch_size=self.batch_size, 
                                  epochs=self.epochs, validation_data=([X_val['left'], X_val['right']], Y_val), 
                                  callbacks=[early_stopping, model_checkpoint])

        print("Training time finished.\n{} epochs in {}".format(self.epochs, datetime.timedelta(seconds=time()-training_start_time)))

        print('START TESTING ... ')
        test_start_time = time()
        result = model.evaluate([X_test['left'], X_test['right']], Y_test)
        print("Testing time finished.\n in {}".format(datetime.timedelta(seconds=time()-test_start_time)))

        raw_pred = model.predict([X_test['left'], X_test['right']])
        Y_pred = rescale_predictions(raw_pred)

        TN, FP, FN, TP = confusion_matrix_m(Y_test, Y_pred)

        print('PACK UP ALL RESULTS ... ')

        summary = eval_summary(result, model.metrics_names)

        results = {
            'summary': summary,
            'details': {
                'true_positive': int(TP),
                'true_negative': int(TN),
                'false_positive': int(FP),
                'false_negative': int(FN)
            }
        }

        return model_trained, model_path, checkpoint_dir + STAMP, results
    