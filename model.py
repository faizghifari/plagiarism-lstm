import os
import datetime
import numpy as np
import tensorflow as tf
import keras.backend as K

from time import time
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda, Dropout, Dense, Bidirectional, GlobalMaxPool1D, GlobalMaxPool2D, Activation
from keras.optimizers import Adadelta, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session

from evaluation import precision_m, recall_m, f1_m, confusion_matrix_m, rescale_predictions, eval_summary

class SiameseBiLSTM():
    def __init__(self, embeddings, embedding_dim, max_seq_len, num_lstm, num_hidden, epochs, batch_size, lstm_dropout, hidden_dropout, learning_rate, patience):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_lstm = num_lstm
        self.num_hidden = num_hidden
        self.lstm_dropout = lstm_dropout
        self.hidden_dropout = hidden_dropout
        self.learning_rate = learning_rate
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
    
    def __set_gpu_option(self, which_gpu, fraction_memory):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
        config.gpu_options.visible_device_list = which_gpu
        set_session(tf.Session(config=config))
        return

    def __build_front_layers(self, input_layer):
        encoded_layer = Embedding(len(self.embeddings), self.embedding_dim, weights=[self.embeddings], 
                                      input_length=self.max_seq_len, trainable=False)(input_layer)
        lstm_layer = Bidirectional(LSTM(self.num_lstm, recurrent_dropout=self.lstm_dropout, return_sequences=True))(encoded_layer)
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
        merged = Dropout(self.hidden_dropout)(merged)
        merged = Dense(self.num_hidden)(merged)
        merged = BatchNormalization()(merged)
        merged = Activation('relu')(merged)
        merged = Dropout(self.hidden_dropout)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[left_input, right_input], outputs=preds)

        model.summary()

        return model
    
    def train(self, X_train, X_val, X_test, Y_train, Y_val, Y_test, loss='binary_crossentropy'):
        # self.__set_gpu_option('0', 0.9)

        model = self.__build_model()
        nadam = Nadam(learning_rate=self.learning_rate)

        model.compile(loss=loss, optimizer=nadam, metrics=['accuracy', f1_m, precision_m, recall_m])

        STAMP = 'lstm_%d_%d' % (self.num_lstm, self.num_hidden)
        checkpoint_dir = './checkpoints/' + str(int(time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)

        print('START TRAINING ... ')
        training_start_time = time()

        model_trained = model.fit([X_train['left'], X_train['right']], Y_train, batch_size=self.batch_size, 
                                  epochs=self.epochs, validation_data=([X_val['left'], X_val['right']], Y_val), 
                                  callbacks=[early_stopping, model_checkpoint])

        print("Training time finished.\n{} epochs in {}".format(self.epochs, datetime.timedelta(seconds=time()-training_start_time)))

        print('START TESTING ... ')
        test_start_time = time()
        result = model.evaluate([X_test['left'], X_test['right']], Y_test)
        raw_pred = model.predict([X_test['left'], X_test['right']])
        Y_pred = rescale_predictions(raw_pred)
        print("Testing time finished.\n in {}".format(datetime.timedelta(seconds=time()-test_start_time)))

        TN, FP, FN, TP = confusion_matrix_m(Y_test, Y_pred)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall)/(precision + recall))

        result[1] = accuracy
        result[2] = f1_score
        result[3] = precision
        result[4] = recall

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
    
