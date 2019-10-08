import os
import json
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path
from model import SiameseBiLSTM
from preprocess import build_vocab_and_transform, build_embeddings, build_train_data, build_test_data, build_padded_data
from evaluation import precision_m, recall_m, f1_m, confusion_matrix_m

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

TRAIN_PATH = os.getenv('TRAIN_PATH')
TEST_PATH = os.getenv('TEST_PATH')
WORD2VEC_PATH = os.getenv('WORD2VEC_PATH')

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM'))
MAX_SEQ_LEN = int(os.getenv('MAX_SEQ_LEN'))
VAL_SIZE = float(os.getenv('VAL_SIZE'))
NUM_LSTM = int(os.getenv('NUM_LSTM'))
NUM_HIDDEN = int(os.getenv('NUM_HIDDEN'))
LSTM_DROPOUT = float(os.getenv('LSTM_DROPOUT'))
HIDDEN_DROPOUT = float(os.getenv('HIDDEN_DROPOUT'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
PATIENCE = int(os.getenv('PATIENCE'))
EPOCHS = int(os.getenv('EPOCHS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

if __name__ == "__main__":
    print('LOAD DATA ... ')
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print('DATA LOADED...')

    columns = ['generated_text', 'compared_text']

    print('BUILD VOCABULARY AND TRANSFORM DATA ... ')
    train_df, test_df, vocab, inv_vocab, w2v = build_vocab_and_transform(train_df, test_df, columns, WORD2VEC_PATH)
    print('BUILD EMBEDDINGS ... ')
    embeddings = build_embeddings(w2v, EMBEDDING_DIM, vocab)
    print('BUILD TRAIN DATA ... ')
    X_train, X_val, Y_train, Y_val = build_train_data(train_df, columns, VAL_SIZE)
    print('BUILD TEST DATA ... ')
    X_test, Y_test = build_test_data(test_df, MAX_SEQ_LEN)
    print('PAD DATA ... ')
    X_train, X_val = build_padded_data(X_train, X_val, MAX_SEQ_LEN)
    print('FINISH DATA PREPARATION \n')

    print('INITIALIZE MODEL ... ')
    model = SiameseBiLSTM(embeddings, EMBEDDING_DIM, MAX_SEQ_LEN, NUM_LSTM, NUM_HIDDEN, EPOCHS, BATCH_SIZE, 
                          LSTM_DROPOUT, HIDDEN_DROPOUT, LEARNING_RATE, PATIENCE)

    model_trained, model_path, checkpoint_dir, results = model.train(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    file_obj = open(checkpoint_dir + '.json', 'w')
    json.dump(results, file_obj)

    print('TRAIN FINISHED, RESULTS AVAILABLE AT : ')
    print(checkpoint_dir + '.json')
