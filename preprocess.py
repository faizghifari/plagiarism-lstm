import re
import itertools
import numpy as np

from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

def __load_stopwords(STOPWORDS_FILE):
    stopwords = set()
    with open(STOPWORDS_FILE, 'r') as f:
        for line in f:
            stopwords.add(line.strip().lower())
    return stopwords

def __is_ok_word(word):
    alphanumeric = set('abcdefghijklmnopqrstuvwxyz0123456789')
    for c in word:
        if c in alphanumeric:
            return True
    return False

def __remove_stopwords(stopwords, query):
    q_split = query.split()
    result = [w for w in q_split if (w not in stopwords) and (__is_ok_word(w))]
    return ' '.join(result)

def __normalize_t2w(text, remove_stopwords):
    stopwords = __load_stopwords('./data/stopwords.txt')
    text = re.sub(r'[^\w]', ' ', text)
    if remove_stopwords :
        text = __remove_stopwords(stopwords, text)
    text = text.lower()
    text = ' '.join(text.split())
    
    return text.split()

def build_vocab_and_transform(df, columns, w2v_path, remove_stopwords):
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    vocab = dict()
    inv_vocab = ['<UNK>']

    for dataset in [df]:
        for index, row in dataset.iterrows():
            for col in columns:
                t2n = []
                text = __normalize_t2w(row[col], remove_stopwords)
                for word in text:
                    if word not in w2v.vocab:
                        continue
                    if word not in vocab:
                        vocab[word] = len(inv_vocab)
                        t2n.append(len(inv_vocab))
                        inv_vocab.append(word)
                    else:
                        t2n.append(vocab[word])
                        
                dataset.set_value(index, col, t2n)
    
    return df, vocab, inv_vocab, w2v

def build_embeddings(w2v, embedding_dim, vocab):
    embeddings = 1 * np.random.randn(len(vocab) + 1, embedding_dim)
    embeddings[0] = 0
    for word, index in vocab.items():
        if word in w2v.vocab:
            embeddings[index] = w2v.word_vec(word)
    
    return embeddings

def build_train_data(train_df, columns, val_size):
    X = train_df[columns]
    Y = train_df['is_plagiarism']

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size)
    X_train = {
        'left': X_train.generated_text,
        'right': X_train.compared_text
    }
    X_val = {
        'left': X_val.generated_text,
        'right': X_val.compared_text
    }
    Y_train = Y_train.values

    return X_train, X_val, Y_train, Y_val

def build_test_data(test_df, maxlen):
    X_test = {
        'left': test_df.generated_text,
        'right':test_df.compared_text
    }
    for dataset, side in itertools.product([X_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=maxlen)

    Y_test = test_df['is_plagiarism'].values
    
    return X_test, Y_test

def build_padded_data(X_train, X_val, maxlen):
    for dataset, side in itertools.product([X_train, X_val], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=maxlen)
    
    return X_train, X_val
