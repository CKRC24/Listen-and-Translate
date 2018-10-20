
# coding: utf-8

import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model, Model
import pickle
import h5py
from sys import argv
from gensim.models import Word2Vec

test_csv_path = argv[1]
embed_dim = 300
embed_model = Word2Vec.load('./word_embedding/word_embedding.bin')
vocab_size = embed_model.wv.syn0.shape[0]
unk_index = vocab_size
pad_index = vocab_size+1


def word2id(sentences, max_length):
    t_list = []
    for s in sentences:
        tokens = s.split()
        tokens = [embed_model.wv.vocab[token].index
                  if token in embed_model.wv.vocab else unk_index
                  for token in tokens]
        target_array = np.array(tokens + [pad_index]*(max_length-len(tokens)))
        t_list.append(target_array)
    return np.array(t_list)

test_speech_data = np.load('./data/test_input_data.npy')
test_caption = pd.read_csv(test_csv_path, header=None)
test_size = len(test_caption)

# 處理資料，把句子接在一起
test_options = []
for i in range(test_size):
    t_opt = []
    for j in range(4):
        temp = test_caption[j][i]
        t_opt.append(temp)
    test_options.append(t_opt)
test_options = np.array(test_options)


test_options_index = []
for i in range(test_size):
    test_options_index.append(word2id(test_options[i], 14))

dim = [512]
seeds = [10, 20, 30, 40, 50]
seeds_2 = [2, 28, 11, 25]
scores = np.zeros((test_size, 4))

print("Start predicting ...")
for i in range(len(seeds)):
    print("Predicting dim[%d] seed[%d]" % (dim[0],seeds[i]))
    model = load_model('model/stacking_ds_' + str(dim[0]) + '_' + str(i+1) + '.h5')
    for t in range(test_size):
        for opt in range(4):
            pred = model.predict([test_speech_data[t].reshape(1, 246, -1),
                                  test_options_index[t][opt].reshape(1, -1)])
            scores[t][opt] += pred[0]

for i in range(len(seeds_2)):
    print("Predicting dim[%d] seed[%d]" % (dim[0],seeds[i]))
    model = load_model('model/stacking_ds_ver2_' + str(dim[0]) + '_' + str(i+1) + '.h5')
    for t in range(test_size):
        for opt in range(4):
            pred = model.predict([test_speech_data[t].reshape(1, 246, -1),
                                  test_options_index[t][opt].reshape(1, -1)])
            scores[t][opt] += pred[0]

ans = np.argmax(scores, axis=1)
index = np.arange(1, test_size+1)
df = pd.DataFrame({'id': index, 'answer': ans})
df.to_csv('./prediction/prediction.csv', index=False)
