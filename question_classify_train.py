#!/usr/bin/env python3
# coding: utf-8
# File: question_classify.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-10

import os
import numpy as np
import jieba.posseg as pseg
from keras.models import Sequential, load_model
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dense, Dropout, LSTM, Bidirectional

class QuestionClassify(object):
    def __init__(self):
        self.label_dict = {
            0:"婚姻家庭",
            1:"劳动纠纷",
            2:"交通事故",
            3:"债权债务",
            4:"刑事辩护",
            5:"合同纠纷",
            6:"房产纠纷",
            7:"侵权",
            8:"公司法",
            9:"医疗纠纷",
            10:"拆迁安置",
            11:"行政诉讼",
            12:"建设工程"
            }
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_file = os.path.join(cur, 'question_train.txt')
        self.embedding_path = os.path.join(cur, 'word_vec_300.bin')
        self.embdding_dict = self.load_embedding(self.embedding_path)
        self.max_length = 60
        self.embedding_size = 300
        self.lstm_modelpath = 'model/lstm_question_classify.h5'
        self.cnn_modelpath = 'model/cnn_question_classify.h5'
        return

    '''加载词向量'''
    def load_embedding(self, embedding_path):
        embedding_dict = {}
        count = 0
        for line in open(embedding_path):
            line = line.strip().split(' ')
            if len(line) < 300:
                continue
            wd = line[0]
            vector = np.array([float(i) for i in line[1:]])
            embedding_dict[wd] = vector
            count += 1
            if count%10000 == 0:
                print(count, 'loaded')
        print('loaded %s word embedding, finished'%count, )
        return embedding_dict

    '''对文本进行分词处理'''
    def seg_sent(self, s):
        wds = [i.word for i in pseg.cut(s) if i.flag[0] not in ['w', 'x']]
        return wds

    '''基于wordvector，通过lookup table的方式找到句子的wordvector的表示'''
    def rep_sentencevector(self, sentence):
        word_list = self.seg_sent(sentence)[:self.max_length]
        embedding_matrix = np.zeros((self.max_length, self.embedding_size))
        for index, wd in enumerate(word_list):
            if wd in self.embdding_dict:
                embedding_matrix[index] = self.embdding_dict.get(wd)
            else:
                continue
        len_sent = len(word_list)
        embedding_matrix = self.modify_sentencevector(embedding_matrix, len_sent)

        return embedding_matrix

    '''对于OOV词,通过左右词的词向量作平均,作为词向量表示'''
    def modify_sentencevector(self, embedding_matrix, len_sent):
        context_window = 2
        for indx, vec in enumerate(embedding_matrix):
            left = indx-context_window
            right = indx+context_window
            if left < 0:
                left = 0
            if right > len(embedding_matrix)-1:
                right = -2
            context = embedding_matrix[left:right+1]
            if vec.tolist() == [0]*300 and indx < len_sent:
                context_vector = context.mean(axis=0)
                embedding_matrix[indx] = context_vector

        return embedding_matrix

    '''对数据进行onehot映射操作'''
    def label_onehot(self, label):
        one_hot = [0]*len(self.label_dict)
        one_hot[int(label)] = 1
        return one_hot

    '''加载数据集'''
    def load_traindata(self):
        train_X = []
        train_Y = []
        count = 0
        for line in open(self.train_file):

            line = line.strip().strip().split('\t')
            if len(line) < 2:
                continue
            count += 1
            sent = line[0]
            label = line[1]
            sent_vector = self.rep_sentencevector(sent)
            label_vector = self.label_onehot(label)
            train_X.append(sent_vector)
            train_Y.append(label_vector)

            if count % 10000 == 0:
                print('loaded %s lines'%count)

        return np.array(train_X), np.array(train_Y)

    '''构造CNN网络模型'''
    def build_cnn_model(self):
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=(self.max_length, self.embedding_size)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(13, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        model.summary()
        return model

    '''构造LSTM网络'''
    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(self.max_length, self.embedding_size)))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
        model.add(Dense(13, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    '''训练CNN模型'''
    def train_cnn(self):
        X_train, Y_train, X_test, Y_test = self.split_trainset()
        model = self.build_cnn_model()
        model.fit(X_train, Y_train, batch_size=100, epochs=20, validation_data=(X_test, Y_test))
        model.save(self.cnn_modelpath)

    '''训练CNN模型'''
    def train_lstm(self):
        X_train, Y_train, X_test, Y_test = self.split_trainset()
        model = self.build_lstm_model()
        model.fit(X_train, Y_train, batch_size=100, epochs=50, validation_data=(X_test, Y_test))
        model.save(self.lstm_modelpath)

    '''划分数据集,按一定比例划分训练集和测试集'''
    def split_trainset(self):
        X, Y = self.load_traindata()
        split_rate = 0.8
        indx = int(len(X)*split_rate)
        X_train = X[:indx]
        Y_train = Y[:indx]
        X_test = X[indx:]
        Y_test = Y[indx:]
        return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    handler = QuestionClassify()
    handler.train_cnn()
    handler.train_lstm()
