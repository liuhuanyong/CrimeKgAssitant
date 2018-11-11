#!/usr/bin/env python3
# coding: utf-8
# File: crime_classify.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-10

import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
import jieba.posseg as pseg
from collections import Counter
from sklearn.externals import joblib

class CrimeClassify(object):
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        crime_file = os.path.join(cur, 'crime.txt')
        self.label_dict = self.build_crime_dict(crime_file)
        self.id_dict = {j:i for i,j in self.label_dict.items()}
        self.train_file = os.path.join(cur, 'crime_train_all.txt')
        self.embedding_path = os.path.join(cur, 'embedding/word_vec_300.bin')
        self.embdding_dict = self.load_embedding(self.embedding_path)
        self.embedding_size = 300
        self.model_path = 'crime_predict_svm_all.model'
        return

    '''构建罪名词类型'''
    def build_crime_dict(self, crimefile):
        label_dict = {}
        i = 0
        for line in open(crimefile):
            crime = line.strip()
            if not crime:
                continue
            label_dict[crime] = i
            i +=1
        return label_dict

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
        wds = [i.word for i in pseg.cut(s) if i.flag[0] not in ['x', 'u', 'c', 'p', 'm', 't']]
        return wds

    '''基于wordvector，通过lookup table的方式找到句子的wordvector的表示'''
    def rep_sentencevector(self, sentence, flag='seg'):
        if flag == 'seg':
            word_list = [i for i in sentence.split(' ') if i]
        else:
            word_list = self.seg_sent(sentence)
        embedding = np.zeros(self.embedding_size)
        sent_len = 0
        for index, wd in enumerate(word_list):
            if wd in self.embdding_dict:
                embedding += self.embdding_dict.get(wd)
                sent_len += 1
            else:
                continue
        return embedding/sent_len

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
            # if count > 1000:
            #     break
            sent = line[1]
            label_id = int(line[0])
            sent_vector = self.rep_sentencevector(sent, flag='seg')
            train_X.append(sent_vector)
            train_Y.append(label_id)
            if count % 10000 == 0:
                print('loaded %s lines'%count)
        return np.array(train_X), np.array(train_Y)

    '''使用SVM进行分类'''
    def train_classifer(self):
        x_train, y_train = self.load_traindata()
        model = LinearSVC()
        model.fit(x_train, y_train)
        joblib.dump(model, self.model_path)
        y_predict = model.predict(x_train)
        all = len(y_predict)
        right = 0
        for i in range(len(y_train)):
            y = y_train[i]
            y_pred = y_predict[i]
            if y_pred == y:
                right += 1
        print('precision:%s/%s=%s'%(right, all, right/all))

    '''使用svm模型进行预测'''
    def predict(self, sent):
        model = joblib.load(self.model_path)
        represent_sent = self.rep_sentencevector(sent, flag='noseg')
        text_vector = np.array(represent_sent).reshape(1, -1)
        res = model.predict(text_vector)[0]
        label = self.id_dict.get(res)
        return label


    '''检查测试合准确率'''
    def check_precision(self):
        model = joblib.load(self.model_path)
        x_train, y_train = self.load_traindata()
        y_predict = model.predict(x_train)
        all = len(y_predict)
        right = 0
        for i in range(len(y_train)):
            y = y_train[i]
            y_pred = y_predict[i]
            if y_pred == y:
                right += 1
        print('precision:%s/%s=%s'%(right, all, right/all))
        # precision:170231 / 204231 = 0.83352184536138
        # precision:2650780 / 2880306 = 0.9203119390786951


def test():
    handler = CrimeClassify()
    # handler.train_classifer()
    while(1):
        sent = input('enter an sent to search:')
        label = handler.predict(sent)
        print(label)

def build_data():
    label_dict = {}
    i = 0
    for line in open('crime.txt'):
        crime = line.strip()
        if not crime:
            continue
        label_dict[crime] = i
        i += 1

    f = open('crime_train_all.txt', 'w+')
    count = 0
    for line in open('accu_train.txt'):
        line = line.strip().split('###')
        if len(line) < 3:
            continue
        crime = line[1].split(';')[0]
        sent = line[-1]
        label = label_dict.get(crime)
        f.write(str(label) + '\t' + sent + '\n')
        count += 1
        print(count)
    f.close()


if __name__ == '__main__':
    test()
    #build_data()
    #handler = CrimeClassify()
    #handler.check_precision()
