#!/usr/bin/env python3
# coding: utf-8
# File: crime_qa_server.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-10

import os
import time
import json
from elasticsearch import Elasticsearch
import numpy as np
import jieba.posseg as pseg

class CrimeQA:
    def __init__(self):
        self._index = "crime_data"
        self.es = Elasticsearch([{"host": "127.0.0.1", "port": 9200}])
        self.doc_type = "crime"
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.embedding_path = os.path.join(cur, 'embedding/word_vec_300.bin')
        self.embdding_dict = self.load_embedding(self.embedding_path)
        self.embedding_size = 300
        self.min_score = 0.4
        self.min_sim = 0.8

    '''根据question进行事件的匹配查询'''
    def search_specific(self, value, key="question"):
        query_body = {
            "query": {
                "match": {
                    key: value,
                }
            }
        }
        searched = self.es.search(index=self._index, doc_type=self.doc_type, body=query_body, size=20)
        # 输出查询到的结果
        return searched["hits"]["hits"]

    '''基于ES的问题查询'''
    def search_es(self, question):
        answers = []
        res = self.search_specific(question)
        for hit in res:
            answer_dict = {}
            answer_dict['score'] = hit['_score']
            answer_dict['sim_question'] = hit['_source']['question']
            answer_dict['answers'] = hit['_source']['answers'].split('\n')
            answers.append(answer_dict)
        return answers


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


    '''计算问句与库中问句的相似度,对候选结果加以二次筛选'''
    def similarity_cosine(self, vector1, vector2):
        cos1 = np.sum(vector1*vector2)
        cos21 = np.sqrt(sum(vector1**2))
        cos22 = np.sqrt(sum(vector2**2))
        similarity = cos1/float(cos21*cos22)
        if similarity == 'nan':
            return 0
        else:
            return  similarity

    '''问答主函数'''
    def search_main(self, question):
        candi_answers = self.search_es(question)
        question_vector = self.rep_sentencevector(question,flag='noseg')
        answer_dict = {}
        for indx, candi in enumerate(candi_answers):
            candi_question = candi['sim_question']
            score = candi['score']/100
            candi_vector = self.rep_sentencevector(candi_question, flag='noseg')
            sim = self.similarity_cosine(question_vector, candi_vector)
            if sim < self.min_sim:
                continue
            final_score = (score + sim)/2
            if final_score < self.min_score:
                continue
            answer_dict[indx] = final_score
        if answer_dict:
            answer_dict = sorted(answer_dict.items(), key=lambda asd:asd[1], reverse=True)
            final_answer = candi_answers[answer_dict[0][0]]['answers']
        else:
            final_answer = '您好,对于此类问题,您可以咨询公安部门'
        #
        # for i in answer_dict:
        #     answer_indx = i[0]
        #     score = i[1]
        #     print(i, score, candi_answers[answer_indx])
        #     print('******'*6)
        return final_answer


if __name__ == "__main__":
    handler = CrimeQA()
    while(1):
        question = input('question:')
        final_answer = handler.search_main(question)
        print('answers:', final_answer)

