#!/usr/bin/env python3
# coding: utf-8
# File: crime_qa_server.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-10

import os
import time
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pymongo

class CrimeQA:
    def __init__(self):
        self._index = "crime_data"
        self.es = Elasticsearch([{"host": "127.0.0.1", "port": 9200}])
        self.doc_type = "crime"


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

    '''问答主函数'''
    def search_main(self, question):
        candi_answers = self.search_es(question)
        for candi in candi_answers:
            print(candi)



if __name__ == "__main__":
    handler = CrimeQA()
    question = '最近买了一把枪,会犯什么罪?'
    handler.search_main(question)

