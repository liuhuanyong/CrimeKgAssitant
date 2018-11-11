#!/usr/bin/env python3
# coding: utf-8
# File: insert_es.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-10-10

import os
import time

import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pymongo

class ProcessIntoES:
    def __init__(self):
        self._index = "crime_data"
        self.es = Elasticsearch([{"host": "127.0.0.1", "port": 9200}])
        self.doc_type = "crime"
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.music_file = os.path.join(cur, 'qa_corpus.json')

    '''创建ES索引，确定分词类型'''
    def create_mapping(self):
        node_mappings = {
            "mappings": {
                self.doc_type: {    # type
                    "properties": {
                        "question": {    # field: 问题
                            "type": "text",    # lxw NOTE: cannot be string
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart",
                            "index": "true"    # The index option controls whether field values are indexed.
                        },
                        "answers": {  # field: 问题
                            "type": "text",  # lxw NOTE: cannot be string
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart",
                            "index": "true"  # The index option controls whether field values are indexed.
                        },
                    }
                }
            }
        }
        if not self.es.indices.exists(index=self._index):
            self.es.indices.create(index=self._index, body=node_mappings)
            print("Create {} mapping successfully.".format(self._index))
        else:
            print("index({}) already exists.".format(self._index))

    '''批量插入数据'''
    def insert_data_bulk(self, action_list):
        success, _ = bulk(self.es, action_list, index=self._index, raise_on_error=True)
        print("Performed {0} actions. _: {1}".format(success, _))


'''初始化ES，将数据插入到ES数据库当中'''
def init_ES():
    pie = ProcessIntoES()
    # 创建ES的index
    pie.create_mapping()
    start_time = time.time()
    index = 0
    count = 0
    action_list = []
    BULK_COUNT = 1000  # 每BULK_COUNT个句子一起插入到ES中

    for line in open(pie.music_file):
        if not line:
            continue
        item = json.loads(line)
        index += 1
        action = {
            "_index": pie._index,
            "_type": pie.doc_type,
            "_source": {
                "question": item['question'],
                "answers": '\n'.join(item['answers']),
            }
        }
        action_list.append(action)
        if index > BULK_COUNT:
            pie.insert_data_bulk(action_list=action_list)
            index = 0
            count += 1
            print(count)
            action_list = []
        end_time = time.time()

        print("Time Cost:{0}".format(end_time - start_time))


if __name__ == "__main__":
    # 将数据库插入到elasticsearch当中
    # init_ES()
    # 按照标题进行查询
    question = '我老公要起诉离婚 我不想离婚怎么办'

