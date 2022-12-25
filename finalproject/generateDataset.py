#!/usr/bin/env python
# coding: utf-8

###########################################
# module for generate data                #
###########################################

# variables
TARGET = 'prison'
TARGET_SET = 'avail' # all, avail, nonavail
TARGET_START = 1 # 0 if you want zero values, INCLUDE THST VALUE
TARGET_END = 36 # INCLUDE THIS VALUE
TARGET_APPROX = 0 # '2'개월 또는 '1000000'원
TARGET_OPT = 'binary' # class, binary
TRAIN_COLUMN = 'fy' # raw, facts, statutes, yh, fy
DATA_DIR = '/content/drive/MyDrive/AI599/data'

###########################################
# function set                            #
###########################################
# Precedent 쪼개기 위한 function set
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import re
import pandas as pd
import datasets

class Precedent:
    def __init__(self, rawPrecedent, tokenizer, model):
        self.raw = rawPrecedent
        self.tokenizer = tokenizer
        self.model = model
        self.__parse__()
        
    def __parse__(self):
        # 주문 - 범죄사실 - 법령의적용 - 양형의이유
        # ms - facts - statuts - yh
        # fy: facts + yh
        self.raw = self.raw.replace('\n', ' ')
        self.etc = self.raw

        tag = '범 죄 사 실'
        l = self.etc.split(tag)
        self.ms = ''
        if 1 < len(l):
            self.ms = l[0].strip()
            self.etc = self.etc.replace(self.ms, '')
            self.etc = self.etc.replace(tag, '')
            self.etc = self.etc.strip()

        tag = '법령의 적용'
        self.facts = ''
        l = self.etc.split(tag)
        if 1 < len(l):
            self.facts = l[0].strip()
            self.etc = self.etc.replace(self.facts, '')
            self.etc = self.etc.replace(tag, '')
            self.etc = self.etc.strip()

        tag = '양형의 이유'
        self.statutes = ''
        l = self.etc.split(tag)
        if 1 < len(l):
            self.statutes = l[0].strip()
            self.etc = self.etc.replace(self.statutes, '')
            self.etc = self.etc.replace(tag, '')
            self.etc = self.etc.strip()

        self.yh = self.etc
        self.etc = self.raw

        self.fy = f'{self.facts} {self.yh}'

def getNumericInKorean(kStrNum):
    retval = 1
    retStr = kStrNum
    retStr = retStr.replace(',', '')
    if '만' in retStr:
        retval = retval * 10000
        retStr = retStr.replace('만', '')
    if '천' in retStr:
        retval = retval * 1000
        retStr = retStr.replace('천', '')
    retval = int(retStr) * retval
    return retval

def getMonthFromText(s):
    year  = re.findall(r'(\d+)년', s, re.S)
    y = 0 if len(year) == 0 else (int)(year[0]);
    month  = re.findall(r'(\d+)개?월', s, re.S)
    m = 0 if len(month) == 0 else (int)(month[0]);
    return y*12 + m

def generateDataSet(dataframe, max_cnt, classopt='class'):
    retval = pd.DataFrame({\
                           'id': pd.Series(dtype='int'), 'money': pd.Series(dtype='int'), 'prison': pd.Series(dtype='int'),'yooye': pd.Series(dtype='int'),\
                           'facts':[], 'statutes':[], 'yh':[], 'fy':[], 'raw':[]})
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
    
    cnts = {}
    for i, row in dataframe.iterrows():
        p = Precedent(row['precedent'], tokenizer, model)
        #print(f'p.ms: {p.ms}, p.fy: {p.fy}')
        # if '처한다' in p.ms and p.ms.startswith('피') and '및' not in p.ms:
        info = {}
        info['money'] = 0
        info['prison'] = 0
        info['yooye'] = 0
        # 주문\n피고인을 벌금 14,000,000(일천사백만)원에 처한다.\
        money  = re.findall(r'벌금\s+([^\s\(\)]+)\s*원', p.ms, re.S)
        if 0 < len(money): # 벌금형
            if classopt == 'class':
                info['money'] = '0' if len(money) == 0 else getNumericInKorean(money[0])
            else:
                info['money'] = '0' if len(money) == 0 else 1
        else: # 금고형, 집행유예 포함
            if '유예' in p.ms: # 집행유예 포함된 금고형.
                msList = p.ms.split('처한다')
                info['yooye'] = getMonthFromText(msList[1]) if classopt == 'class' else 1
                info['prison'] = getMonthFromText(msList[0]) if classopt == 'class' else 1
                # yy = 1
                #prison = 1
            else: # 그냥 금고형
                info['prison'] = getMonthFromText(p.ms) if classopt == 'class' else 1
                #prison = 1

        if TARGET_START <= info[TARGET] and info[TARGET] <= TARGET_END and info[TARGET] not in []:
        #    if prison in [0, 12, 18]:
            if info[TARGET] not in cnts:
                cnts[info[TARGET]] = 0
            cnts[info[TARGET]] = cnts[info[TARGET]] + 1
            if cnts[info[TARGET]] > max_cnt:
                continue
            #retval.loc[i] = [i, [info['money']], [info['prison']], [info['yooye']], p.facts, p.statutes, p.yh, p.fy, p.raw]
            retval.loc[i] = [i, info['money'], info['prison'], info['yooye'], p.facts, p.statutes, p.yh, p.fy, p.raw]
        
        #retval.loc[i] = [i, [mn], [prison], [yy], p.etc, p.raw]  
    return retval


###########################################
# main                                    #
###########################################

# get datasets
dataset_card = "lbox/lbox_open"
task = "precedent_corpus"
data = datasets.load_dataset(dataset_card, task)

# ## filtering lbox_open data
# 2019.6.25 이후에 발생한 음주운전 데이터를 추출합니다.
df = pd.DataFrame( (v for v in data['train']) )

# 주문과 형량이 나타나는 사용 가능한 데이터 우선 추출
avail_df = df[ \
           df['precedent'].str.contains('법령의 적용') \
           & df['precedent'].str.contains('피고인을') \
           & df['precedent'].str.contains('처한다.') \
           & df['precedent'].str.contains('2020|2021') # 2019.6.25 개정 이후 사건만 다룸
           & df['precedent'].str.contains('도로교통법') \
           & df['precedent'].str.contains('제44조 제1항') \
           & df['precedent'].str.contains('주문') \
           & df['precedent'].str.contains('범 죄 사 실') \
           & df['precedent'].str.contains('법령의 적용') \
           #& (df['precedent'].str.len() < 2500)
           ]

# print(len(avail_df))

# 양형의 사유가 있는 것과 없는 것 분리
yesy = avail_df[avail_df['precedent'].str.contains('양형의 이유')]
noy = avail_df[~avail_df['precedent'].str.contains('양형의 이유')]


# 양형의 사유가 없는 집합에서 "벌"-"나", "징"-"나" 데이터셋 생성. 이를 a, b라고 칭함
dataset = {}
dataset['all'] = generateDataSet(avail_df, 2000, TARGET_OPT)
dataset['avail'] = generateDataSet(yesy, 2000, TARGET_OPT)
dataset['nonavail'] = generateDataSet(noy, 2000, TARGET_OPT)

# 데이터셋 저장
print(dataset['all'])