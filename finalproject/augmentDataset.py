#!/usr/bin/env python
# coding: utf-8

###########################################
# module for generate data
###########################################

# variables
DATA_DIR = '/content/drive/MyDrive/AI599/data'

# get datasets
import datasets

dataset_card = "lbox/lbox_open"
task = "precedent_corpus"
data = datasets.load_dataset(dataset_card, task)

# ## filtering lbox_open data
# 2019.6.25 이후에 발생한 음주운전 데이터를 추출합니다.

import pandas as pd
df = pd.DataFrame( (v for v in data['train']) )
# df = pd.Series( (v['precedent'] for v in data['train']) )

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
           & (df['precedent'].str.len() < 2500)
           ]

print(len(avail_df))


# In[12]:


# 양형의 사유가 있는 것과 없는 것 분리
yesy = avail_df[avail_df['precedent'].str.contains('양형의 이유')]
noy = avail_df[~avail_df['precedent'].str.contains('양형의 이유')]
yesy


# In[13]:


# 주문
# 피고인을 징역 1년 4개월에 처한다.
# 다만, 이 판결 확정일부터 3년간 위 형의 집행을 유예한다.
# 피고인에게 3년간 보호관찰을 받을 것과 80시간의 사회봉사 및 40시간의 준법운전강의 수강을 명한다.

# 이유
# 범죄사실
# 1. 특정범죄가중처벌등에관한법률위반(위험운전치상)
# 피고인은 B 말리부 승용차의 운전업무에 종사하는 사람이다.
# 피고인은 2020. 11. 7. 07:00경 자동차운전면허 없이 혈중알코올농도 0.119%의 술에 취한 상태로 고양시 일산동구 백석동 백마주유소 사거리를 대곡역 방향에서 백석역 방향으로 진행하게 되었다. 그곳은 신호등이 설치된 교차로이므로, 이러한 경우 자동차운전업무에 종사하는 사람에게는 전방좌우를 잘 살피고 신호등이 표시하는 신호를 따라 안전운전을 하여 사고를 미리 방지하여야 할 업무상 주의의무가 있다.
# 피고인은 위와 같이 음주의 영향으로 정상적인 운전이 곤란한 상태에서 이러한 주의의무를 게을리 한 채 신호등의 정지신호를 위반하여 그대로 직진한 과실로, 마침 진행방향 좌측에서 우측으로 신호등의 직진신호에 따라 진행 중이던 피해자 C 운전의 D 렉스턴 승용차의 우측 부분을 위 말리부 승용차의 앞부분으로 들이받았다.
# 결국 피고인은 음주의 영향으로 정상적인 운전이 곤란한 상태에서 자동차를 운전하여 피해자에게 약 2주간의 치료를 필요로 하는 어깨 관절의 염좌 및 긴장 등의 상해를 입게 하였다.
# 2. 도로교통법위반(음주운전), 도로교통법위반(무면허운전)
# 피고인은 2016. 7. 28. 의정부지방법원 고양지원에서 도로교통법위반(음주운전)죄로 벌금 300만 원을 받았다.
# 피고인은 제1항 기재 일시, 서울 용산구 E에 있는 F 앞 도로부터 위 사고지점까지 약 26km 구간에서 자동차운전면허를 받지 아니하고 혈중알코올농도 0.119%의 술에 취한 상태로 위 말리부 승용차를 운전하였다.
# 이로써 피고인은 도로교통법 제44조 제1항(술에 취한 상태에서의 운전 금지)을 2회 이상 위반함과 동시에 자동차운전면허를 받지 아니하고 자동차를 운전하였다.
# 증거의 요지
# 1. 피고인의 법정진술
# 1. C의 진술서
# 1. 교통사고보고(실황조사서), 사고현장사진
# 1. 주취운전자정황진술보고서, 수사보고(주취운전자정황보고)
# 1. 음주운전단속결과통보
# 1. 진단서
# 1. 자동차운전면허대장
# 1. 동영상CD
# 1. 판시 전과 : 범죄경력 등 조회회보서, 수사보고(피의자 동종처벌전력보고), 약식명령문
# 법령의 적용
# 1. 범죄사실에 대한 해당법조
# 특정범죄 가중처벌 등에 관한 법률 제5조의11 제1항 전단(위험운전치상), 도로교통법 제152조 제1호, 제43조(무면허운전), 구 도로교통법(2020. 6. 9. 법률 제17371호로 개정되기 전의 것) 제148조의2 제1항, 제44조 제1항(음주운전)
# 1. 상상적 경합
# 형법 제40조, 제50조
# 1. 형의 선택
# 각 징역형 선택
# 1. 경합범가중
# 형법 제37조 전단, 제38조 제1항 제2호, 제50조
# 1. 작량감경
# 형법 제53조, 제55조 제1항 제3호
# 1. 집행유예
# 형법 제62조 제1항
# 1. 보호관찰, 사회봉사명령, 수강명령
# 형법 제62조의2
# 양형의 이유
# 1. 양형기준에 따른 권고형의 범위
# [유형의 결정] 교통범죄 > 02. 위험운전 교통사고 > [제1유형] 위험운전 치상
# [특별 양형인자] 감경요소: 처벌불원
# 가중요소: 교통사고처리 특례법 제3조 제2항 단서(8호, 제외) 중 위법성이 중한 경우 또는 난폭운전의 경우
# [권고영역 및 권고형의 범위] 기본영역, 징역 1년 ~ 2년 6개월(법률상 처단형의 하한에 따름)
# 2. 선고형의 결정
# - 유리한 정상 : 처벌불원, 중하지 아니한 상해, 진지한 반성, 판시 전과 외 다른 형사처벌 전력 없음, 자동차종합보험 가입
# - 불리한 정상 : 교통사고처리 특례법 제3조 제2항 단서 중 위법성이 중한 경우(음주, 무면허, 신호위반), 피해자에게 교통사고 발생 또는 피해 확대에 아무런 과실이 없음
# - 그 밖에 피고인의 나이, 성행, 환경, 범행에 이르게 된 동기와 경위, 혈중알코올농도, 범행 전후의 정황 등 변론에 나타난 여러 정상을 종합하여 양형기준상 권고형의 범위 내에서 형을 정하되, 그 형의 집행을 유예한다.


# In[14]:


# 주문
# 피고인을 벌금 6,000,000원에 처한다.
# 피고인이 위 벌금을 납입하지 아니하는 경우 100,000원을 1일로 환산한 기간 피고인을 노역장에 유치한다.
# 위 벌금에 상당한 금액의 가납을 명한다.

# 이유
# 범 죄 사 실
# 1. 도로교통법위반(음주운전)
# 피고인은 2020. 11. 1. 18:45경 충주시 B 부근 도로에서부터 충주시 C에 있는, D 앞 도로에 이르기까지 약 1km 구간에서 혈중알콜농도 0.094%의 술에 취한 상태로 ESM5 승용차를 운전하였다.
# 2. 교통사고처리특례법위반(치상)
# 피고인은 E SM5 승용차의 운전업무에 종사하는 사람이다.
# 피고인은 2020. 11. 1. 18:45경 위 차량을 운전하여 충주시 C에 있는, D 앞 교차로를 봉방동 행정복지센터 방면에서 F연립 방면으로 진행하게 되었다.
# 그곳은 신호등이 설치되어 있지 않은 교차로였으므로 이러한 경우 자동차의 운전업무에 종사하는 사람에게는 술에 취하지 않은 상태에서 전방좌우를 잘 살피는 한편, 위와 같이 교통정리를 하고 있지 아니하는 교차로에 동시에 들어가려고 할 경우 우측도로의 차에 진로를 양보하는 등 안전하게 운전하여 사고를 미리 방지하여야 할 업무상 주의의무가 있었다.
# 그럼에도 불구하고 피고인은 이를 게을리 한 채 제1항 기재와 같이 술에 취하여 주변을 제대로 살피지 않고 그대로 교차로에 진입한 과실로 마침 피고인의 진행방면 우측의 성광교회 방면에서 D 방면으로 진행 중이던 피해자 G(남, 19세)가 운전하는 H 모닝 승용차 앞 범퍼를 피고인의 차량 우측 뒤 범퍼 부분으로 충돌하였다.
# 결국 피고인은 위와 같은 업무상 과실로 피해자에게 약 2주간의 치료가 필요한 목뼈의 염좌 및 긴장의 상해를 입게 하였다.
# 증거의 요지
# 1. 피고인의 법정진술
# 1. G의 진술서(교통사고발생상황)
# 1. 수사보고(주취운전자 A의 음주운전 출발지점 및 단속지점에 대하여), 수사보고(주취운전자 정황보고), 내사보고(본 건 교통사고 상황), 수사보고(피해자 인적피해 확인)
# 1. 교통사고 발생상황보고, 사고현장사진, 사고메모, 주취운전자 정황진술보고서, 음주운전 단속결과통보, 실황조사서
# 법령의 적용
# 1. 범죄사실에 대한 해당법조 및 형의 선택
# 도로교통법 제148조의2 제3항 제2호, 제44조 제1항(음주운전의 점), 교통사고처리특례법 제3조 제1항, 제2항 단서 제8호, 형법 제268조(업무상과실치상의 점), 각 벌금형 선택
# 1. 경합범가중
# 형법 제37조 전단, 제38조 제1항 제2호, 제50조
# 1. 노역장유치
# 형법 제70조 제1항, 제69조 제2항
# 1. 가납명령
# 형사소송법 제334조 제1항
# 양형의 이유
# 아래의 정상 및 피고인의 나이, 성행, 가정환경, 범행의 동기, 수단과 결과, 범행 후의 정황 등 이 사건 기록과 변론에 나타난 여러 양형조건들을 참작하여 주문과 같이 형을 정한다.
# ○ 불리한 정상 : 피고인은 술에 취한 상태로 운전하다가 교통사고를 일으켜 피해자에게 상해를 입혔는바 죄질이 좋지 않다. 피고인은 1997년 교통사고처리특례법위반죄로 벌금형 1회, 2001년 도로교통법위반(음주운전)죄로 벌금형 1회 처벌받은 전력이 있음에도 또 다시 이 사건 동종 범행을 저질렀다.
# ○ 유리한 정상 : 피고인은 범행을 시인하고 있다. 다행히 이 사건 업무상과실치상 범행의 피해자가 입은 상해의 정도가 비교적 중하지 않다. 피고인은 피해자의 피해를 회복하였고, 피해자와 원만히 합의하여 피해자가 피고인의 처벌을 원하지 않고 있다.


# In[15]:


yesy.precedent.loc[229]


# In[16]:


# !pip install --quiet pytorch>=1.10.0
# !pip install --quiet transformers==4.16.2
# !pip install --quiet pytorch-lightning==1.5.10
# !pip install --quiet streamlit==1.2.0


# In[17]:


# import torch
# from transformers import PreTrainedTokenizerFast
# from transformers import BartForConditionalGeneration

# tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
# model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

# text = """
# 1일 오후 9시까지 최소 20만3220명이 코로나19에 신규 확진됐다. 또다시 동시간대 최다 기록으로, 사상 처음 20만명대에 진입했다.
# 방역 당국과 서울시 등 각 지방자치단체에 따르면 이날 0시부터 오후 9시까지 전국 신규 확진자는 총 20만3220명으로 집계됐다.
# 국내 신규 확진자 수가 20만명대를 넘어선 것은 이번이 처음이다.
# 동시간대 최다 기록은 지난 23일 오후 9시 기준 16만1389명이었는데, 이를 무려 4만1831명이나 웃돌았다. 전날 같은 시간 기록한 13만3481명보다도 6만9739명 많다.
# 확진자 폭증은 3시간 전인 오후 6시 집계에서도 예견됐다.
# 오후 6시까지 최소 17만8603명이 신규 확진돼 동시간대 최다 기록(24일 13만8419명)을 갈아치운 데 이어 이미 직전 0시 기준 역대 최다 기록도 넘어섰다. 역대 최다 기록은 지난 23일 0시 기준 17만1451명이었다.
# 17개 지자체별로 보면 서울 4만6938명, 경기 6만7322명, 인천 1만985명 등 수도권이 12만5245명으로 전체의 61.6%를 차지했다. 서울과 경기는 모두 동시간대 기준 최다로, 처음으로 각각 4만명과 6만명을 넘어섰다.
# 비수도권에서는 7만7975명(38.3%)이 발생했다. 제주를 제외한 나머지 지역에서 모두 동시간대 최다를 새로 썼다.
# 부산 1만890명, 경남 9909명, 대구 6900명, 경북 6977명, 충남 5900명, 대전 5292명, 전북 5150명, 울산 5141명, 광주 5130명, 전남 4996명, 강원 4932명, 충북 3845명, 제주 1513명, 세종 1400명이다.
# 집계를 마감하는 자정까지 시간이 남아있는 만큼 2일 0시 기준으로 발표될 신규 확진자 수는 이보다 더 늘어날 수 있다. 이에 따라 최종 집계되는 확진자 수는 21만명 안팎을 기록할 수 있을 전망이다.
# 한편 전날 하루 선별진료소에서 이뤄진 검사는 70만8763건으로 검사 양성률은 40.5%다. 양성률이 40%를 넘은 것은 이번이 처음이다. 확산세가 계속 거세질 수 있다는 얘기다.
# 이날 0시 기준 신규 확진자는 13만8993명이었다. 이틀 연속 13만명대를 이어갔다.
# """

# text = text.replace('\n', ' ')

# raw_input_ids = tokenizer.encode(text)
# input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

# summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
# result = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

# result
# # '1일 0 9시까지 최소 20만3220명이 코로나19에 신규 확진되어 역대 최다 기록을 갈아치웠다.'


# In[18]:


# Precedent 쪼개기 위한 function set
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

import re

class Precedent:
    def __init__(self, rawPrecedent, tokenizer, model):
        self.raw = rawPrecedent
        self.tokenizer = tokenizer
        self.model = model

        # text = """
        # 1일 오후 9시까지 최소 20만3220명이 코로나19에 신규 확진됐다. 또다시 동시간대 최다 기록으로, 사상 처음 20만명대에 진입했다.
        # 방역 당국과 서울시 등 각 지방자치단체에 따르면 이날 0시부터 오후 9시까지 전국 신규 확진자는 총 20만3220명으로 집계됐다.
        # 국내 신규 확진자 수가 20만명대를 넘어선 것은 이번이 처음이다.
        # 동시간대 최다 기록은 지난 23일 오후 9시 기준 16만1389명이었는데, 이를 무려 4만1831명이나 웃돌았다. 전날 같은 시간 기록한 13만3481명보다도 6만9739명 많다.
        # 확진자 폭증은 3시간 전인 오후 6시 집계에서도 예견됐다.
        # 오후 6시까지 최소 17만8603명이 신규 확진돼 동시간대 최다 기록(24일 13만8419명)을 갈아치운 데 이어 이미 직전 0시 기준 역대 최다 기록도 넘어섰다. 역대 최다 기록은 지난 23일 0시 기준 17만1451명이었다.
        # 17개 지자체별로 보면 서울 4만6938명, 경기 6만7322명, 인천 1만985명 등 수도권이 12만5245명으로 전체의 61.6%를 차지했다. 서울과 경기는 모두 동시간대 기준 최다로, 처음으로 각각 4만명과 6만명을 넘어섰다.
        # 비수도권에서는 7만7975명(38.3%)이 발생했다. 제주를 제외한 나머지 지역에서 모두 동시간대 최다를 새로 썼다.
        # 부산 1만890명, 경남 9909명, 대구 6900명, 경북 6977명, 충남 5900명, 대전 5292명, 전북 5150명, 울산 5141명, 광주 5130명, 전남 4996명, 강원 4932명, 충북 3845명, 제주 1513명, 세종 1400명이다.
        # 집계를 마감하는 자정까지 시간이 남아있는 만큼 2일 0시 기준으로 발표될 신규 확진자 수는 이보다 더 늘어날 수 있다. 이에 따라 최종 집계되는 확진자 수는 21만명 안팎을 기록할 수 있을 전망이다.
        # 한편 전날 하루 선별진료소에서 이뤄진 검사는 70만8763건으로 검사 양성률은 40.5%다. 양성률이 40%를 넘은 것은 이번이 처음이다. 확산세가 계속 거세질 수 있다는 얘기다.
        # 이날 0시 기준 신규 확진자는 13만8993명이었다. 이틀 연속 13만명대를 이어갔다.
        # """

        # text = text.replace('\n', ' ')

        # raw_input_ids = self.tokenizer.encode(text)
        # input_ids = [self.tokenizer.bos_token_id] + raw_input_ids + [self.tokenizer.eos_token_id]

        # summary_ids = self.model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
        # result = self.tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

        # result

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
        
        # print(f'before facts:\n {self.facts}')
        # raw_input_ids = self.tokenizer.encode(self.facts)
        # input_ids = [self.tokenizer.bos_token_id] + raw_input_ids + [self.tokenizer.eos_token_id]
        # summary_ids = self.model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
        # self.facts = self.tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        # print(f'after facts:\n {self.facts}')

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


# In[19]:


# 양형의 사유가 없는 집합에서 "벌"-"나", "징"-"나" 데이터셋 생성. 이를 a, b라고 칭함
dataset = {}
dataset['all'] = generateDataSet(avail_df, 2000, TARGET_OPT)
dataset['avail'] = generateDataSet(yesy, 2000, TARGET_OPT)
dataset['nonavail'] = generateDataSet(noy, 2000, TARGET_OPT)


# In[20]:


dataset['avail']


# In[21]:


dataset['all'][TARGET].value_counts()


# In[22]:


dataset['avail'][TARGET].value_counts()


# In[23]:


dataset['nonavail'][TARGET].value_counts()


# In[24]:


# from imblearn.over_sampling import SMOTENC
# from sklearn.preprocessing import MultiLabelBinarizer

# # for reproducibility purposes
# seed = 100
# # SMOTE number of neighbors
# k = 1

# df = dataset['all']


# In[25]:


# import torch
# from imblearn.over_sampling import RandomOverSampler

# from sklearn.model_selection import train_test_split
# import datasets
# import tensorflow as tf
# from datasets import Dataset

# import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# from functools import reduce
# import random

# import torch
# import numpy as np
# import transformers
# import pytorch_lightning as pl
# from transformers import AutoTokenizer, BertForSequenceClassification

# import pandas as pd

# def convertDFtoDS(df):
#     return tf.data.Dataset.from_tensor_slices(dict(df))

# backbone_card = 'bert-base-multilingual-cased'
# tokenizer = AutoTokenizer.from_pretrained(backbone_card)


# def resampling_data(df):
#     inputs = tokenizer(df[TARGET].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)
#     input_ids = inputs.input_ids
#     attention_mask = inputs.attention_mask
#     # (# of samples, # of features) 크기의 array, dataframe, sparse matrix 모두 가능 
#     # 여기선 그냥 array 
#     x = [[input_id, mask] for input_id, mask in zip(input_ids, attention_mask)]
#     y = df[TARGET].tolist()
#     return x, y

# ros = RandomOverSampler(random_state=42)

# x, y = resampling_data(df)
# x_ros, y_ros = ros.fit_resample(x, y)

# print(x)


# # class ResampledDataset(torch.utils.data.Dataset): 
# #     def __init__(self, x_rus, y_rus):
# #         self.input_ids = []
# #         self.attention_mask = []
# #         for input_id, mask in x_rus:
# #             self.input_ids.append(input_id)
# #             self.attention_mask.append(mask)
# #         self.labels = y_rus

# #     def __len__(self):
# #         return len(self.input_ids)

# #     def __getitem__(self, idx):
# #         return {'input_ids': self.input_ids[idx] ,'attention_mask': self.attention_mask[idx], 'labels':self.labels[idx]}
        
# # # huggingface Trainer에 전달할 수 있는 Dataset 
# # train_data = ResampledDataset(x_ros, y_ros)
   
# # ...

# # trainer = Trainer(
# #    model=model,            
# #    args=training_args,            
# #    compute_metrics=compute_metrics, 
# #    train_dataset=train_data,       
# #    eval_dataset=eval_data      
# #    )


# In[26]:


# # make a new df made of all the columns, except the target class
# X = df.loc[:, df.columns != TARGET]
# y = df[TARGET]
# #sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
# sm = SMOTENC(categorical_features=[0,], random_state=seed)
# X_res, y_res = sm.fit_resample(X, y)


# In[27]:


# sm


# In[28]:


# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state = 42)
# X_train_oversampled, y_train_oversampled = sm.fit_sample(X_train, y_train)
# X_train = pd.DataFrame(X_train_oversampled, columns=X_train.columns)


# In[29]:


# # hs's rule
# [1,2,3,4,5]
# [6,10]
# []
# []
# []

# # yk's rule
# [1 2 3]
# [4 5 6]
# [7 8 9] 8
# [10 11 12]
# [13 14 15] 14
# [16 17 18]

# # 1~5 / 6~8 / 9~12 / 13~15 / 16~18 / 19~22 / 24~29 / 30~36 /


# ## 전처리된 데이터로 학습.
# 모델이 학습할 수 있도록 데이터의 형태를 변경해 학습합니다.

# In[30]:


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from functools import reduce
import random

import torch
import numpy as np
import transformers
import pytorch_lightning as pl
from transformers import AutoTokenizer, BertForSequenceClassification

import pandas as pd


# In[31]:


# https://data-newbie.tistory.com/721
from torch.utils.data import Sampler
class OverSampler(Sampler):
    """Over Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector
        self.indices = list(range(len(self.class_vector)))
        self.batch_size =batch_size
        uni_label = torch.unique(class_vector) 
        uni_label_sorted, _  = torch.sort(uni_label)
        print(uni_label_sorted)
        uni_label_sorted = uni_label_sorted.detach().numpy()
        label_bin = torch.bincount(class_vector.int()).detach().numpy()
        label_to_count = dict(zip(uni_label_sorted , label_bin))
        weights = [ len(class_vector) / label_to_count[float(label)] for label in           class_vector]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.batch_size, replacement=True))

    def __len__(self):
        return len(self.class_vector)


# In[32]:



# In[33]:


dataset['avail']


# In[34]:


ccnt = dataset[TARGET_SET][TARGET].value_counts()

class_counts = dataset[TARGET_SET][TARGET].value_counts().to_list() #43200, 4800
num_samples = sum(class_counts) # 48000 - 전체 데이터 갯수
labels = dataset[TARGET_SET][TARGET].to_list()

ccnt_map = {}
for index, row in dataset[TARGET_SET][TARGET].value_counts().iteritems():
    ccnt_map[index] = (float)(num_samples/row)

weights = []
for l in labels:
    weights.append(ccnt_map[l])

len(weights)
