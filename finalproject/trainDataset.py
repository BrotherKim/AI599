#!/usr/bin/env python
# coding: utf-8

###########################################
# module for augmentate data
###########################################

# variables
TARGET = 'prison'
TARGET_APPROX = 0 # '2'개월 또는 '1000000'원
TARGET_COUNT = 10000 # 한 컬럼 최대 갯수
TARGET_OPT = 'class' # class, binary
TARGET_SET = f'{TARGET}_{TARGET_OPT}_avail' # all, avail, nonavail

TRAIN_COLUMN = 'fy' # raw, facts, statutes, yh, fy
DATA_DIR = '/content/drive/MyDrive/AI599/data'
MODEL_DIR = '/content/drive/MyDrive/AI599/model'

###########################################
# main                                    #
###########################################

import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import re
import pandas as pd
import datasets

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from functools import reduce
import random

import numpy as np
import transformers
import pytorch_lightning as pl
from transformers import AutoTokenizer, BertForSequenceClassification


# load dataset from gdrive
dataset = {}
dataset[f'{TARGET}_{TARGET_OPT}_all'] = None
dataset[f'{TARGET}_{TARGET_OPT}_avail'] = None
dataset[f'{TARGET}_{TARGET_OPT}_nonavail'] = None
for name, dset in dataset.items():
    dataset[name] = pd.read_csv(f'{DATA_DIR}/{name}.csv')

print(dataset[f'{TARGET}_{TARGET_OPT}_all'][TARGET].value_counts())
print(dataset[f'{TARGET}_{TARGET_OPT}_avail'][TARGET].value_counts())
print(dataset[f'{TARGET}_{TARGET_OPT}_nonavail'][TARGET].value_counts())


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


# set weight
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


# ready to train

#from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class StatutesDataModule(pl.LightningDataModule):
    def __init__(self, column, trainset, statutes, tokenizer, datadf, batch_size=16, max_input_len=512):
        
        super().__init__()
        self.column = column
        self.trainset = trainset
        self.statutes = statutes
        self.tokenizer = tokenizer
        #self.data = data
        self.batch_size = batch_size
        self.max_input_len = max_input_len 

        #divide train/valid/test set
        train, test = train_test_split(datadf, test_size=0.2, random_state=1)
        train, valid = train_test_split(train, test_size=0.2, random_state=1)
        self.data = datasets.DatasetDict({'train': Dataset.from_dict(train), 'valid': Dataset.from_dict(valid), 'test': Dataset.from_dict(test)})

        ###############################################
        #위에 데이터셋 class, 및 등등 선언되어있다 가정#
        ###############################################

        #class 0 : 43200개, class 1 : 4800개
        ccnt = train[TARGET].value_counts()

        class_counts = train[TARGET].value_counts().to_list() #43200, 4800
        num_samples = sum(class_counts) # 48000 - 전체 데이터 갯수
        labels = train[TARGET].to_list()

        #클래스별 가중치 부여 [48000/43200, 48000/4800] => class 1에 가중치 높게 부여하게 됨
        ccnt_map = {}
        for index, row in train[TARGET].value_counts().iteritems():
            ccnt_map[index] = (float)(num_samples/row)

        # 해당 데이터의 label에 해당되는 가중치
        weights = []
        for l in labels:
            weights.append(ccnt_map[l])

        self.trainsampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples)*1)

    def setup(self, stage):
        pass
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn, num_workers=2, sampler=self.trainsampler)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['valid'], batch_size=self.batch_size*2, shuffle=False, collate_fn=self._collate_fn)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.batch_size*2, shuffle=False, collate_fn=self._collate_fn)
    
    def _collate_fn(self, batch):
        inputs = self.tokenizer([x[self.trainset] for x in batch], max_length=self.max_input_len, padding=True, truncation=True, return_tensors='pt')
        labels = [x[self.column] for x in batch]

        return inputs, labels
class StatutesClassifier(pl.LightningModule):
    def __init__(self, statutes, backbone, learning_rate=1e-5, mode='multi-class', approx_val=2):
        super().__init__()
        self.statutes = statutes
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.mode = mode
        self.criterion = torch.nn.CrossEntropyLoss() if mode == 'multi-class' else torch.nn.BCEWithLogitsLoss()
        self.approx_val = approx_val
        #self.criterion = torch.nn.MSELoss()
        #self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, batch):
        inputs, labels = batch
        logits = self.backbone(**inputs).logits
        targets = torch.zeros_like(logits)
        for i, label in enumerate(labels):
            label_id = self.statutes.index(label) if label < len(self.statutes) else 0
            # label_id = [(self.statutes.index(x) if x in self.statutes else 0) for x in label]
            #targets[i, label_id] = 1 / len(label_id) if self.mode == 'multi-class' else 1
            targets[i, label_id] = 1
        loss = self.criterion(logits, targets)

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch) 
    
    def validation_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch) 

    def test_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, True)

    def _evaluation_step(self, batch):
        loss, logits = self.forward(batch)
        _, gts = batch
        if self.mode == 'multi-class':
            prs = np.array(self.statutes)[logits.argmax(-1).cpu()].tolist()
        else:
            prs = [np.array(self.statutes)[np.where(logit.sigmoid().cpu() > 0.5)[0]] for logit in logits]

        # if isinstance(prs, list) == False:
        #     prs = [prs]
        
        return {'loss': loss.item(), 'gts': gts, 'prs': prs}

    def _evaluation_epoch_end(self, outputs, test=False):
        avg_loss = np.mean([x['loss'] for x in outputs])
        print(outputs)
        gts = reduce(lambda x,y: x + y, [x['gts'] for x in outputs], [])
        prs = reduce(lambda x,y: x + y, [x['prs'] for x in outputs], [])
        if self.mode == 'multi-class':
            #acc = sum([(pr in gt) for gt, pr in zip(gts, prs)]) / len(gts)
            acc = sum([abs(pr - gt) <= self.approx_val for gt, pr in zip(gts, prs)]) / len(gts)
            # acc = sum([abs(pr - gt[0]) <= self.approx_val for gt, pr in zip(gts, prs)]) / len(gts)
            
        else:
            acc = sum([(set(pr) == set(gt)) for gt, pr in zip(gts, prs)]) / len(gts)
        
        print('='*50)
        print(f'avg_loss: {avg_loss}')
        print(f'ACC: {acc}')
        
        #target_ids = random.sample(range(len(gts)), 2)
        # target_ids = random.sample(range(len(prs)), 2)
        # print(gts)
        # print(prs)
        # print(f'target_ids: {target_ids}')
        # for target_id in target_ids:
        #     print(f'GT: {gts[target_id]}\t\t\tPR: {prs[target_id]}')
        
        # if self.mode == 'multi-class' and test:
        #     #df = pd.DataFrame(columns=['idx', 'gt', 'pr', 'etc'])
        #     #df = pd.DataFrame({'idx': pd.Series(dtype='int'), 'gt':pd.Series(dtype='int'), 'pr':pd.Series(dtype='int'),'etc':[]})
        #     df = pd.DataFrame({'idx': pd.Series(dtype='int'), 'gt':pd.Series(dtype='int'), 'pr':pd.Series(dtype='int')})
        #     print(data)
        #     for i, (gt, pr) in enumerate(zip(gts, prs)):
        #         if pr not in gt:
        #             # df = df.append(pd.DataFrame({'idx': [i],
        #             #                              'gt': ', '.join((str)(gt)),
        #             #                              'pr': pr, 
        #             #                              'etc': data['test'][i]['etc']}),
        #             #                 ignore_index=True)
        #             df = df.append(pd.DataFrame({'idx': [i],
        #                                          'gt': ', '.join((str)(gt)),
        #                                          'pr': pr}),
        #                             ignore_index=True)
        #     df.to_csv('GT_vs_PR.csv', index=False)
            

    def configure_optimizers(self):
        grouped_params = [{'params': list(filter(lambda p: p.requires_grad, self.parameters())), 'lr': self.learning_rate}]
        optimizer = torch.optim.AdamW(grouped_params, lr=self.learning_rate)

        return {'optimizer': optimizer}
from sklearn.model_selection import train_test_split
import datasets
import tensorflow as tf
from datasets import Dataset

def convertDFtoDS(df):
    return tf.data.Dataset.from_tensor_slices(dict(df))

backbone_card = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(backbone_card)

# # coded by cat
# cfdxf,er?"
# ]\w
# ws"
import gc

def execTrain(dataset, target, trainset):
    #init cuda
    gc.collect()
    torch.cuda.empty_cache()

    # #divide train/valid/test set
    # train, test = train_test_split(dataset, test_size=0.2, random_state=1)
    # train, valid = train_test_split(train, test_size=0.2, random_state=1)
    # d = datasets.DatasetDict({'train': Dataset.from_dict(train), 'valid': Dataset.from_dict(valid), 'test': Dataset.from_dict(test)})

    #print(set(dataset[target].to_list()))

    # labels init

    labels = sorted(set(dataset[target].to_list()))
    data_module = StatutesDataModule(target, trainset, labels, tokenizer, dataset, batch_size=16, max_input_len=512)
    #data_module = StatutesDataModule(target, trainset, labels, tokenizer, dataset, batch_size=16, max_input_len=384)

    # model
    backbone = BertForSequenceClassification.from_pretrained(backbone_card, num_labels=len(labels))
    model = StatutesClassifier(labels, backbone, learning_rate=2e-5, mode='multi-class', approx_val=TARGET_APPROX)

    # trainer
    n_gpus = 1 #torch.cuda.device_count()
    trainer = pl.Trainer(max_epochs=20, gpus=n_gpus, fast_dev_run=not True)

    # train and eval
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    return model

def execEval(dataset, target, trainset, modelpath):
    #init cuda
    gc.collect()
    torch.cuda.empty_cache()

    #divide train/valid/test set
    train, test = train_test_split(dataset, test_size=0.2, random_state=1)
    train, valid = train_test_split(train, test_size=0.2, random_state=1)
    d = datasets.DatasetDict({'train': Dataset.from_dict(train), 'valid': Dataset.from_dict(valid), 'test': Dataset.from_dict(test)})

    # labels init
    labels = sorted(set.union(*map(set, dataset[target])))
    data_module = StatutesDataModule(target, trainset, labels, tokenizer, d, batch_size=16, max_input_len=512)

    # model
    backbone = BertForSequenceClassification.from_pretrained(backbone_card, num_labels=len(labels))
    model = StatutesClassifier(labels, backbone, learning_rate=2e-5, mode='multi-class')
    model = torch.load(modelpath)

    # trainer
    n_gpus = 1 #torch.cuda.device_count()
    trainer = pl.Trainer(max_epochs=20, gpus=n_gpus, fast_dev_run=not True)

    trainer.test(model, data_module)

## RUN
# check variables
print(TARGET)
print(TARGET_SET)
print(TARGET_APPROX)
print(TARGET_OPT)
print(TRAIN_COLUMN)
dataset[TARGET_SET]
type(dataset[TARGET_SET])
# 양형의 이유 있는 데이터 중 etc로 prison classification, MSE
#model = execTrain(dataset[TARGET_SET], TARGET, TRAIN_COLUMN)
#torch.save(model, f'{MODEL_DIR}/{TARGET}_{TARGET_SET}_{TRAIN_COLUMN}_{TARGET_START}_{TARGET_END}_model_imb.pt')