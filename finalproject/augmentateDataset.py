#!/usr/bin/env python
# coding: utf-8

###########################################
# module for generate data                #
###########################################

# variables
TARGET = 'prison'
TARGET_START = 0 # 0 if you want zero values, INCLUDE THST VALUE
TARGET_END = 36 # INCLUDE THIS VALUE
TARGET_COUNT = 10000 # 한 컬럼 최대 갯수
TARGET_OPT = 'class' # class, binary
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


# https://github.com/jucho2725/ktextaug