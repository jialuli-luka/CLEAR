import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker
from mbert import mBERT

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter
from transformers import BertTokenizer

import sys
import random
import math

import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils import padding_idx, add_idx, Tokenizer
from collections import defaultdict

from transformers import BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup

import heapq

import CLIP.clip as clip


# Generate image features using CLIP

import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

path = "../views_img"
files= os.listdir(path)
features = dict()
for scan in files:
    views = os.listdir(path+"/"+scan)
    for view in views:
        imgs = os.listdir(path+"/"+scan+"/"+view)
        image_features = None
        for img in imgs:
            image = preprocess(Image.open(path+"/"+scan+"/"+view+"/"+img)).unsqueeze(0).to(device)
            image_feature = model.encode_image(image)
            if image_features is None:
                image_features = image_feature.unsqueeze(0)
            else:
                image_features = torch.cat((image_features, image_feature.unsqueeze(0)),0)
        features[scan+"_"+view] = image_features



