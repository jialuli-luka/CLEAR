import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer
import utils
import model
import param
from param import args
from collections import defaultdict

from transformers import BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch.nn import DataParallel as DP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class mBERT():
    def __init__(self, env, results_path, tok, episode_len=20):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = []
        self.tok = tok
        self.episode_len = episode_len

        model_config = BertConfig.from_pretrained("bert-base-multilingual-cased", return_dict=True, cache_dir="./cache/")
        self.encoder = BertModel.from_pretrained("bert-base-multilingual-cased", config=model_config, cache_dir="./cache/")
        self.encoder = DP(self.encoder).to(device)

        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        encoder_parameters = [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.bert_decay,
            },
            {"params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.encoder_optimizer = AdamW(encoder_parameters, lr=args.bert_lr)
        self.encoder_scheduler = get_linear_schedule_with_warmup(self.encoder_optimizer, 0.2 * args.iters, args.iters)

        self.losses = []

        self.contrastive_loss = model.ContrastiveLoss()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch_contrastive(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = []
        seq_lengths = []
        attention_mask = []
        pair_seq = []
        pair_length = []
        pair_attn = []
        for ob in obs:
            seq_tensor.append(ob['instr_encoding'])
            seq_lengths.append(ob['seq_length'])
            attention_mask.append(ob['seq_mask'])
            pair_seq.append(ob['paired_encoding'])
            pair_length.append(ob['paired_length'])
            pair_attn.append(ob['paired_mask'])
        seq_tensor = torch.from_numpy(np.array(seq_tensor))
        seq_lengths = torch.from_numpy(np.array(seq_lengths))
        attention_mask = torch.from_numpy(np.array(attention_mask))
        mask = utils.length2mask(seq_lengths, args.maxInput)    #attention_mask = 1-mask
        pair_seq = torch.from_numpy(np.array(pair_seq))
        pair_length = torch.from_numpy(np.array(pair_length))
        pair_attn = torch.from_numpy(np.array(pair_attn))
        pair_mask = utils.length2mask(pair_length, args.maxInput)

        return Variable(seq_tensor, requires_grad=False).long().to(device), \
               mask.to(device),  attention_mask.to(device), \
               list(seq_lengths), \
               Variable(pair_seq, requires_grad=False).long().to(device), \
               pair_mask.to(device), pair_attn.to(device), \
               list(pair_length)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).to(device)

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return torch.from_numpy(candidate_feat).to(device), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).to(device)

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def finetune(self, train_ml=True, reset=True):

        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        seq, seq_mask, attention_mask, seq_lengths, pair_seq, pair_mask, pair_attn, pair_length = self._sort_batch_contrastive(obs)

        text_features = self.encoder(seq, attention_mask=attention_mask)

        paired_text = self.encoder(pair_seq, attention_mask=pair_attn)

        con_loss = self.contrastive_loss(text_features.last_hidden_state[:, 0, :], paired_text.last_hidden_state[:, 0, :])

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'path_id': ob['path_id']
        } for ob in obs]

        if train_ml:
            self.loss += con_loss * args.con_weight / batch_size
            self.logs['con_loss'].append(con_loss)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        return traj, con_loss.item() * args.con_weight / batch_size

    def get_text_features(self, seq, attention_mask):

        text_features = self.encoder(seq, attention_mask=attention_mask)

        return text_features.cpu().numpy()

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, **kwargs):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        self.encoder.eval()
        self.env.reset_epoch(shuffle=(iters is not None))  # If iters is not none, shuffle the env batch
        self.losses = []
        visited = []
        looped = False
        # We rely on env showing the entire batch before repeating anything
        results = 0.
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                self.loss = 0
                traj, loss = self.finetune(train_ml=False, **kwargs)
                results += loss
        else:  # Do a full round
            while True:
                self.loss = 0
                trajs, loss = self.finetune(train_ml=False, **kwargs)
                for traj in trajs:
                    if traj['instr_id'] in visited:
                        looped = True
                    else:
                        visited.append(traj['instr_id'])
                if looped:
                    break
                results += loss

        return results

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.encoder.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.encoder_optimizer.zero_grad()

            self.loss = 0
            self.finetune(**kwargs)

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.encoder_scheduler.step()

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer)]

        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTENER")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer)]

        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1
