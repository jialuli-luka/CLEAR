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


class Visual():
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = []
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        self.visual_encoder = model.VisualEncoder(self.env.feature_size)
        self.visual_encoder = DP(self.visual_encoder).to(device)

        self.visual_optimizer = args.optimizer(self.visual_encoder.parameters(), lr=args.lr)

        self.losses = []

        self.contrastive_loss = model.ContrastiveLoss()
        self.l2_loss = nn.MSELoss(size_average=False)
        self.l1_loss = nn.L1Loss(size_average=False)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = []
        seq_lengths = []
        attention_mask = []
        for ob in obs:
            seq_tensor.append(ob['instr_encoding'])
            seq_lengths.append(ob['seq_length'])
            attention_mask.append(ob['seq_mask'])
        seq_tensor = torch.from_numpy(np.array(seq_tensor))
        seq_lengths = torch.from_numpy(np.array(seq_lengths))
        attention_mask = torch.from_numpy(np.array(attention_mask))
        mask = utils.length2mask(seq_lengths, args.maxInput)    #attention_mask = 1-mask

        return Variable(seq_tensor, requires_grad=False).long().to(device), \
               mask.to(device),  attention_mask.to(device), \
               list(seq_lengths)

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

    def get_objects_constraints(self, obs, step):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        object = []
        pair_object = []
        for ob in obs:
            if step < len(ob['objects']):
                object.append(ob['objects'][step])
                pair_object.append(ob['pair_objects'][step])
            else:
                object.append(ob['objects'][-1])
                pair_object.append(ob['pair_objects'][-1])
        object = torch.from_numpy(np.array(object))
        pair_object = torch.from_numpy(np.array(pair_object))

        return Variable(object, requires_grad=False).long().to(device), Variable(pair_object, requires_grad=False).long().to(device)

    def _feature_variable(self, obs, step):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.env.feature_size), dtype=np.float32)
        pair_features = np.empty((len(obs), args.views, self.env.feature_size), dtype=np.float32)
        indicator = np.array([False] * len(obs))
        for i, ob in enumerate(obs):
            assert len(ob['feature']) == len(ob['pair_feature'])
            # features[i, :, :] = ob['feature']   # Image feat
            if step < len(ob['feature']):
                features[i, :, :] = ob['feature'][step]
                pair_features[i, :, :] = ob['pair_feature'][step]
            else:
                features[i, :, :] = ob['feature'][-1]
                pair_features[i, :, :] = ob['pair_feature'][-1]
                indicator[i] = True
        return Variable(torch.from_numpy(features), requires_grad=False).to(device), Variable(torch.from_numpy(pair_features), requires_grad=False).to(device), indicator

    def get_input_feat(self, obs, step):

        f_t, pair_f_t, indicator = self._feature_variable(obs, step)      # Image features from obs
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]

        return f_t, pair_f_t, candidate_leng, indicator

    def _teacher_action(self, obs, ended, step):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                if step + 1 >= len(ob['path']):
                    a[i] = len(ob['candidate'])
                else:
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['path'][step + 1]:
                            a[i] = k
                            break
                    else:   # Stop here
                        assert False
        return torch.from_numpy(a).to(device)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None and state.location.viewpointId != traj[i]['path'][-1][0]:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def finetune(self, train_ml=True, reset=True):

        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'path_id': ob['path_id']
        } for ob in obs]
        #
        # Initialization the tracking state
        ended = np.array([False] * batch_size)   # Indices match permuation of the model, not env
        #
        # Init the logs
        con_loss = 0.
        total = 0.

        for t in range(self.episode_len):
            contra_label = np.arange(batch_size)
            contra_label[ended] = args.ignoreid
            contra_label = torch.from_numpy(contra_label).detach().cuda()
            image, pair_image, candidate_leng, indicator = self.get_input_feat(obs, t)

            if args.objects_constraints:
                object, pair_object = self.get_objects_constraints(obs, t)
                object_mask = torch.logical_not(torch.sum(torch.logical_and(object, pair_object), dim=-1)).unsqueeze(-1)
            else:
                object_mask = None

            output_image = self.visual_encoder(image, objects=object_mask)
            output_pair = self.visual_encoder(pair_image, objects=object_mask)
            con_loss += self.contrastive_loss(output_image, output_pair, contra_label)

            # Supervised training
            target = self._teacher_action(obs, ended, t).cpu().numpy()

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = target
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, None, traj)
            obs = np.array(self.env._get_obs())

            total += batch_size - np.sum(ended)

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml:
            self.loss += con_loss * args.con_weight / total
            self.logs['con_loss'].append(con_loss)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        return traj, con_loss.item() * args.con_weight / total

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, **kwargs):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        self.visual_encoder.eval()
        self.env.reset_epoch(shuffle=(iters is not None))  # If iters is not none, shuffle the env batch
        self.losses = []
        visited = []
        looped = False
        results = 0.
        if iters is not None:
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

        self.visual_encoder.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.visual_optimizer.zero_grad()

            self.loss = 0
            self.finetune(**kwargs)

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.visual_encoder.parameters(), 40.)
            self.visual_optimizer.step()

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
        all_tuple = [("visual_encoder", self.visual_encoder, self.visual_optimizer)]

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
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                if optimizer is not None:
                    optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("visual_encoder", self.visual_encoder, self.visual_optimizer)]

        for param in all_tuple:
            recover_state(*param)
        return states['visual_encoder']['epoch'] - 1
