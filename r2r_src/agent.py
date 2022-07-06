
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

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        if args.dataset == 'CVDN':
            output = [{'instr_id': k, 'trajectory': v[0], 'inst_idx': v[1]} for k, v in self.results.items()]
        else:
            if args.save_attention:
                output = [{'instr_id': k, 'trajectory': v[0], 'path_id': v[1], 'attn': v[2]} for k, v in self.results.items()]
            else:
                output = [{'instr_id': k, 'trajectory': v[0], 'path_id':v[1]} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = [traj['path'], traj['path_id']]
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        if args.save_attention:
                            self.results[traj['instr_id']] = [traj['path'], traj['path_id'], traj['attn']]
                        else:
                            self.results[traj['instr_id']] = [traj['path'], traj['path_id']]

                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20, last_epoch=-1, eval=None):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # For NDTW reward
        self.eval = eval

        # Models
        # enc_hidden_size = args.rnn_dim//2 if args.bidir else args.rnn_dim
        # if args.finetune:
        if args.encoder_backbone == 'bert':
            model_config = BertConfig.from_pretrained("bert-base-multilingual-cased", return_dict=True, cache_dir="./cache/")
            # model_config = BertConfig.from_pretrained("bert-base-multilingual-cased")
            self.encoder = BertModel.from_pretrained("bert-base-multilingual-cased", config=model_config, cache_dir="./cache/")
            self.encoder = DP(self.encoder).to(device)
        elif args.encoder_backbone == 'mclip':
            self.encoder = multilingual_clip.load_model('M-BERT-Base-ViT-B')
            self.encoder = DP(self.encoder).to(device)

        self.encoder_pos = model.EncoderMBert(args.bert_dim)
        self.encoder_pos = DP(self.encoder_pos).to(device)
        # else:
        #     self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
        #     self.encoder_pos = model.EncoderMBert(args.bert_dim).to(device)
        # self.encoder = model.EncoderMBert(args.bert_dim).to(device)
        self.decoder = model.AttnDecoderLSTM(args.aemb, args.bert_dim, args.dropout, feature_size=self.feature_size + args.angle_feat_size)
        self.decoder = DP(self.decoder).to(device)
        self.critic = model.Critic()
        self.critic = DP(self.critic).to(device)
        self.models = (self.encoder, self.encoder_pos, self.decoder, self.critic)
        if args.discriminator:
            self.discriminator = model.Discriminator()
            self.discriminator = DP(self.discriminator).to(device)
            self.models = (self.encoder, self.encoder_pos, self.decoder, self.critic, self.discriminator)
        if args.load_visual is not None:
            self.visual_encoder = model.VisualEncoder(self.feature_size)
            self.visual_encoder = DP(self.visual_encoder).to(device)
            if args.visual_combine:
                self.visual_combine = model.VisualCombine(self.feature_size).to(device)
                self.models = (self.encoder, self.encoder_pos, self.visual_encoder, self.visual_combine, self.decoder, self.critic)
            else:
                self.models = (self.encoder, self.encoder_pos, self.visual_encoder, self.decoder, self.critic)

        # Optimizers
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        encoder_parameters = [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.bert_decay,
            },
            {"params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        if args.encoder_backbone == 'mclip':
            if args.optim_mclip == 'adam':
                self.encoder_optimizer = args.optimizer_mclip(encoder_parameters, lr=args.bert_lr)
                self.encoder_scheduler = None
            else:
                self.encoder_optimizer = AdamW(encoder_parameters, lr=args.bert_lr)
                self.encoder_scheduler = get_linear_schedule_with_warmup(self.encoder_optimizer, 0.2 * args.iters,
                                                                         args.iters, last_epoch=last_epoch)
        else:
            self.encoder_optimizer = AdamW(encoder_parameters, lr=args.bert_lr)
            self.encoder_scheduler = get_linear_schedule_with_warmup(self.encoder_optimizer, 0.2*args.iters, args.iters, last_epoch=last_epoch)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        # if args.finetune:
        encoder_pos_parameters = [
            {
                "params": [p for n, p in self.encoder_pos.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.bert_decay,
            },
            {"params": [p for n, p in self.encoder_pos.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.encoder_pos_optimizer = AdamW(encoder_pos_parameters, lr=args.bert_lr)
        self.encoder_pos_scheduler = get_linear_schedule_with_warmup(self.encoder_pos_optimizer, 0.2*args.iters, args.iters, last_epoch=last_epoch)
        self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer, self.encoder_pos_optimizer)
        if args.discriminator:
            if args.dis_optim == 'rms':
                self.discriminator_optimizer = args.optimizer(self.discriminator.parameters(), lr=args.lr)
            elif args.dis_optim == 'adam':
                discriminator_parameters = [
                    {
                        "params": [p for n, p in self.discriminator.named_parameters() if
                                   not any(nd in n for nd in no_decay)],
                        "weight_decay": args.bert_decay,
                    },
                    {"params": [p for n, p in self.discriminator.named_parameters() if any(nd in n for nd in no_decay)],
                     "weight_decay": 0.0},
                ]
                self.discriminator_optimizer = AdamW(discriminator_parameters, lr=args.bert_lr)
                self.discriminator_scheduler = get_linear_schedule_with_warmup(self.discriminator_optimizer, 0.2*args.iters, args.iters, last_epoch=last_epoch)
            self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer, self.encoder_pos_optimizer, self.discriminator_optimizer)
        if args.load_visual is not None:
            self.visual_optimizer = args.optimizer(self.visual_encoder.parameters(), lr=args.lr)
            if args.visual_combine:
                self.visual_combine_optimizer = args.optimizer(self.visual_combine.parameters(), lr=args.lr)
                self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer, self.encoder_pos_optimizer, self.visual_optimizer, self.visual_combine_optimizer)
            else:
                self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer, self.encoder_pos_optimizer, self.visual_optimizer)

        # else:
        #     self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        if args.discriminator:
            self.dis_criterion = nn.CrossEntropyLoss(size_average=False)

        if args.contrastive:
            self.contrastive_loss = model.ContrastiveLoss()

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

    def _sort_batch_instructions(self, obs):
        txts = []
        for ob in obs:
            txts.append(ob['instruction'])
        return txts

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
        return Variable(torch.from_numpy(candidate_feat), requires_grad=False).to(device), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).to(device)

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def _teacher_action_(self, obs, ended):
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
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).to(device)

    def _teacher_action(self, obs, ended):
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
                if ob['viewpoint'] == ob['path'][-1]:
                    a[i] = len(ob['candidate'])
                    assert ob['teacher'] == ob['viewpoint']
                elif ob['viewpoint'] in ob['path']:
                    step = ob['path'].index(ob['viewpoint'])
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['path'][step + 1]:
                            a[i] = k
                            break
                    else:
                        assert False
                else:
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['teacher']:  # Next view point
                            a[i] = k
                            break
                    else:  # Stop here
                        assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                        a[i] = len(ob['candidate'])

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
                if args.save_attention:
                    traj[i]['path_attn'].append(state.location.viewpointId)
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

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        if speaker is not None:         # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.feature_size).to(device))
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))

        if args.contrastive and train_ml:
            seq, seq_mask, attention_mask, seq_lengths, pair_seq, pair_mask, pair_attn, pair_length = self._sort_batch_contrastive(obs)
        else:
            seq, seq_mask, attention_mask, seq_lengths = self._sort_batch(obs)

        # text_features = self.encoder(seq, attention_mask=attention_mask)

        if args.encoder_backbone == 'bert':
            text_features = self.encoder(seq, attention_mask=attention_mask)
            ctx, h_t, c_t = self.encoder_pos(
                text_features.last_hidden_state)  # batch_size, sequence_length, hidden_size
            ctx_mask = seq_mask
        elif args.encoder_backbone == 'mclip':
            txts = self._sort_batch_instructions(obs)
            _, text_features, attention_mask = self.encoder(txts, sequence=True)
            # print(text_features.shape)
            # exit()
            ctx, h_t, c_t = self.encoder_pos(text_features)
            ctx_mask = torch.logical_not(attention_mask)

        if args.contrastive and train_ml:
            paired_text = self.encoder(pair_seq, attention_mask=pair_attn)
            paired_ctx, paired_h_t, paired_c_t = self.encoder_pos(paired_text.last_hidden_state)
            paired_mask = pair_mask

        # Discriminator
        if args.discriminator and train_ml:
            dis_loss = 0.
            dis_logit = self.discriminator(text_features.last_hidden_state)
            dis_target = np.zeros(len(obs), dtype=np.int64)
            for i, ob in enumerate(obs):
                if ob['language'] == 'en':
                    dis_target[i] = 0
                elif ob['language'] == 'hi':
                    dis_target[i] = 1
                elif ob['language'] == 'te':
                    dis_target[i] = 2
                else:
                    print("error")
                    exit()
            dis_target = torch.from_numpy(dis_target).to(device)
            # print(dis_logit.shape)
            # print(dis_target.shape)
            dis_loss = args.dis_weight * self.dis_criterion(dis_logit, dis_target)
            self.logs['dis_loss'].append(dis_loss / batch_size)

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            if train_rl and args.ndtw:
                last_ndtw[i] = self.eval.ndtw(ob['scan'], [(ob['viewpoint'], ob['heading'], ob['elevation'])], ob['path'])


        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'path_id': ob['path_id'],
            'gt': ob['path'],
            'path_attn': [ob['viewpoint']],
            'attn': []
        } for ob in obs]

        # For test result submission
        visited = [set() for _ in obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)   # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        con_loss = 0.

        h1 = h_t
        for t in range(self.episode_len):
            # print(t)
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)
            if args.load_visual is not None:
                f_t_shared = f_t.clone()
                candidate_feat_shared = candidate_feat.clone()
                f_t_shared[:,:,:-args.angle_feat_size] = self.visual_encoder(f_t[:,:,:-args.angle_feat_size])
                candidate_feat_shared[:,:,:-args.angle_feat_size] = self.visual_encoder(candidate_feat[:,:,:-args.angle_feat_size])
                if args.visual_combine:
                    f_t_combined = f_t.clone()
                    candidate_feat_combined = candidate_feat.clone()
                    f_t_combined[:,:,:-args.angle_feat_size] = self.visual_combine(f_t_shared[:,:,:-args.angle_feat_size], f_t[:,:,:-args.angle_feat_size])
                    candidate_feat_combined[:,:,:-args.angle_feat_size] = self.visual_combine(candidate_feat_shared[:,:,:-args.angle_feat_size], candidate_feat[:,:,:-args.angle_feat_size])
                else:
                    f_t_combined = f_t_shared
                    candidate_feat_combined = candidate_feat_shared
            else:
                f_t_combined = f_t
                candidate_feat_combined = candidate_feat

            if speaker is not None:       # Apply the env drop mask to the feat
                candidate_feat_combined[..., :-args.angle_feat_size] *= noise
                f_t_combined[..., :-args.angle_feat_size] *= noise

            if args.contrastive and train_ml:
                h_t, c_t, logit, h1, weighted_ctx, weighted_pair = self.decoder(input_a_t, f_t_combined, candidate_feat_combined,
                                                   h_t, h1, c_t,
                                                   ctx, ctx_mask,
                                                   already_dropfeat=(speaker is not None),
                                                   pair_ctx=paired_ctx, pair_mask=paired_mask)
                con_loss += self.contrastive_loss(weighted_ctx, weighted_pair)

            else:
                if args.save_attention:
                    h_t, c_t, logit, h1, _, _, l_attn, v_attn = self.decoder(input_a_t, f_t_combined, candidate_feat_combined,
                                                             h_t, h1, c_t,
                                                             ctx, ctx_mask,
                                                             already_dropfeat=(speaker is not None))
                    for i in range(batch_size):
                        if obs[i]['instructions'] == "Right now you're facing towards a fireplace. Now turn right, there are two open door in front of you, move towards the open door which is to your right and exit the room through the door. Now slightly turn left, there is an open door in front of you, move towards the door and enter in to the room. Now you are facing towards a sofa, move forward and stand in front of the sofa and it is your end point.":
                            traj[i]['attn'].append(l_attn[i])
                else:
                    h_t, c_t, logit, h1, _, _ = self.decoder(input_a_t, f_t_combined, candidate_feat_combined,
                                                   h_t, h1, c_t,
                                                   ctx, ctx_mask,
                                                   already_dropfeat=(speaker is not None))

            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            if args.submit:     # Avoding cyclic path
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, None, traj)
            obs = np.array(self.env._get_obs())
            # perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            # Calculate the mask and reward
            if train_rl:
                dist = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    if args.ndtw:
                        ndtw_score[i] = self.eval.ndtw(ob['scan'], traj[i]['path'], ob['path'])
                    else:
                        ndtw_score[i] = 0
                    if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                        reward[i] = 0.
                        mask[i] = 0.
                    else:       # Calculate the reward
                        action_idx = cpu_a_t[i]
                        if action_idx == -1:        # If the action now is end
                            if dist[i] < 3:         # Correct
                                reward[i] = 2. + ndtw_score[i] * 2.0
                            else:                   # Incorrect
                                reward[i] = -2.
                        else:                       # The action is not end
                            reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0:                           # Quantification
                                reward[i] = 1 + ndtw_reward
                            elif reward[i] < 0:
                                reward[i] = -1 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            if args.load_visual is not None:
                f_t_shared = f_t.clone()
                candidate_feat_shared = candidate_feat.clone()
                f_t_shared[:,:,:-args.angle_feat_size] = self.visual_encoder(f_t[:,:,:-args.angle_feat_size])
                candidate_feat_shared[:,:,:-args.angle_feat_size] = self.visual_encoder(candidate_feat[:,:,:-args.angle_feat_size])
                if args.visual_combine:
                    f_t_combined = f_t.clone()
                    candidate_feat_combined = candidate_feat.clone()
                    f_t_combined[:,:,:-args.angle_feat_size] = self.visual_combine(f_t_shared[:,:,:-args.angle_feat_size], f_t[:,:,:-args.angle_feat_size])
                    candidate_feat_combined[:,:,:-args.angle_feat_size] = self.visual_combine(candidate_feat_shared[:,:,:-args.angle_feat_size], candidate_feat[:,:,:-args.angle_feat_size])
                else:
                    f_t_combined = f_t_shared
                    candidate_feat_combined = candidate_feat_shared
            else:
                f_t_combined = f_t
                candidate_feat_combined = candidate_feat

            if speaker is not None:
                candidate_feat_combined[..., :-args.angle_feat_size] *= noise
                f_t_combined[..., :-args.angle_feat_size] *= noise
            last_h_, _, _, _, _, _ = self.decoder(input_a_t, f_t_combined, candidate_feat_combined,
                                            h_t, h1, c_t,
                                            ctx, ctx_mask,
                                            speaker is not None)
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).to(device)
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).to(device)
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss

        if train_ml is not None:
            self.logs['us_loss'].append(ml_loss * train_ml / batch_size)
            self.loss += ml_loss * train_ml / batch_size
            if args.discriminator:
                self.loss += dis_loss / batch_size
            if args.contrastive:
                self.loss += con_loss * args.con_weight / batch_size
                self.logs['con_loss'].append(con_loss)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        # for i in range(batch_size):
        #     if obs[i]['instructions'] == "Right now you're facing towards a fireplace. Now turn right, there are two open door in front of you, move towards the open door which is to your right and exit the room through the door. Now slightly turn left, there is an open door in front of you, move towards the door and enter in to the room. Now you are facing towards a sofa, move forward and stand in front of the sofa and it is your end point.":
        #         print(traj[i]['attn'])
        #         print(traj[i]['path_attn'])
        #         print(traj[i]['instr_id'])
        #         exit()
            # if traj[i]['path_attn'][:len(traj[i]['gt'])] == traj[i]['gt'] and obs[i]['language'] == 'en':
            #     print(traj[i]['attn'])
            #     print(obs[i]['scan'])
            #     print(traj[i]['gt'])
            #     print(obs[i]['instructions'])

        return traj

    def get_path_ins_representation(self, speaker=None):
        self.encoder.eval()
        self.encoder_pos.eval()
        self.decoder.eval()
        self.critic.eval()
        obs = np.array(self.env.reset())

        batch_size = len(obs)

        path_representation = np.zeros((batch_size, args.bert_dim))
        ins_representation = np.zeros((batch_size, args.bert_dim))

        seq, seq_mask, attention_mask, seq_lengths = self._sort_batch(obs)

        # text_features = self.encoder(seq, attention_mask=attention_mask)
        text_features = self.encoder(seq, attention_mask=attention_mask)
        ctx, h_t, c_t = self.encoder_pos(
            text_features.last_hidden_state)  # batch_size, sequence_length, hidden_size
        ctx_mask = seq_mask

        ins_representation = ctx[:,0,:].detach().cpu().numpy()

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'path_id': ob['path_id'],
            'gt': ob['path'],
            'path_attn': [ob['viewpoint']],
            'attn': []
        } for ob in obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)   # Indices match permuation of the model, not env

        hidden_states = []

        h1 = h_t
        for t in range(self.episode_len):
            # print(t)
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)
            if args.load_visual is not None:
                f_t_shared = f_t.clone()
                candidate_feat_shared = candidate_feat.clone()
                f_t_shared[:,:,:-args.angle_feat_size] = self.visual_encoder(f_t[:,:,:-args.angle_feat_size])
                candidate_feat_shared[:,:,:-args.angle_feat_size] = self.visual_encoder(candidate_feat[:,:,:-args.angle_feat_size])
                f_t_combined = f_t_shared
                candidate_feat_combined = candidate_feat_shared
            else:
                f_t_combined = f_t
                candidate_feat_combined = candidate_feat

            h_t, c_t, logit, h1, _, _ = self.decoder(input_a_t, f_t_combined, candidate_feat_combined,
                                                   h_t, h1, c_t,
                                                   ctx, ctx_mask,
                                                   already_dropfeat=(speaker is not None))

            hidden_states.append(h_t)

            # Supervised training
            target = self._teacher_action(obs, ended)

            a_t = target

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()

            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1):
                    path_representation[i,:] = h1[i,:].detach().cpu().numpy()
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, None, traj)
            obs = np.array(self.env._get_obs())
            # perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        return ins_representation, path_representation, traj

    def _dijkstra(self):
        """
        The dijkstra algorithm.
        Was called beam search to be consistent with existing work.
        But it actually finds the Exact K paths with smallest listener log_prob.
        :return:
        [{
            "scan": XXX
            "instr_id":XXX,
            'instr_encoding": XXX
            'dijk_path': [v1, v2, ..., vn]      (The path used for find all the candidates)
            "paths": {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        }]
        """
        def make_state_id(viewpoint, action):     # Make state id
            return "%s_%s" % (viewpoint, str(action))
        def decompose_state_id(state_id):     # Make state id
            viewpoint, action = state_id.split("_")
            action = int(action)
            return viewpoint, action

        # Get first obs
        obs = self.env._get_obs()

        # Prepare the state id
        batch_size = len(obs)
        results = [{"scan": ob['scan'],
                    "instr_id": ob['instr_id'],
                    "instr_encoding": ob["instr_encoding"],
                    "dijk_path": [ob['viewpoint']],
                    "paths": []} for ob in obs]

        # Encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        recover_idx = np.zeros_like(perm_idx)
        for i, idx in enumerate(perm_idx):
            recover_idx[idx] = i
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx, h_t, c_t, ctx_mask = ctx[recover_idx], h_t[recover_idx], c_t[recover_idx], seq_mask[recover_idx]    # Recover the original order

        # Dijk Graph States:
        id2state = [
            {make_state_id(ob['viewpoint'], -95):
                 {"next_viewpoint": ob['viewpoint'],
                  "running_state": (h_t[i], h_t[i], c_t[i]),
                  "location": (ob['viewpoint'], ob['heading'], ob['elevation']),
                  "feature": None,
                  "from_state_id": None,
                  "score": 0,
                  "scores": [],
                  "actions": [],
                  }
             }
            for i, ob in enumerate(obs)
        ]    # -95 is the start point
        visited = [set() for _ in range(batch_size)]
        finished = [set() for _ in range(batch_size)]
        graphs = [utils.FloydGraph() for _ in range(batch_size)]        # For the navigation path
        ended = np.array([False] * batch_size)

        # Dijk Algorithm
        for _ in range(300):
            # Get the state with smallest score for each batch
            # If the batch is not ended, find the smallest item.
            # Else use a random item from the dict  (It always exists)
            smallest_idXstate = [
                max(((state_id, state) for state_id, state in id2state[i].items() if state_id not in visited[i]),
                    key=lambda item: item[1]['score'])
                if not ended[i]
                else
                next(iter(id2state[i].items()))
                for i in range(batch_size)
            ]

            # Set the visited and the end seqs
            for i, (state_id, state) in enumerate(smallest_idXstate):
                assert (ended[i]) or (state_id not in visited[i])
                if not ended[i]:
                    viewpoint, action = decompose_state_id(state_id)
                    visited[i].add(state_id)
                    if action == -1:
                        finished[i].add(state_id)
                        if len(finished[i]) >= args.candidates:     # Get enough candidates
                            ended[i] = True

            # Gather the running state in the batch
            h_ts, h1s, c_ts = zip(*(idXstate[1]['running_state'] for idXstate in smallest_idXstate))
            h_t, h1, c_t = torch.stack(h_ts), torch.stack(h1s), torch.stack(c_ts)

            # Recover the env and gather the feature
            for i, (state_id, state) in enumerate(smallest_idXstate):
                next_viewpoint = state['next_viewpoint']
                scan = results[i]['scan']
                from_viewpoint, heading, elevation = state['location']
                self.env.env.sims[i].newEpisode(scan, next_viewpoint, heading, elevation) # Heading, elevation is not used in panoramic
            obs = self.env._get_obs()

            # Update the floyd graph
            # Only used to shorten the navigation length
            # Will not effect the result
            for i, ob in enumerate(obs):
                viewpoint = ob['viewpoint']
                if not graphs[i].visited(viewpoint):    # Update the Graph
                    for c in ob['candidate']:
                        next_viewpoint = c['viewpointId']
                        dis = self.env.distances[ob['scan']][viewpoint][next_viewpoint]
                        graphs[i].add_edge(viewpoint, next_viewpoint, dis)
                    graphs[i].update(viewpoint)
                results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], viewpoint))

            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            # Run one decoding step
            h_t, c_t, alpha, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat,
                                                      h_t, h1, c_t,
                                                      ctx, ctx_mask,
                                                      False)

            # Update the dijk graph's states with the newly visited viewpoint
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))
            log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            _, max_act = log_probs.max(1)

            for i, ob in enumerate(obs):
                current_viewpoint = ob['viewpoint']
                candidate = ob['candidate']
                current_state_id, current_state = smallest_idXstate[i]
                old_viewpoint, from_action = decompose_state_id(current_state_id)
                assert ob['viewpoint'] == current_state['next_viewpoint']
                if from_action == -1 or ended[i]:       # If the action is <end> or the batch is ended, skip it
                    continue
                for j in range(len(ob['candidate']) + 1):               # +1 to include the <end> action
                    # score + log_prob[action]
                    modified_log_prob = log_probs[i][j].detach().cpu().item()
                    new_score = current_state['score'] + modified_log_prob
                    if j < len(candidate):                        # A normal action
                        next_id = make_state_id(current_viewpoint, j)
                        next_viewpoint = candidate[j]['viewpointId']
                        trg_point = candidate[j]['pointId']
                        heading = (trg_point % 12) * math.pi / 6
                        elevation = (trg_point // 12 - 1) * math.pi / 6
                        location = (next_viewpoint, heading, elevation)
                    else:                                                 # The end action
                        next_id = make_state_id(current_viewpoint, -1)    # action is -1
                        next_viewpoint = current_viewpoint                # next viewpoint is still here
                        location = (current_viewpoint, ob['heading'], ob['elevation'])

                    if next_id not in id2state[i] or new_score > id2state[i][next_id]['score']:
                        id2state[i][next_id] = {
                            "next_viewpoint": next_viewpoint,
                            "location": location,
                            "running_state": (h_t[i], h1[i], c_t[i]),
                            "from_state_id": current_state_id,
                            "feature": (f_t[i].detach().cpu(), candidate_feat[i][j].detach().cpu()),
                            "score": new_score,
                            "scores": current_state['scores'] + [modified_log_prob],
                            "actions": current_state['actions'] + [len(candidate)+1],
                        }

            # The active state is zero after the updating, then setting the ended to True
            for i in range(batch_size):
                if len(visited[i]) == len(id2state[i]):     # It's the last active state
                    ended[i] = True

            # End?
            if ended.all():
                break

        # Move back to the start point
        for i in range(batch_size):
            results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], results[i]['dijk_path'][0]))
        """
            "paths": {
                "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        """
        # Gather the Path
        for i, result in enumerate(results):
            assert len(finished[i]) <= args.candidates
            for state_id in finished[i]:
                path_info = {
                    "trajectory": [],
                    "action": [],
                    "listener_scores": id2state[i][state_id]['scores'],
                    "listener_actions": id2state[i][state_id]['actions'],
                    "visual_feature": []
                }
                viewpoint, action = decompose_state_id(state_id)
                while action != -95:
                    state = id2state[i][state_id]
                    path_info['trajectory'].append(state['location'])
                    path_info['action'].append(action)
                    path_info['visual_feature'].append(state['feature'])
                    state_id = id2state[i][state_id]['from_state_id']
                    viewpoint, action = decompose_state_id(state_id)
                state = id2state[i][state_id]
                path_info['trajectory'].append(state['location'])
                for need_reverse_key in ["trajectory", "action", "visual_feature"]:
                    path_info[need_reverse_key] = path_info[need_reverse_key][::-1]
                result['paths'].append(path_info)

        return results

    def beam_search(self, speaker):
        """
        :param speaker: The speaker to be used in searching.
        :return:
        {
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "speaker_scores": [log_prob_word1, log_prob_word2, ..., ],
            }]
        }
        """
        self.env.reset()
        results = self._dijkstra()
        """
        return from self._dijkstra()
        [{
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                    "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }]
        }]
        """

        # Compute the speaker scores:
        for result in results:
            lengths = []
            num_paths = len(result['paths'])
            for path in result['paths']:
                assert len(path['trajectory']) == (len(path['visual_feature']) + 1)
                lengths.append(len(path['visual_feature']))
            max_len = max(lengths)
            img_feats = torch.zeros(num_paths, max_len, 36, self.feature_size + args.angle_feat_size)
            can_feats = torch.zeros(num_paths, max_len, self.feature_size + args.angle_feat_size)
            for j, path in enumerate(result['paths']):
                for k, feat in enumerate(path['visual_feature']):
                    img_feat, can_feat = feat
                    img_feats[j][k] = img_feat
                    can_feats[j][k] = can_feat
            img_feats, can_feats = img_feats.to(device), can_feats.to(device)
            features = ((img_feats, can_feats), lengths)
            insts = np.array([result['instr_encoding'] for _ in range(num_paths)])
            seq_lengths = np.argmax(insts == self.tok.word_to_index['<EOS>'], axis=1)   # len(seq + 'BOS') == len(seq + 'EOS')
            insts = torch.from_numpy(insts).to(device)
            speaker_scores = speaker.teacher_forcing(train=True, features=features, insts=insts, for_listener=True)
            for j, path in enumerate(result['paths']):
                path.pop("visual_feature")
                path['speaker_scores'] = -speaker_scores[j].detach().cpu().numpy()[:seq_lengths[j]]
        return results

    def beam_search_test(self, speaker):
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        looped = False
        self.results = {}
        while True:
            for traj in self.beam_search(speaker):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj
            if looped:
                break

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.critic.train()
            self.encoder_pos.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.critic.eval()
            self.encoder_pos.eval()
            if args.discriminator:
                self.discriminator.eval()
            if args.load_visual is not None:
                self.visual_encoder.eval()
                if args.visual_combine:
                    self.visual_combine.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

        if not args.finetune:
            self.encoder.eval()

        if not args.tune_visual and args.load_visual is not None:
            self.visual_encoder.eval()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.encoder_pos.parameters(), 40.)

        if args.finetune:
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            self.encoder_optimizer.step()
            if self.encoder_scheduler:
                self.encoder_scheduler.step()
        self.encoder_pos_optimizer.step()
        self.encoder_pos_scheduler.step()
        self.decoder_optimizer.step()
        self.critic_optimizer.step()
        if args.discriminator:
            self.discriminator_optimizer.step()
            if args.dis_optim == 'adam':
                self.discriminator_scheduler.step()

        if args.load_visual is not None and args.tune_visual:
            self.visual_optimizer.step()

        if args.visual_combine:
            torch.nn.utils.clip_grad_norm(self.visual_combine.parameters(), 40.)
            self.visual_combine_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.decoder.train()
        self.critic.train()
        if args.finetune:
            self.encoder.train()
            self.encoder_pos.train()
        else:
            self.encoder.eval()
            self.encoder_pos.train()

        if args.discriminator:
            self.discriminator.train()

        if args.load_visual is not None and args.tune_visual:
            self.visual_encoder.train()
        elif args.load_visual is not None and not args.tune_visual:
            self.visual_encoder.eval()
        else:
            pass

        if args.visual_combine:
            self.visual_combine.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.encoder_pos_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            if args.finetune:
                self.encoder_optimizer.zero_grad()

            if args.discriminator:
                self.discriminator_optimizer.zero_grad()

            if args.load_visual is not None and args.tune_visual:
                self.visual_optimizer.zero_grad()

            if args.visual_combine:
                self.visual_combine_optimizer.zero_grad()

            self.loss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.encoder_pos.parameters(), 40.)

            if args.finetune:
                torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
                self.encoder_optimizer.step()
                if self.encoder_scheduler:
                    self.encoder_scheduler.step()
            self.encoder_pos_optimizer.step()
            self.encoder_pos_scheduler.step()
            self.decoder_optimizer.step()
            self.critic_optimizer.step()
            if args.discriminator:
                self.discriminator_optimizer.step()
                if args.dis_optim == 'adam':
                    self.discriminator_scheduler.step()

            if args.load_visual is not None and args.tune_visual:
                self.visual_optimizer.step()

            if args.visual_combine:
                torch.nn.utils.clip_grad_norm(self.visual_combine.parameters(), 40.)
                self.visual_combine_optimizer.step()

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
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        # if args.finetune:
        all_tuple.append(("encoder_pos", self.encoder_pos, self.encoder_pos_optimizer))
        if args.discriminator:
            all_tuple.append(("discriminator", self.discriminator, self.discriminator_optimizer))
        if args.load_visual is not None:
            all_tuple.append(("visual_encoder", self.visual_encoder, self.visual_optimizer))
        if args.visual_combine:
            all_tuple.append(("visual_combine", self.visual_combine, self.visual_combine_optimizer))
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
                optimizer.load_state_dict(states[name]['optimizer'])

        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]

        # if args.finetune:
        all_tuple.append(("encoder_pos", self.encoder_pos, self.encoder_pos_optimizer))
        if args.discriminator:
            all_tuple.append(("discriminator", self.discriminator, self.discriminator_optimizer))
        if args.load_visual is not None:
            all_tuple.append(("visual_encoder", self.visual_encoder, self.visual_optimizer))
        if args.visual_combine:
            all_tuple.append(("visual_combine", self.visual_combine, self.visual_combine_optimizer))
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

    def load_encoder(self, path):
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
                optimizer.load_state_dict(states[name]['optimizer'])

        recover_state("encoder", self.encoder, self.encoder_optimizer)
        return states['encoder']['epoch'] - 1

    def load_visual(self, path):
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("model",model_keys)
                print("load",load_keys)
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])

        recover_state("visual_encoder", self.visual_encoder, self.visual_optimizer)
        return states['visual_encoder']['epoch'] - 1
