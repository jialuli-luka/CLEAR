
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import time
import json
import numpy as np
from collections import defaultdict
from mbert import mBERT
from visual import Visual

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features, read_object_constraints
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter
from transformers import BertTokenizer

from transformers import BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup


log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

if args.feature_size == 2048:
    IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
elif args.feature_size == 512 and args.HAMT:
    IMAGENET_FEATURES = 'HAMT/datasets/R2R/features/pth_vit_base_patch32_224_clip.hdf5'
elif args.feature_size == 512:
    IMAGENET_FEATURES = 'img_features/CLIP-ViT-B-32-views.tsv'
elif args.feature_size == 768:
    IMAGENET_FEATURES = 'img_features/HAMT-768.hdf5'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES

if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + "-fast" + ext

feedback_method = args.feedback # teacher or sample

print(args)

def train(train_env, train_eval, tok, n_iters, log_every=500, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction, eval=train_eval)

    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)

    start_iter = 0

    if args.load_encoder is not None:
        print("LOAD THE encoder from %s" % args.load_encoder)
        listner.load_encoder(os.path.join(args.load_encoder))

    if args.load_visual is not None and not args.visual_random:
        print("LOAD THE visual encoder from %s" % args.load_visual)
        listner.load_visual(os.path.join(args.load_visual))

    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))
        listner.encoder_scheduler = get_linear_schedule_with_warmup(listner.encoder_optimizer, 0.2 * args.iters, args.iters,
                                                                 last_epoch=start_iter)
        listner.encoder_pos_scheduler = get_linear_schedule_with_warmup(listner.encoder_pos_optimizer, 0.2 * args.iters,
                                                                     args.iters, last_epoch=start_iter)

    start = time.time()

    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
    best_val_ndtw =  {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
    best_val_gp = {'val_seen': {"accu": 0., "state": "", 'update': False},
                     'val_unseen': {"accu": 0., "state": "", 'update': False}}
    if args.fast_train:
        log_every = 40

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if args.accumulate_grad:
            listner.env = train_env
            for _ in range(interval):
                listner.zero_grad()
                listner.accumulate_gradient(feedback_method)
                listner.accumulate_gradient(feedback_method)
                listner.optim_step()
        else:
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)   # Train interval iters

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
        dis_loss = sum(listner.logs['dis_loss']) / total
        con_loss = sum(listner.logs['con_loss']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/unsupervised", predict_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        writer.add_scalar("discriminator", dis_loss, idx)
        writer.add_scalar("loss/contrastive", con_loss, idx )
        print("total_actions", total)
        print("max_length", length)

        # Run validation
        loss_str = ""
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

            listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result = listner.get_results()
            if args.dataset == 'CVDN':
                score_summary, _ = evaluator.score_cvdn(result)
            else:
                score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                if metric in ['ndtw']:
                    writer.add_scalar("ndtw/%s" % env_name, val, idx)
                    if env_name in best_val_ndtw:
                        if val > best_val_ndtw[env_name]['accu']:
                            best_val_ndtw[env_name]['accu'] = val
                            best_val_ndtw[env_name]['update'] = True
                if metric in ['dist_to_end_reduction']:
                    writer.add_scalar("GP/%s" % env_name, val, idx)
                    if env_name in best_val_gp:
                        if val > best_val_gp[env_name]['accu']:
                            best_val_gp[env_name]['accu'] = val
                            best_val_gp[env_name]['update'] = True
                loss_str += ', %s: %.3f' % (metric, val)

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

        for env_name in best_val_ndtw:
            if best_val_ndtw[env_name]['update']:
                best_val_ndtw[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val_ndtw[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s_ndtw" % (env_name)))

        if args.dataset == 'CVDN':
            for env_name in best_val_gp:
                if best_val_gp[env_name]['update']:
                    best_val_gp[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val_gp[env_name]['update'] = False
                    listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s_gp" % (env_name)))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 5000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])
            for env_name in best_val_ndtw:
                print(env_name, best_val_ndtw[env_name]['state'])
            if args.dataset == 'CVDN':
                for env_name in best_val_gp:
                    print(env_name, best_val_gp[env_name]['state'])

        if iter % 50000 == 0:
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        sr = []
        ndtw = []
        sdtw = []
        spl = []
        weights = []
        if env_name != 'test':
            if args.eval_environment:
                score_summary, scores = evaluator.score_environment(result)
                loss_str = "Env name: %s" % env_name
                for scan, value in score_summary.items():
                    loss_str = loss_str + '/n' + scan
                    for metric, val in value.items():
                        if metric == 'success_rate':
                            sr.append(val)
                        elif metric == 'ndtw':
                            ndtw.append(val)
                        elif metric == 'sdtw':
                            sdtw.append(val)
                        elif metric == 'spl':
                            spl.append(val)
                        loss_str += ', %s: %.4f' % (metric, val)
                    weights.append(len(scores[scan]['nav_errors']))
                print(loss_str)
                mean_sr = np.average(np.array(sr), weights=weights)
                std_sr = np.average((np.array(sr) - mean_sr) ** 2, weights=weights)
                mean_ndtw = np.average(np.array(ndtw), weights=weights)
                std_ndtw = np.average((np.array(ndtw) - mean_ndtw) ** 2, weights=weights)
                mean_sdtw = np.average(np.array(sdtw), weights=weights)
                std_sdtw = np.average((np.array(sdtw) - mean_sdtw) ** 2, weights=weights)
                mean_spl = np.average(np.array(spl), weights=weights)
                std_spl = np.average((np.array(spl) - mean_spl) ** 2, weights=weights)
                print("SR:", mean_sr, std_sr ** (1 / 2))
                print("SPL:", mean_spl, std_spl ** (1 / 2))
                print("NDTW:", mean_ndtw, std_ndtw ** (1 / 2))
                print("SDTW:", mean_sdtw, std_sdtw ** (1 / 2))
                print(weights)
            else:
                if args.dataset == 'CVDN':
                    score_summary, _ = evaluator.score_cvdn(result)
                else:
                    score_summary, _ = evaluator.score(result)
                loss_str = "Env name: %s" % env_name
                for metric,val in score_summary.items():
                    loss_str += ', %s: %.4f' % (metric, val)
                print(loss_str)

        if True:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def cleanup():
    dist.destroy_process_group()


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir="./cache/")

    feat_dict = read_img_features(features)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    train_eval = Evaluation(['train'], featurized_scans, tok)
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']

    if not args.beam:
        val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.submit:
        val_envs['test'] = (R2RBatch(feat_dict, batch_size=args.batchSize, splits=['test'], tokenizer=tok), None)

    if args.train == 'listener':
        train(train_env, train_eval, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, val_envs=val_envs)
        else:
            valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    elif args.train == 'tsne':
        tsne_representation(train_env, tok, val_envs=val_envs)
    else:
        assert False


def train_mbert():
    setup()

    tok = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir="cache")

    feat_dict = read_img_features(features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)

    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    mbert(train_env, tok, args.iters, val_envs=val_envs)


def mbert(train_env, tok, n_iters, log_every=100, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    encoder = mBERT(train_env, "", tok, args.maxAction)

    start_iter = 0

    start = time.time()

    best_loss = {'val_seen': {"loss": 10000000000., "state": ""},
                'val_unseen': {"loss": 10000000000., "state": ""},
                 'train': {"loss": 1000000000., "state": ""}}

    for idx in range(start_iter, start_iter + n_iters, log_every):
        encoder.logs = defaultdict(list)
        interval = min(log_every, n_iters - idx)
        iter = idx + interval

        encoder.env = train_env
        encoder.train(interval, feedback=feedback_method)

        # Log the training stats to tensorboard
        con_loss = sum(encoder.logs['con_loss']) / (args.batchSize * log_every)

        writer.add_scalar("loss/contrastive", con_loss, idx)

        # Run validation
        loss_str = ""

        for env_name, (env, evaluator) in val_envs.items():
            encoder.env = env

            iters = None if env_name != 'train' else 20  # 20 * 64 = 1280

            result = encoder.test(use_dropout=False, iters=iters)
            loss_str += ", %s " % env_name
            loss_str += ', %s: %.3f' % ("contrastive loss", result)

            writer.add_scalar("loss/%s" % (env_name), result, idx)

            if result < best_loss[env_name]["loss"]:
                best_loss[env_name]["loss"] = result
                best_loss[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                print('Save the model with %s BEST env loss %0.4f' % (env_name, result))
                encoder.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                    iter, float(iter) / n_iters * 100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_loss:
                print(env_name, best_loss[env_name]['state'])

        if iter % 50000 == 0:
            encoder.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

    encoder.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def train_visual():
    setup()

    tok = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir="./cache/")

    if args.HAMT:
        feat_dict_tmp = read_img_features('img_features/CLIP-ViT-B-32-views.tsv')
        feat_dict = dict()
        import h5py
        with h5py.File(features, 'r') as f1:
            for key, v in feat_dict_tmp.items():
                fts = f1[key][...].astype(np.float32)[:, :args.feature_size]
                feat_dict[key] = fts
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    else:
        feat_dict = read_img_features(features)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    if args.objects_constraints:
        objects = read_object_constraints(args.objects_constraints)
        train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok, objects=objects)

        val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                     tokenizer=tok, objects=objects), Evaluation([split], featurized_scans, tok))
                    for split in ['train', 'val_seen', 'val_unseen']}
    else:
        train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)

        val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                     tokenizer=tok), Evaluation([split], featurized_scans, tok))
                    for split in ['train', 'val_seen', 'val_unseen']}

    visual(train_env, tok, args.iters, val_envs=val_envs)


def visual(train_env, tok, n_iters, log_every=100, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    encoder = Visual(train_env, "", tok, args.maxAction)

    start_iter = 0

    start = time.time()

    if args.load_encoder is not None:
        print("LOAD THE encoder from %s" % args.load_encoder)
        encoder.load_encoder(os.path.join(args.load_encoder))

    if args.load is not None and not args.visual_random:
        print("LOAD THE visual encoder from %s" % args.load)
        start_iter = encoder.load(os.path.join(args.load))

    best_loss = {'val_seen': {"loss": 10000000000., "state": ""},
                'val_unseen': {"loss": 10000000000., "state": ""},
                 'train': {"loss": 1000000000., "state": ""}}

    for idx in range(start_iter, start_iter + n_iters, log_every):
        encoder.logs = defaultdict(list)
        interval = min(log_every, n_iters - idx)
        iter = idx + interval

        encoder.env = train_env
        encoder.train(interval, feedback=feedback_method)

        # Log the training stats to tensorboard
        con_loss = sum(encoder.logs['con_loss']) / (args.batchSize * log_every)

        writer.add_scalar("loss/contrastive", con_loss, idx)

        # Run validation
        loss_str = ""

        for env_name, (env, evaluator) in val_envs.items():
            encoder.env = env

            iters = None if env_name != 'train' else 20  # 20 * 64 = 1280

            result = encoder.test(use_dropout=False, iters=iters)
            loss_str += ", %s " % env_name
            loss_str += ', %s: %.3f' % ("contrastive loss", result)

            writer.add_scalar("loss/%s" % (env_name), result, idx)

            if result < best_loss[env_name]["loss"]:
                best_loss[env_name]["loss"] = result
                best_loss[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                print('Save the model with %s BEST env loss %0.4f' % (env_name, result))
                encoder.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                    iter, float(iter) / n_iters * 100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_loss:
                print(env_name, best_loss[env_name]['state'])

        if iter % 50000 == 0:
            encoder.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

    encoder.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'tsne':
        train_val_tsne()
    elif args.train == 'auglistener':
        train_val_augment()
    elif args.train == 'mbert':
        train_mbert()
    elif args.train == 'visual':
        train_visual()
    else:
        assert False
