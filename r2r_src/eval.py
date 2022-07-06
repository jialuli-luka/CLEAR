''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs, load_datasets_visual
from agent import BaseAgent
from param import args


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans, tok):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for split in splits:
            if args.train == 'visual':
                load_func = load_datasets_visual
            else:
                load_func = load_datasets
            for item in load_func([split]):
                if scans is not None and item['scan'] not in scans:
                    continue
                if args.dataset == 'RxR':
                    if args.train == 'visual' and item['pair_path'] == item['path']:  # ignore data
                        continue
                    language = item['language'].split('-')[0]
                    self.gt[str(item['path_id'])] = item
                    self.scans.append(item['scan'])
                    self.instr_ids += [item['instruction_id']]
                if args.dataset == 'CVDN':
                    self.gt[item['inst_idx']] = item
                    self.instr_ids.append(item['inst_idx'])
                    self.scans.append(item['scan'])

                    planner_goal = item['planner_path'][-1]
                    if planner_goal in item['player_path'][1:]:
                        self.gt[item['inst_idx']]['path'] = item['player_path'][:]
                    else:
                        self.gt[item['inst_idx']]['path'] = item['planner_path'][:]
                else:
                    self.scans.append(item['scan'])
                    self.gt[str(item['path_id'])] = item
                    if 'instructions' not in item:
                        self.instr_ids += [item['instruction_id']]
                    else:
                        self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]

        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def ndtw(self, scan, prediction, reference):
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction) + 1):
            for j in range(1, len(reference) + 1):
                best_previous_cost = min(dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])
                cost = self.distances[scan][prediction[i - 1][0]][reference[j - 1]]
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]
        ndtw = np.exp(-dtw / (self.error_margin * len(reference)))
        return ndtw

    def length(self, scan, nodes):
        return float(np.sum([self.distances[scan][edge[0]][edge[1]] for edge in zip(nodes[:-1], nodes[1:])]))

    def cls(self, scan, prediction, reference):
        predict = [p[0] for p in prediction]
        prediction = predict
        coverage = np.mean(
            [np.exp(-np.min([self.distances[scan][u][v] for v in prediction]) / self.error_margin) for u in reference])
        expected = coverage * self.length(scan, reference)
        score = expected / (expected + np.abs(expected - self.length(scan, prediction)))
        return coverage * score

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        # gt = self.gt[instr_id.split('_')[-2]]
        # gt = self.gt[str(instr_id)]
        if args.dataset == 'RxR':
            gt = self.gt[str(instr_id)]
        else:
            gt = self.gt[instr_id.split('_')[-2]]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )
        self.scores['ndtw'].append(self.ndtw(gt['scan'], path, gt['path']))
        self.scores['cls'].append(self.cls(gt['scan'], path, gt['path']))
        self.scores['success_rate'].append(1 if self.scores['nav_errors'][-1] < self.error_margin else 0)
        self.scores['sdtw'].append(self.scores['ndtw'][-1] * self.scores['success_rate'][-1])

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                if args.dataset == 'RxR':
                    self._score_item(item['path_id'], item['trajectory'])
                else:
                    self._score_item(item['instr_id'], item['trajectory'])
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths']),
            'cls': np.average(self.scores['cls']),
            'ndtw': float(sum(self.scores['ndtw']) / len(self.scores['ndtw'])),
            'sdtw': float(sum(self.scores['sdtw']) / len(self.scores['sdtw']))
        }
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))

        spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
            for error, p, l in
            zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['spl'] = np.average(spl)

        return score_summary, self.scores

    def _score_item_environment(self, instr_id, path):
        if args.dataset == 'RxR':
            gt = self.gt[str(instr_id)]
        else:
            gt = self.gt[instr_id.split('_')[-2]]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        scan = gt['scan']
        self.scores[scan]['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores[scan]['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores[scan]['trajectory_steps'].append(len(path)-1)
        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores[scan]['trajectory_lengths'].append(distance)
        self.scores[scan]['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )
        self.scores[scan]['ndtw'].append(self.ndtw(gt['scan'], path, gt['path']))
        self.scores[scan]['cls'].append(self.cls(gt['scan'], path, gt['path']))
        self.scores[scan]['success_rate'].append(1 if self.scores[scan]['nav_errors'][-1] < self.error_margin else 0)
        self.scores[scan]['sdtw'].append(self.scores[scan]['ndtw'][-1] * self.scores[scan]['success_rate'][-1])

    def score_environment(self, output_file):
        self.scores = dict()
        for scan in self.scans:
            self.scores[scan] = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                if args.dataset == 'RxR':
                    self._score_item_environment(item['path_id'], item['trajectory'])
                else:
                    self._score_item_environment(item['instr_id'], item['trajectory'])
        if 'train' not in self.splits:  # Exclude the training from this. (Because training eval may be partial)
            assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s' \
                                        % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
            total = 0
            for scan in self.scans:
                total += len(self.scores[scan]['nav_errors'])
            assert total == len(self.instr_ids)
        score_summary = dict()
        for scan in self.scans:
            score_summary[scan] = defaultdict(list)
        for scan in self.scans:
            score_summary[scan] = {
            'nav_error': np.average(self.scores[scan]['nav_errors']),
            'oracle_error': np.average(self.scores[scan]['oracle_errors']),
            'steps': np.average(self.scores[scan]['trajectory_steps']),
            'lengths': np.average(self.scores[scan]['trajectory_lengths']),
            'cls': np.average(self.scores[scan]['cls']),
            'ndtw': float(sum(self.scores[scan]['ndtw']) / len(self.scores[scan]['ndtw'])),
            'sdtw': float(sum(self.scores[scan]['sdtw']) / len(self.scores[scan]['sdtw']))
            }
            num_successes = len([i for i in self.scores[scan]['nav_errors'] if i < self.error_margin])
            score_summary[scan]['success_rate'] = float(num_successes) / float(len(self.scores[scan]['nav_errors']))
            oracle_successes = len([i for i in self.scores[scan]['oracle_errors'] if i < self.error_margin])
            score_summary[scan]['oracle_rate'] = float(oracle_successes) / float(len(self.scores[scan]['oracle_errors']))

            spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
                   for error, p, l in
                   zip(self.scores[scan]['nav_errors'], self.scores[scan]['trajectory_lengths'], self.scores[scan]['shortest_lengths'])
                   ]
            score_summary[scan]['spl'] = np.average(spl)

        return score_summary, self.scores

    def _score_item_cvdn(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id)]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        planner_goal = gt['planner_path'][
            -1]  # for calculating oracle planner success (e.g., passed over desc goal?)
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        nearest_planner_position = self._get_nearest(gt['scan'], planner_goal, path)
        dist_to_end_start = None
        dist_to_end_end = None
        for end_pano in gt['end_panos']:
            d = self.distances[gt['scan']][start][end_pano]
            if dist_to_end_start is None or d < dist_to_end_start:
                dist_to_end_start = d
            d = self.distances[gt['scan']][final_position][end_pano]
            if dist_to_end_end is None or d < dist_to_end_end:
                dist_to_end_end = d
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['suc_length'].append(self.distances[gt['scan']][start][goal])
        self.scores['final_length'].append(self.distances[gt['scan']][start][final_position])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['oracle_plan_errors'].append(self.distances[gt['scan']][nearest_planner_position][planner_goal])
        self.scores['dist_to_end_reductions'].append(dist_to_end_start - dist_to_end_end)
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            if prev[0] != curr[0]:
                try:
                    self.graphs[gt['scan']][prev[0]][curr[0]]
                except KeyError as err:
                    print('Error: The provided trajectory moves from %s to %s but the navigation graph contains no ' \
                          'edge between these viewpoints. Please ensure the provided navigation trajectories ' \
                          'are valid, so that trajectory length can be accurately calculated.' % (prev[0], curr[0]))
                    raise
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal])
        self.scores['ndtw'].append(self.ndtw(gt['scan'], path, gt['path']))

    def score_cvdn(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)

        item_list = output_file
        for item in item_list:
            if item['inst_idx'] in instr_ids:
                instr_ids.remove(item['inst_idx'])
                self._score_item_cvdn(item['inst_idx'], item['trajectory'])
                item['dist_to_end_reductions'] = self.scores['dist_to_end_reductions'][-1]
                item['ndtw'] = self.scores['ndtw'][-1]

        num_successes = 0.0
        suc_length = []
        for i, j in zip(self.scores['nav_errors'], self.scores['suc_length']):
            if i < self.error_margin:
                num_successes += 1
                suc_length.append(j)
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        oracle_plan_successes = len([i for i in self.scores['oracle_plan_errors'] if i < self.error_margin])

        spls = []
        for err, length, sp in zip(self.scores['nav_errors'], self.scores['trajectory_lengths'],
                                   self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                if sp > 0:
                    spls.append(sp / max(length, sp))
                else:  # In IF, some Q/A pairs happen when we're already in the goal region, so taking no action is correct.
                    spls.append(1 if length == 0 else 0)
            else:
                spls.append(0)

        score_summary = {
            'suc_length': np.average(suc_length),
            'final_length': np.average(self.scores['final_length']),
            'length': np.average(self.scores['trajectory_lengths']),
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle success_rate': float(oracle_successes) / float(len(self.scores['oracle_errors'])),
            'success_rate': float(num_successes) / float(len(self.scores['nav_errors'])),
            'spl': np.average(spls),
            'oracle path_success_rate': float(oracle_plan_successes) / float(len(self.scores['oracle_plan_errors'])),
            'dist_to_end_reduction': sum(self.scores['dist_to_end_reductions']) / float(
                len(self.scores['dist_to_end_reductions'])),
            'ndtw': float(sum(self.scores['ndtw']) / len(self.scores['ndtw']))
        }

        assert score_summary['spl'] <= score_summary['success_rate']
        return score_summary, self.scores

    def bleu_score(self, path2inst):
        from bleu import compute_bleu
        refs = []
        candidates = []
        for path_id, inst in path2inst.items():
            path_id = str(path_id)
            assert path_id in self.gt
            # There are three references
            refs.append([self.tok.split_sentence(sent) for sent in self.gt[path_id]['instructions']])
            candidates.append([self.tok.index_to_word[word_id] for word_id in inst])

        tuple = compute_bleu(refs, candidates, smooth=False)
        bleu_score = tuple[0]
        precisions = tuple[1]

        return bleu_score, precisions


RESULT_DIR = 'tasks/R2R/results/'

def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen', 'test']:
        env = R2RBatch(None, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    eval_simple_agents()
