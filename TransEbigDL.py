from pyspark import SparkContext
from bigdl.util.common import *
import pandas as pd
from random import uniform, sample, choice
import itertools
import re
from bigdl.nn.layer import *
import numpy as np
import time
import random

def norm(lyst):
    var = np.linalg.norm(lyst)
    i = 0
    while i < len(lyst):
        lyst[i] = lyst[i] / var
        i += 1
    # return list
    return np.array(lyst)


def dist_L1(h, t, l):
    s = h + l - t
    # dist = np.fabs(s).sum()
    return np.fabs(s).sum()


class TransE:
    def __init__(self, entity_list, rels_list, triplets_list, margin=1, learing_rate=0.01, dim=50, normal_form="L1"):
        self.learning_rate = learing_rate
        self.loss = 0
        self.entity_list = entity_list
        self.rels_list = rels_list
        self.triplets_list = triplets_list
        self.margin = margin
        self.dim = dim
        self.normal_form = normal_form
        self.entity_vector_dict = {}
        self.rels_vector_dict = {}
        self.loss_list = []
        self.training_triple_pool = set(triplets_list)
        self.batch_pos = []
        self.batch_neg = []
        self.distance_pos = []
        self.distance_neg = []
        # self.build_graph()


    def initialize(self):
        entity_vector_dict, rels_vector_dict = {}, {}
        entity_vector_compo_list, rels_vector_compo_list = [], []
        for item, dict, compo_list, name in zip(
                [self.entity_list.values(), self.rels_list.values()], [entity_vector_dict, rels_vector_dict],
                [entity_vector_compo_list, rels_vector_compo_list], ["entity_vector_dict", "rels_vector_dict"]):
            for entity_or_rel in item:
                n = 0
                compo_list = []
                while n < self.dim:
                    random = uniform(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5))
                    compo_list.append(random)
                    n += 1
                compo_list = norm(compo_list)
                dict[entity_or_rel] = compo_list
            print("The " + name + "'s initialization is over. It's number is %d." % len(dict))
        self.entity_vector_dict = entity_vector_dict
        self.rels_vector_dict = rels_vector_dict

    def transE(self, cycle_index=20):
        count = 0
        print("\n********** Start TransE training **********")
        for i in range(cycle_index):

            if count == 0:
                start_time = time.time()
            count += 1
            print("Epoch number:", count)
            Sbatch = self.sample(1500)

            raw_batch = Sbatch
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []

                for head, tail, relation in batch_pos:
                    corrupt_head_prob = np.random.binomial(1, 0.5)
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            head_neg = random.choice(list(self.entity_list.values()))
                        else:
                            tail_neg = random.choice(list(self.entity_list.values()))
                        if (head_neg, tail_neg, relation) not in (self.training_triple_pool and self.batch_neg):
                            break
                    batch_neg.append((head_neg, tail_neg, relation))
            self.batch_neg += batch_neg
            self.batch_pos += batch_pos
        print(len(self.batch_neg))

    def sample(self, size):
        return sample(self.triplets_list, size)

    def build_graph(self):
            self.infer(self.batch_pos, self.batch_neg)
            self.loss = self.calculate_loss(self.distance_pos, self.distance_neg, self.margin)
            # tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            # self.merge = tf.summary.merge_all()

    def infer(self, triple_pos, triple_neg):
        for i, j in zip(triple_pos, triple_neg):

            head_pos = self.entity_vector_dict.get(i[0])
            tail_pos = self.entity_vector_dict.get(i[1])
            relation_pos = self.entity_vector_dict.get(i[2])
            head_neg = self.entity_vector_dict.get(j[0])
            tail_neg = self.entity_vector_dict.get(j[1])
            relation_neg = self.entity_vector_dict.get(j[2])
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg

            self.distance_pos.append(distance_pos)
            self.distance_neg.append(distance_neg)


    def calculate_loss(self, distance_pos, distance_neg, margin):
        # score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
        # score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
        score_pos = np.sum(np.fabs(self.distance_pos), axis=1)
        score_neg = np.sum(np.fabs(self.distance_pos), axis=1)

        # loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
        # return loss


if __name__ == "__main__":
    entity2id = pd.read_table("/home/saba/Documents/TransE/data/FB15k/entity2id.txt", header=None)
    dict_entities = dict(zip(entity2id[0], entity2id[1]))
    print(len(entity2id))
    relation_df = pd.read_table("/home/saba/Documents/TransE/data/FB15k/relation2id.txt", header=None)
    dict_relations = dict(zip(relation_df[0], relation_df[1]))
    print(len(relation_df))
    training_df = pd.read_table("/home/saba/Documents/TransE/data/FB15k/train.txt", header=None)
    training_triples = list(zip([dict_entities[h] for h in training_df[0]],
                                 [dict_entities[t] for t in training_df[1]],
                                  [dict_relations[r] for r in training_df[2]]))

    transE = TransE(dict_entities, dict_relations, training_triples, margin=1, dim=50)
    print("\nTransE is initializing...")
    transE.initialize()
    transE.transE(20)
    transE.build_graph()
    print("Hello")
