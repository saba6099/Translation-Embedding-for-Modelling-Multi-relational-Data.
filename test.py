import numpy as np
from bigdl.nn.criterion import *
import pandas as pd
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import Sample
from bigdl.dataset.transformer import *
import numpy as np
import time
import random
from random import sample
from pyspark import SparkContext, SparkConf
init_engine()





class TransE:
    def __init__(self, entity_list, rels_list, triplets_list, margin=1, learning_rate=0.01, dim=50, normal_form="L1"):
        self.learning_rate = learning_rate
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

    def sample(self, size):
        return sample(self.triplets_list, size)

    def makeRDD(self, sc):
        data = (np.concatenate((self.batch_pos, self.batch_neg), axis = 1))
        print(data)
        train_rdd = sc.parallelize(data)
        print(train_rdd.collect())


    def transE(self, cycle_index=20):
        count = 0
        print("\n********** Start TransE training **********")
        for i in range(cycle_index):

            if count == 0:
                start_time = time.time()
            count += 1
            print("Epoch number:", count)
            Sbatch = self.sample(4)

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
        print((self.batch_pos))
        print((self.batch_neg))

    def createrdd(self,sc):
        #sc = SparkContext('spark://Heena:7077')
        # trainrdd = sc.parallelize(self.batch_pos)
        #tt = sc.textFile("/home/heena/PycharmProjects/BigDLSample/dummydata/entity2id.txt")
        #print(tt.take(2))
        #print(np.array(self.batch_pos))
        #b=['{} {}'.format(*i) for i in zip(np.array(self.batch_pos),np.array(self.batch_neg))]
        #print(b)

        train = list(map(lambda x, y: (x,y), np.array(self.batch_pos),np.array(self.batch_neg)))
        print(train)
        #train_rdd = sc.parallelize(train)
        #print(train_rdd.take(3))

        #
        # print(3))
        # #sc.stop()





if __name__ == "__main__":
    conf=SparkConf().setAppName('test').setMaster('spark://saba-Aspire-VN7-591G:7077')
    sc = SparkContext.getOrCreate(conf)
    entity2id = pd.read_table("/home/saba/Documents/TransE/dummydata/entity2id.txt", header=None)
    dict_entities = dict(zip(entity2id[0], entity2id[1]))
    print(len(entity2id))
    relation_df = pd.read_table("/home/saba/Documents/TransE/dummydata/relation2id.txt", header=None)
    dict_relations = dict(zip(relation_df[0], relation_df[1]))
    print(len(relation_df))
    training_df = pd.read_table("/home/saba/Documents/TransE/dummydata/train.txt", header=None)
    training_triples = list(zip([dict_entities[h] for h in training_df[0]],
                                 [dict_entities[t] for t in training_df[1]],
                                  [dict_relations[r] for r in training_df[2]]))

    transE = TransE(dict_entities, dict_relations, training_triples, margin=1, dim=50)
    transE.transE(1)
    transE.makeRDD(sc)
    sc.stop()