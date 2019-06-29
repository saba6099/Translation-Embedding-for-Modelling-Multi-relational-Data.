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
# from bigdl.nn.keras.topology
from bigdl.nn.keras.layer import Embedding
init_engine()
import os
os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"




class TransE:
    def __init__(self, entity_dict, rels_dict, triplets_list, margin=1, learning_rate=0.01, dim=50, normal_form="L1"):
        self.learning_rate = learning_rate
        self.loss = 0
        self.entity_dict = entity_dict
        self.rels_dict = rels_dict
        self.triplets_list = triplets_list
        self.margin = margin
        self.dim = dim
        self.normal_form = normal_form
        self.entity_vector_dict = {}
        self.relation_vector_dict = {}
        self.loss_list = []
        self.training_triple_pool = set(triplets_list)
        self.batch_pos = []
        self.batch_neg = []
        self.distance_pos = []
        self.distance_neg = []
        self.entity_embeddings = []
        self.relation_embeddings = []
        self.embeddings = np.empty((12,2))
        self.entity = []
        self.relation = []
        self.triple_embeddings = []

    def sample(self, size):
        return sample(self.triplets_list, size)

    def makeRDD(self, sc):
        #
        # data = (np.concatenate((self.batch_pos, self.batch_neg), axis = 1))
        # print(data)
        train_rdd = sc.parallelize(self.triple_embeddings)
        print(train_rdd.collect())
        self.all_heads_true = train_rdd.map(lambda x: (x[0]))
        self.all_tails_true = train_rdd.map(lambda x: (x[1]))
        self.all_relations_true = train_rdd.map(lambda x: (x[2]))
        self.all_heads_corrupt = train_rdd.map(lambda x: (x[3]))
        self.all_tails_corrupt = train_rdd.map(lambda x: (x[4]))
        self.all_relations_corrupt = train_rdd.map(lambda x: (x[5]))

        print(self.all_relations_corrupt.collect())
        print("hello")
        #layer = LookupTable(9, 4, 2.0, 0.1, 2.0, True)
        #input = (np.array[train_rdd.collect()]).astype("float32")

        #output = layer.forward(input)
        #gradInput = layer.backward(input, output)
        #print(output)
        #print(gradInput)
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
                            head_neg = random.choice(list(self.entity_dict.values()))
                        else:
                            tail_neg = random.choice(list(self.entity_dict.values()))
                        if (head_neg, tail_neg, relation) not in (self.training_triple_pool and self.batch_neg):
                            break
                    batch_neg.append((head_neg, tail_neg, relation))
            self.batch_neg += batch_neg
            self.batch_pos += batch_pos
        print((self.batch_pos))
        print((self.batch_neg))

    def create_embeddings(self):
        self.entity  = list((self.entity_dict.values()))
        self.relation = list((self.rels_dict.values()))

        self.entity  = [x+1 for x in self.entity]
        self.relation  = [x+1 for x in self.relation]

        model = Sequential()
        len = 8
        # model.add(Embedding(len, 2, input_shape=(8,)))
        layer = LookupTable(8, 2, 2, 0.1, 2.0, True)
        model.add(layer)
        self.entity_embeddings = model.forward(np.array(list(self.entity)))
        print(self.entity_embeddings)
        for i in range((self.entity_embeddings).shape[0]):
            self.entity_vector_dict[i] = self.entity_embeddings[i]

        model = Sequential()
        len = 4
        model.add(Embedding(len, 2, input_shape=(4,)))
        # model.add(LookupTable(14, 2, 2, 0.1, 2.0, True))
        self.relation_embeddings = model.forward(np.array(list(self.relation)))
        print(self.relation_embeddings)
        for i in range((self.relation_embeddings).shape[0]):
            self.relation_vector_dict[i] = self.relation_embeddings[i]

        model = Sequential()
        branches = ParallelTable()
        branch1 = Sequential().add(Sum(2))
        branch2 = Sequential().add(Sum(2))
        branches.add(branch1).add(branch2)
        model.add(branches)

        output = model.forward([self.all_heads_true, self.all_tails_true, self.all_relations_true], [self.all_heads_corrupt, self.all_tails_corrupt, self.all_relations_corrupt])
        print(output)


    def createrdd(self,sc):
        #sc = SparkContext('spark://Heena:7077')
        # trainrdd = sc.parallelize(self.batch_pos)
        #tt = sc.textFile("/home/heena/PycharmProjects/BigDLSample/dummydata/entity2id.txt")
        #print(tt.take(2))
        #print(np.array(self.batch_pos))
        #b=['{} {}'.format(*i) for i in zip(np.array(self.batch_pos),np.array(self.batch_neg))]
        #print(b)

        # train = list(map(lambda x, y: (x,y), np.array(self.batch_pos),np.array(self.batch_neg)))
        # print(train)
        #train_rdd = sc.parallelize(train)
        #print(train_rdd.take(3))

        #
        # print(3))
        # #sc.stop()

        for i, j in zip(self.batch_pos, self.batch_neg):

            head_pos = self.entity_vector_dict.get(i[0])
            tail_pos = self.entity_vector_dict.get(i[1])
            relation_pos = self.relation_vector_dict.get(i[2])
            head_neg = self.entity_vector_dict.get(j[0])
            tail_neg = self.entity_vector_dict.get(j[1])
            relation_neg = self.relation_vector_dict.get(j[2])

            # a  = np.array([head_pos, tail_pos, relation_pos, head_neg, tail_neg, relation_neg])
            # self.batch_pos =  np.append([], [a], axis = 0)

            embed = [head_pos, tail_pos, relation_pos, head_neg, tail_neg, relation_neg]
            self.triple_embeddings.append(embed)


        print("Hello")






if __name__ == "__main__":
    conf=SparkConf().setAppName('test').setMaster('spark://saba-Aspire-VN7-591G:7077')
    sc = SparkContext.getOrCreate(conf)
    entity2id = pd.read_table("/home/saba/Documents/Big Data Lab/dummydata/entity2id.txt", header=None)
    dict_entities = dict(zip(entity2id[0], entity2id[1]))
    print(len(entity2id))
    relation_df = pd.read_table("/home/saba/Documents/Big Data Lab/dummydata/relation2id.txt", header=None)
    dict_relations = dict(zip(relation_df[0], relation_df[1]))
    print(len(relation_df))
    training_df = pd.read_table("/home/saba/Documents/Big Data Lab/dummydata/train.txt", header=None)
    training_triples = list(zip([dict_entities[h]+1 for h in training_df[0]],
                                 [dict_entities[t]+1 for t in training_df[1]],
                                  [dict_relations[r]+1 for r in training_df[2]]))

    transE = TransE(dict_entities, dict_relations, training_triples, margin=1, dim=50)
    transE.transE(1)
    transE.create_embeddings()
    transE.createrdd(sc)
    transE.makeRDD(sc)
    sc.stop()