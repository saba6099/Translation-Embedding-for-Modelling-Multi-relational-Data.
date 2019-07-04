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

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *

from pyspark import SparkContext, SparkConf
os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"


def create_model(total_embeddings, embedding_dim = 2, margin=1.0):
    model = Sequential()
    model.add(Reshape([6]))
    embedding = LookupTable(total_embeddings, embedding_dim)
    model.add(embedding)
    model.add(Reshape([2, 3, 1, 2])).add(Squeeze(1))
    # print(model.forward((train_data)))
    model.add(SplitTable(1))

    branches = ParallelTable()
    branch1 = Sequential()
    pos_h_l = Sequential().add(ConcatTable().add(Select(1, 1)).add(Select(1, 3)))
    pos_add = pos_h_l.add(CAddTable())
    pos_t = Sequential().add(Select(1, 2)).add(MulConstant(-1.0))
    triplepos_meta = Sequential().add(ConcatTable().add(pos_add).add(pos_t))
    triplepos_dist = triplepos_meta.add(CAddTable()).add(Abs())
    triplepos_score = triplepos_dist.add(Unsqueeze(1)).add(Mean(3, 1)).add(MulConstant(float(embedding_dim)))
    branch1.add(triplepos_score)  # .add(AddConstant(1.0))#.add(Unsqueeze(1))
    # pos_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
    # a = Sequential().add(Narrow(1,1,2)).add(SplitTable(2))
    # c = Sequential().add(CAveTable())
    # b = Sequential().add(Select(1,2))

    # branch1 = Sequential().add(CAveTable()).add(MulConstant(1.0))
    # branch1 = Sequential().add(c)

    branch2 = Sequential()
    neg_h_l = Sequential().add(ConcatTable().add(Select(1, 1)).add(Select(1, 3)))
    neg_add = neg_h_l.add(CAddTable())
    neg_t = Sequential().add(Select(1, 2)).add(MulConstant(-1.0))
    tripleneg_meta = Sequential().add(ConcatTable().add(neg_add).add(neg_t))
    tripleneg_dist = tripleneg_meta.add(CAddTable()).add(Abs())
    tripleneg_score = tripleneg_dist.add(Unsqueeze(1)).add(Mean(3, 1)).add(MulConstant(float(embedding_dim)))
    branch2.add(tripleneg_score)  # .add(Unsqueeze(1))
    # pos_add= branch1.add(CAddTable())
    # branch2 = Sequential().add(Narrow(1, 2))

    # pos_h_l = Sequential().add(ConcatTable().add(Select(2,1)).add(Select(2,3)))
    # pos_add= pos_h_l.add(CAddTable())
    # branch1.add(pos_add)

    #
    # branch2 = Sequential().add(Select(2,2))
    branches.add(branch1).add(branch2)
    model.add(branches)
    # model.add(CAveTable())
    # print(model.forward(train_data))
    # model.add(a)
    pos_plus_margin = Sequential().add(SelectTable(1)).add(AddConstant(margin))

    model.add(ConcatTable().add(pos_plus_margin).add(SelectTable(2))).add(CSubTable()).add(Abs())
    return model


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
        self.batch_total = []



    def generate_corrupted_triplets(self, cycle_index=1):
        count = 0
        print("\n********** Start TransE training **********")
        for i in range(cycle_index):

            if count == 0:
                start_time = time.time()
            count += 1
            print("Batch number:", count)
            Sbatch = self.triplets_list
            #Sbatch=self.

            raw_batch = Sbatch
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                batch_total = []
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
                    batch_neg = [[head_neg], [tail_neg], [relation]]
                    batch_pos = [[head],[tail],[relation]]
                    batch_total.append([batch_pos, batch_neg])
            self.batch_neg += batch_neg
            self.batch_pos += batch_pos
            self.batch_total+=(batch_total)
        print(("batch_pos",self.batch_pos))
        print(("batch_neg",self.batch_neg))

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




    def make_samples(self,total_embeddings):
        sample = np.array(self.batch_total)
        sample_rdd = sc.parallelize(sample)
        labels = np.zeros(len(self.triplets_list))
        labels = sc.parallelize(labels)
        record = sample_rdd.zip(labels)
        train_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))
        print("Hello")


        optimizer = Optimizer(
            model = create_model(total_embeddings),
            training_rdd = train_data,
            criterion = AbsCriterion(False),
            optim_method = SGD(learningrate=0.01, learningrate_decay=0.0002),
            end_trigger = MaxEpoch(10),
            batch_size = 4)

        optimizer.optimize()


if __name__ == "__main__":
    conf=SparkConf().setAppName('test').setMaster('spark://saba-Aspire-VN7-591G:7077')
    sc = SparkContext.getOrCreate(conf)
    entities = pd.read_table("/home/saba/Documents/Big Data Lab/data/WN18/entity2id.txt", header=None)
    dict_entities = dict(zip(entities[0], entities[1]))
    print(len(entities))
    relations = pd.read_table("/home/saba/Documents/Big Data Lab/data/WN18/relation2id.txt", header=None)
    dict_relations = dict(zip(relations[0], relations[1]))
    print(len(relations))
    training_df = pd.read_table("/home/saba/Documents/Big Data Lab/data/WN18/train.txt", header=None)
    training_triples = list(zip([dict_entities[h] + 1 for h in training_df[0]],
                                 [dict_entities[t] + 1 for t in training_df[1]],
                                  [dict_relations[r]+len(entities) + 1 for r in training_df[2]]))

    transE = TransE(dict_entities, dict_relations, training_triples, margin=1, dim=50)
    transE.generate_corrupted_triplets(1)
    no_entities_relations=len(entities)+len(relations)
    print("type",(no_entities_relations))
    transE.make_samples(no_entities_relations)


    #transE.create_embeddings()
    #transE.createrdd(sc)
    #transE.makeRDD(sc)
    sc.stop()