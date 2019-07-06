import numpy as np
from bigdl.nn.criterion import *
import pandas as pd
from bigdl.util.common import *
import time
import random
from bigdl.optim.optimizer import *

init_engine()
import os

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *

from pyspark import SparkContext, SparkConf
os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"


def create_model(total_embeddings, embedding_dim = 30, margin=1.0):
    model = Sequential()
    model.add(Reshape([6]))
    embedding = LookupTable(total_embeddings, embedding_dim)
    model.add(embedding)
    model.add(Reshape([2, 3, 1, embedding_dim])).add(Squeeze(1))
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

    branch2 = Sequential()
    neg_h_l = Sequential().add(ConcatTable().add(Select(1, 1)).add(Select(1, 3)))
    neg_add = neg_h_l.add(CAddTable())
    neg_t = Sequential().add(Select(1, 2)).add(MulConstant(-1.0))
    tripleneg_meta = Sequential().add(ConcatTable().add(neg_add).add(neg_t))
    tripleneg_dist = tripleneg_meta.add(CAddTable()).add(Abs())
    tripleneg_score = tripleneg_dist.add(Unsqueeze(1)).add(Mean(3, 1)).add(MulConstant(float(embedding_dim)))
    branch2.add(tripleneg_score)  # .add(Unsqueeze(1))

    branches.add(branch1).add(branch2)
    model.add(branches)

    pos_plus_margin = Sequential().add(SelectTable(1)).add(AddConstant(margin))

    model.add(ConcatTable().add(pos_plus_margin).add(SelectTable(2))).add(CSubTable()).add(Abs())
    return model


class TransE:
    def __init__(self, entity_dict, rels_dict, triplets_list, validation_triples,test_triples, margin=1, learning_rate=0.01, dim=50, normal_form="L1"):
        self.learning_rate = learning_rate
        self.loss = 0
        self.entity_dict = entity_dict
        self.rels_dict = rels_dict
        self.triplets_list = triplets_list
        self.validation_triples = validation_triples
        self.test_triples=test_triples
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
        self.total_embeddings = 0



    def generate_corrupted_triplets(self, type = "train", cycle_index=1):
        count = 0
        print("\n********** Start TransE training **********")
        for i in range(cycle_index):

            if count == 0:
                start_time = time.time()
            count += 1
            print("Batch number:", count)
            if(type == "train"):
                Sbatch = self.triplets_list
            else:
                Sbatch = self.validation

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

    def distance(self, head, tail, relation, corrupted_head, corrupted_tail, corrupted_relation):
        # sample=np.array([[[[head],[tail],[relation]],[[corrupted_head],[corrupted_tail],[corrupted_relation]]]])
        sample = np.array([[[[1],
                             [2],
                             [3]],
                            [[4],
                             [5],
                             [6]]],
                           [[[7],
                             [8],
                             [9]],
                            [[10],
                             [11],
                             [12]]],
                           [[[13],
                             [14],
                             [15]],
                            [[16],
                             [17],
                             [18]]],
                           [[[19],
                             [20],
                             [21]],
                            [[22],
                             [23],
                             [24]]],
                           [[[25],
                             [26],
                             [27]],
                            [[28],
                             [29],
                             [30]]],
                           [[[31],
                             [32],
                             [33]],
                            [[34],
                             [35],
                             [36]]],
                           [[[37],
                             [38],
                             [39]],
                            [[40],
                             [41],
                             [42]]],
                           [[[43],
                             [44],
                             [45]],
                            [[46],
                             [47],
                             [48]]]])
        # test_data = JTensor.from_ndarray(sample)
        sample_rdd = sc.parallelize(sample)
        labels = np.zeros(8)
        # labels=np.array([[1],[1],[1],[1]])
        labels = sc.parallelize(labels)
        record = sample_rdd.zip(labels)
        test_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))

        # model = self.make_samples()

        model = Model.loadModel("/home/saba/Documents/Big Data Lab/data/model.bigdl", "/home/saba/Documents/Big Data Lab/data/model.bin")
        result = model.evaluate(test_data, 8, [Top1Accuracy()])
        print(result)
        return result

    def generate_corrupted_test_triplets(self, cycle_index=1):
        batch_neg_head_replaced = {}
        batch_neg_tail_replaced = {}
        list_head_replace = []
        list_tail_replace = []
        count=0
        for i in range(cycle_index):

            if count == 0:
                start_time = time.time()
            count += 1
            print("Batch number:", count)
            Sbatch = self.test_triples
            # Sbatch=self.

            raw_batch = Sbatch
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                batch_total = []
                for head, tail, relation in batch_pos:
                    rank_list = {}
                    corrupt_entity_list = list(self.entity_dict.values())

                    for i in range(0, len(corrupt_entity_list)):
                        if (corrupt_entity_list[i], tail, relation) not in (self.training_triple_pool):
                            rank_list = self.distance(head, tail, relation, corrupt_entity_list[i], tail, relation)
                            # list_head_replace.append((corrupt_entity_list[i], tail, relation))


                        else:
                            continue
                    for i in range(0, len(corrupt_entity_list)):
                        if (head, corrupt_entity_list[i], relation) not in (self.training_triple_pool):
                            rank_list = self.distance(head, tail, relation, head, corrupt_entity_list[i], relation)
                            # list_tail_replace.append((head, corrupt_entity_list[i], relation ))

                        else:
                            continue
                    # batch_neg_head_replaced[(head, tail, relation)] = list_head_replace
                    # batch_neg_tail_replaced[(head, tail, relation)] = list_tail_replace
                    # list_tail_replace = []
                    # list_head_replace = []

        # print("batch_neg_head_replaced",batch_neg_head_replaced)

        # print("batch_neg_tail_replaced",batch_neg_tail_replaced )


        with open("/home/saba/Documents/Big Data Lab/data/FB15k/batch_neg_head_replaced.txt", 'w') as f:
            for key, value in batch_neg_head_replaced.items():
                f.write('%s:%s\n' % (key, value))
        with open("/home/saba/Documents/Big Data Lab/data/FB15k/batch_neg_tail_replaced.txt", 'w') as f:
            for key, value in batch_neg_head_replaced.items():
                f.write('%s:%s\n' % (key, value))

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
            end_trigger = MaxEpoch(1),
            batch_size = 4)

        trained_model = optimizer.optimize()

        sample = np.array([[[[1],
                             [2],
                             [3]],
                            [[4],
                             [5],
                             [6]]],
                           [[[7],
                             [8],
                             [9]],
                            [[10],
                             [11],
                             [12]]],
                           [[[13],
                             [14],
                             [15]],
                            [[16],
                             [17],
                             [18]]],
                           [[[19],
                             [20],
                             [21]],
                            [[22],
                             [23],
                             [24]]],
                           [[[25],
                             [26],
                             [27]],
                            [[28],
                             [29],
                             [30]]],
                           [[[31],
                             [32],
                             [33]],
                            [[34],
                             [35],
                             [36]]],
                           [[[37],
                             [38],
                             [39]],
                            [[40],
                             [41],
                             [42]]],
                           [[[43],
                             [44],
                             [45]],
                            [[46],
                             [47],
                             [48]]]])
        # test_data = JTensor.from_ndarray(sample)
        sample_rdd = sc.parallelize(sample)
        labels = np.zeros(8)
        # labels=np.array([[1],[1],[1],[1]])
        labels = sc.parallelize(labels)
        record = sample_rdd.zip(labels)
        test_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))

        #model = self.make_samples()

        #model = Model.loadModel("/home/saba/Documents/Big Data Lab/data/model.bigdl",
        #                        "/home/saba/Documents/Big Data Lab/data/model.bin")
        result = trained_model.predict(test_data)
        print("result",result)
        return trained_model
        # trained_model.saveModel("/home/saba/Documents/Big Data Lab/data/model.bigdl", "/home/saba/Documents/Big Data Lab/data/model.bin", True)
    #
    # def validate(self):
    #     sample = np.array(self.batch_total)
    #     sample_rdd = sc.parallelize(sample)
    #     labels = np.zeros(len(self.triplets_list))
    #     labels = sc.parallelize(labels)
    #     record = sample_rdd.zip(labels)
    #     train_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))
    #     print("Hello")
    #     optimizer.set_validation(batch_size, val_rdd, trigger, validationMethod)
    #


if __name__ == "__main__":
    conf=SparkConf().setAppName('test').setMaster('spark://saba-Aspire-VN7-591G:7077')
    sc = SparkContext.getOrCreate(conf)
    entities = pd.read_table("/home/saba/Documents/Big Data Lab/data/FB15k/entity2id.txt", header=None)
    dict_entities = dict(zip(entities[0], entities[1]))
    print(len(entities))
    relations = pd.read_table("/home/saba/Documents/Big Data Lab/data/FB15k/relation2id.txt", header=None)
    dict_relations = dict(zip(relations[0], relations[1]))
    print(len(relations))
    training_df = pd.read_table("/home/saba/Documents/Big Data Lab/data/FB15k/train.txt", header=None)
    validation_df = pd.read_table("/home/saba/Documents/Big Data Lab/data/FB15k/valid.txt", header=None)
    test_df=pd.read_table("/home/saba/Documents/Big Data Lab/data/FB15k/test.txt", header=None)

    training_triples = list(zip([dict_entities[h] + 1 for h in training_df[0]],
                                 [dict_entities[t] + 1 for t in training_df[1]],
                                  [dict_relations[r]+len(entities) + 1 for r in training_df[2]]))
    validation_triples = list(zip([dict_entities[h] + 1 for h in validation_df[0]],
                                [dict_entities[t] + 1 for t in validation_df[1]],
                                [dict_relations[r] + len(entities) + 1 for r in validation_df[2]]))
    testing_triples=list(zip([dict_entities[h] + 1 for h in test_df[0]],
                                [dict_entities[t] + 1 for t in test_df[1]],
                                [dict_relations[r] + len(entities) + 1 for r in test_df[2]]))

    transE = TransE(dict_entities, dict_relations, training_triples, validation_triples, testing_triples,margin=1, dim=50)
    transE.generate_corrupted_triplets(type="train")


    transE.total_embeddings = len(entities)+len(relations)
    #print("type",(no_entities_relations))

    transE.make_samples(transE.total_embeddings)

    # transE.generate_corrupted_test_triplets()
    # transE.validation = validation_triples
    # transE.generate_corrupted_triplets()
    # transE.validate(no_entities_relations)


    #transE.create_embeddings()
    #transE.createrdd(sc)
    #transE.makeRDD(sc)
    sc.stop()