import numpy as np
from bigdl.nn.criterion import *
import pandas as pd
from bigdl.util.common import *
import time
import sys
import random
from bigdl.optim.optimizer import *

init_engine()

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *

from pyspark import SparkContext, SparkConf


def numpy_model(sample):
    head_pos = []
    relation_pos = []
    tail_pos = []
    head_neg = []
    relation_neg = []
    tail_neg = []
    score_pos = []
    score_neg = []
    output = embedding.forward(sample)
    print(output)
    head_pos.append(output[0])
    tail_pos.append(output[1])
    relation_pos.append(output[2])
    head_neg.append(output[3])
    tail_neg.append(output[4])
    relation_neg.append(output[5])
    distance_pos = (head_pos[0] + relation_pos[0] - tail_pos[0])
    distance_neg = (head_neg[0] + relation_neg[0] - tail_neg[0])
    score_pos.append(np.fabs(distance_pos).sum())
    score_neg.append(np.fabs(distance_neg).sum())

    print("Positive Score", score_pos)
    print("Negative Score",score_neg)

def create_model(total_embeddings, embedding_dim = 2, margin=1.0):
    model = Sequential()
    model.add(Reshape([6]))
    global embedding
    embedding = LookupTable(total_embeddings, embedding_dim)
    model.add(embedding)

    model.add(Reshape([2, 3, 1, embedding_dim])).add(Squeeze(1))
    # print(model.forward((train_data)))
    model.add(SplitTable(1))
    # return model

    branches = ParallelTable()
    branch1 = Sequential()
    pos_h_l = Sequential().add(ConcatTable().add(Select(1, 1)).add(Select(1, 3)))
    pos_add = pos_h_l.add(CAddTable())
    pos_t = Sequential().add(Select(1, 2)).add(MulConstant(-1.0))
    triplepos_meta = Sequential().add(ConcatTable().add(pos_add).add(pos_t))
    triplepos_dist = triplepos_meta.add(CAddTable()).add(Abs())
    triplepos_score = triplepos_dist.add(Unsqueeze(1)).add(Mean(3, 1)).add(MulConstant(float(embedding_dim)))
    branch1.add(triplepos_score)

    branch2 = Sequential()
    neg_h_l = Sequential().add(ConcatTable().add(Select(1, 1)).add(Select(1, 3)))
    neg_add = neg_h_l.add(CAddTable())
    neg_t = Sequential().add(Select(1, 2)).add(MulConstant(-1.0))
    tripleneg_meta = Sequential().add(ConcatTable().add(neg_add).add(neg_t))
    tripleneg_dist = tripleneg_meta.add(CAddTable()).add(Abs())
    tripleneg_score = tripleneg_dist.add(Unsqueeze(1)).add(Mean(3, 1)).add(MulConstant(float(embedding_dim)))
    branch2.add(tripleneg_score)

    branches.add(branch1).add(branch2)
    model.add(branches)
    return model

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
        self.batch_total_validation = []
        self.total_embeddings = 0


    def training(self, total_embeddings):
        sample = np.array(self.batch_total)
        sample_rdd = sc.parallelize(sample)
        # labels = np.zeros(len(self.triplets_list))
        # labels = sc.parallelize(labels)
        # record = sample_rdd.zip(labels)
        train_data = sample_rdd.map(lambda t: Sample.from_ndarray(t, labels=[np.array(1.)]))
        print("Train Data", train_data.take(1))
        print("Hello")

        # sample = np.array(self.batch_total_validation)
        # sample_rdd = sc.parallelize(sample)
        # # labels = np.zeros(len(self.test_triples))
        # # labels = sc.parallelize(labels)
        # # record = sample_rdd.zip(labels)
        # test_data = sample_rdd.map(lambda t: Sample.from_ndarray(t, labels=[np.array(1.)]))
        # # print(test_data.take(2))

        # sample = np.array([[[[1],
        #                      [2],
        #                      [3]],
        #                     [[4],
        #                      [5],
        #                      [6]]]])
        sample = np.array([1,2,3,4,5,6])

        model = create_model(total_embeddings)
        print(model.parameters())

        #compare models
        numpymodel = numpy_model(sample)
        output = model.forward(sample)
        print("Model Output", output)
        # sys.exit()

        # optimizer = Optimizer(
        #     model=model,
        #     training_rdd=train_data,
        #     criterion=MarginRankingCriterion(),
        #     optim_method=SGD(learningrate=0.01, learningrate_decay=0.001, weightdecay=0.001,
        #                      momentum=0.0, dampening=DOUBLEMAX, nesterov=False,
        #                      leaningrate_schedule=None, learningrates=None,
        #                      weightdecays=None, bigdl_type="float"),
        #     end_trigger=MaxEpoch(2),
        #     batch_size=4)
        #
        # self.trained_model = optimizer.optimize()
        #
        # print(self.trained_model.parameters())

        # result = trained_model.evaluate(train_data, 8, [Loss(AbsCriterion(False))])
        # result = self.trained_model.predict(test_data)
        # print("Result", result.take(1))
        # # print("Result", result)
        # predmodel = Sequential().add(model).add(JoinTable(1, 1))
        # result = predmodel.predict(train_data)
        # print("Result", result.take(5))


    def generate_training_corrupted_triplets(self, cycle_index=1):
        count = 0
        print("\n********** Start TransE training **********")
        for i in range(cycle_index):

            if count == 0:
                start_time = time.time()
            count += 1
            Sbatch = self.test_triples

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

    def predict_score(self, sample):
        sample = JTensor.from_ndarray(sample)
        return self.trained_model.predict(sample)


    def test_and_calculate_rank(self, cycle_index=1):
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

            raw_batch = Sbatch
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                batch_total = []
                for head, tail, relation in batch_pos:
                    rank_list_head = {}
                    rank_list_tail = {}
                    corrupt_entity_list = list(self.entity_dict.values())

                    for i in range(0, len(corrupt_entity_list)):
                        if (corrupt_entity_list[i], tail, relation) not in (self.training_triple_pool):
                            sample = [head, tail, relation, corrupt_entity_list[i], tail, relation]
                            rank_list_head[i] = self.predict_score(sample)
                            # list_head_replace.append((corrupt_entity_list[i], tail, relation))

                        else:
                            continue
                    for i in range(0, len(corrupt_entity_list)):
                        if (head, corrupt_entity_list[i], relation) not in (self.training_triple_pool):
                            sample = [head, tail, relation, corrupt_entity_list[i], tail, relation]
                            rank_list_tail[i] = self.predict_score(sample)
                            # list_tail_replace.append((head, corrupt_entity_list[i], relation ))
                        else:
                            continue

                    head_rank = sorted(rank_list_head.items())
                    tail_rank = sorted(rank_list_tail.items())

                    # batch_neg_head_replaced[(head, tail, relation)] = list_head_replace
                    # batch_neg_tail_replaced[(head, tail, relation)] = list_tail_replace
                    # list_tail_replace = []
                    # list_head_replace = []



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
    transE.generate_training_corrupted_triplets()
    transE.total_embeddings = len(entities)+len(relations)

    transE.training(transE.total_embeddings)
    transE.test_and_calculate_rank()

    sc.stop()

