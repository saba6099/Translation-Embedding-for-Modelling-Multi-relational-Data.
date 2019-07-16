import numpy as np
from bigdl.nn.criterion import *
import pandas as pd
from bigdl.util.common import *
import time
import sys
import random
from bigdl.optim.optimizer import *

init_engine()
import os

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *

from pyspark import SparkContext, SparkConf
# os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"
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


def create_model(total_embeddings, embedding_dim = 5, margin=1.0):
    model = Sequential()
    model.add(Reshape([6]))

    embedding = LookupTable(total_embeddings, embedding_dim)
    model.add(embedding)

    # print(model.forward(train_data.take(1)[0].features))
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
        self.batch_total_validation = []
        self.total_embeddings = 0


    def make_samples(self,total_embeddings):
        sample = np.array(self.batch_total)
        sample_rdd = sc.parallelize(sample)
        labels = np.zeros(len(self.triplets_list))
        labels = sc.parallelize(labels)
        record = sample_rdd.zip(labels)
        train_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))
        print("Train Data",train_data.take(10))
        print("Hello")


        sample = np.array(self.batch_total_validation)
        sample_rdd = sc.parallelize(sample)
        labels = np.zeros(len(self.test_triples))
        labels = sc.parallelize(labels)
        record = sample_rdd.zip(labels)
        test_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))
        # print(test_data.take(10))

        sample = np.array([[[[1],
                             [2],
                             [3]],
                            [[4],
                             [5],
                             [6]]]])
        sample = np.array([1, 2, 3, 4, 5, 6])

        model = create_model(total_embeddings)
        print(model.parameters())

        # compare models
        numpymodel = numpy_model(sample)
        output = model.forward(sample)
        print("Model Output", output)
        # sys.exit()
        print(sample.shape)
        # sample = np.random.randint(1, total_embeddings - 1, (4, 2, 3, 1))
        # train_data = JTensor.from_ndarray(sample)
        # model = create_model(total_embeddings, train_data)
        model = create_model(total_embeddings)
        print(model.parameters())
        # output = model.forward(test_data)
        # print("Model Output", output)
        # sys.exit()


        optimizer = Optimizer(
            model = model,
            training_rdd = train_data,
            criterion = AbsCriterion(False),
            optim_method = SGD(learningrate=0.01,learningrate_decay=0.001,weightdecay=0.001,
                   momentum=0.0,dampening=DOUBLEMAX,nesterov=False,
                   leaningrate_schedule=None,learningrates=None,
                   weightdecays=None,bigdl_type="float"),
            end_trigger = MaxEpoch(2),
            batch_size = 4)

        # optimizer.set_validation(
        #     batch_size=128,
        #     val_rdd=validation_data,
        #     trigger=EveryEpoch(),
        #     val_method=[Top1Accuracy()]
        # )

        trained_model = optimizer.optimize()

        print(trained_model.parameters())

        # result = trained_model.evaluate(train_data, 8, [Loss(AbsCriterion(False))])
        # result = trained_model.evaluate(test_data, 8, [Loss()])


        result = trained_model.predict(test_data)
        print("Result",result.take(1))
        # print("Result", result)

    def generate_corrupted_triplets(self, type = "train", cycle_index=1):
        count = 0
        print("\n********** Start TransE training **********")
        for i in range(cycle_index):

            if count == 0:
                start_time = time.time()
            count += 1
            # print("Batch number:", count)
            if(type == "train"):
                Sbatch = self.triplets_list
            else:
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
            if(type == "validation"):
                self.batch_total_validation+=batch_total
            else:
                self.batch_total+=(batch_total)
        # print(("batch_pos",self.batch_pos))
        # print(("batch_neg",self.batch_neg))
    #
    # def distance(self, head, tail, relation, corrupted_head, corrupted_tail, corrupted_relation):
    #     # sample=np.array([[[[head],[tail],[relation]],[[corrupted_head],[corrupted_tail],[corrupted_relation]]]])
    #     sample = np.array([[[[1],
    #                          [2],
    #                          [3]],
    #                         [[4],
    #                          [5],
    #                          [6]]],
    #                        [[[7],
    #                          [8],
    #                          [9]],
    #                         [[10],
    #                          [11],
    #                          [12]]],
    #                        [[[13],
    #                          [14],
    #                          [15]],
    #                         [[16],
    #                          [17],
    #                          [18]]],
    #                        [[[19],
    #                          [20],
    #                          [21]],
    #                         [[22],
    #                          [23],
    #                          [24]]],
    #                        [[[25],
    #                          [26],
    #                          [27]],
    #                         [[28],
    #                          [29],
    #                          [30]]],
    #                        [[[31],
    #                          [32],
    #                          [33]],
    #                         [[34],
    #                          [35],
    #                          [36]]],
    #                        [[[37],
    #                          [38],
    #                          [39]],
    #                         [[40],
    #                          [41],
    #                          [42]]],
    #                        [[[43],
    #                          [44],
    #                          [45]],
    #                         [[46],
    #                          [47],
    #                          [48]]]])
    #     # test_data = JTensor.from_ndarray(sample)
    #     sample_rdd = sc.parallelize(sample)
    #     labels = np.zeros(8)
    #     # labels=np.array([[1],[1],[1],[1]])
    #     labels = sc.parallelize(labels)
    #     record = sample_rdd.zip(labels)
    #     test_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))
    #
    #     # model = self.make_samples()
    #
    #     model = Model.loadModel("/home/saba/Documents/Big Data Lab/data/model.bigdl", "/home/saba/Documents/Big Data Lab/data/model.bin")
    #     result = model.evaluate(test_data, 8, [Top1Accuracy()])
    #     print(result)
    #     return result

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




        #print(trained_model("embedding"))

        # sample = np.array([[[[1],
        #                      [2],
        #                      [3]],
        #                     [[4],
        #                      [5],
        #                      [6]]],
        #                    [[[7],
        #                      [8],
        #                      [9]],
        #                     [[10],
        #                      [11],
        #                      [12]]],
        #                    [[[13],
        #                      [14],
        #                      [15]],
        #                     [[16],
        #                      [17],
        #                      [18]]],
        #                    [[[19],
        #                      [20],
        #                      [21]],
        #                     [[22],
        #                      [23],
        #                      [24]]],
        #                    [[[25],
        #                      [26],
        #                      [27]],
        #                     [[28],
        #                      [29],
        #                      [30]]],
        #                    [[[31],
        #                      [32],
        #                      [33]],
        #                     [[34],
        #                      [35],
        #                      [36]]],
        #                    [[[37],
        #                      [38],
        #                      [39]],
        #                     [[40],
        #                      [41],
        #                      [42]]],
        #                    [[[43],
        #                      [44],
        #                      [45]],
        #                     [[46],
        #                      [47],
        #                      [48]]]])
        # # test_data = JTensor.from_ndarray(sample)
        # sample_rdd = sc.parallelize(sample)
        # labels = np.zeros(8)
        # # labels=np.array([[1],[1],[1],[1]])
        # labels = sc.parallelize(labels)
        # record = sample_rdd.zip(labels)
        # test_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))

        #model = self.make_samples()

        #model = Model.loadModel("/home/saba/Documents/Big Data Lab/data/model.bigdl",
        #                        "/home/saba/Documents/Big Data Lab/data/model.bin")

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
    transE.generate_corrupted_triplets(type="validation")

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




"""
/usr/bin/python3.6 "/home/saba/Documents/Big Data Lab/test.py"
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
2019-07-13 12:46:05 WARN  Utils:66 - Your hostname, saba-Aspire-VN7-591G resolves to a loopback address: 127.0.1.1; using 192.168.0.104 instead (on interface wlp7s0)
2019-07-13 12:46:05 WARN  Utils:66 - Set SPARK_LOCAL_IP if you need to bind to another address
2019-07-13 12:46:06 WARN  NativeCodeLoader:62 - Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
2019-07-13 12:46:07 WARN  SparkContext:66 - Using an existing SparkContext; some configuration may not take effect.
cls.getname: com.intel.analytics.bigdl.python.api.Sample
BigDLBasePickler registering: bigdl.util.common  Sample
cls.getname: com.intel.analytics.bigdl.python.api.EvaluatedResult
BigDLBasePickler registering: bigdl.util.common  EvaluatedResult
cls.getname: com.intel.analytics.bigdl.python.api.JTensor
BigDLBasePickler registering: bigdl.util.common  JTensor
cls.getname: com.intel.analytics.bigdl.python.api.JActivity
BigDLBasePickler registering: bigdl.util.common  JActivity
14951
1345

********** Start TransE training **********

********** Start TransE training **********
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Train Data [Sample: features: [JTensor: storage: [  9448.   5031.  15304.   9448.  12895.  15304.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [  4887.  13681.  15271.   4887.    174.  15271.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [  7375.  13063.  15600.   7375.  11894.  15600.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [ 11437.   7446.  15095.  11437.   3336.  15095.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [ 12511.   4747.  15333.  12511.  14336.  15333.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [  6548.   4440.  15198.  10966.   4440.  15198.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [  5282.  10597.  15531.   5282.   2233.  15531.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [ 11563.   6581.  15703.  11563.  11471.  15703.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [  3016.   2120.  15339.   4724.   2120.  15339.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float], Sample: features: [JTensor: storage: [  8642.   6184.  16178.   2073.   6184.  16178.], shape: [2 3 1], float], labels: [JTensor: storage: [ 0.], shape: [1], float]]
Hello
(1, 2, 3, 1)
creating: createSequential
creating: createReshape
creating: createLookupTable
creating: createReshape
creating: createSqueeze
creating: createSplitTable
creating: createParallelTable
creating: createSequential
creating: createSequential
creating: createConcatTable
creating: createSelect
creating: createSelect
creating: createCAddTable
creating: createSequential
creating: createSelect
creating: createMulConstant
creating: createSequential
creating: createConcatTable
creating: createCAddTable
creating: createAbs
creating: createUnsqueeze
creating: createMean
creating: createMulConstant
creating: createSequential
creating: createSequential
creating: createConcatTable
creating: createSelect
creating: createSelect
creating: createCAddTable
creating: createSequential
creating: createSelect
creating: createMulConstant
creating: createSequential
creating: createConcatTable
creating: createCAddTable
creating: createAbs
creating: createUnsqueeze
creating: createMean
creating: createMulConstant
creating: createSequential
creating: createSelectTable
creating: createAddConstant
creating: createConcatTable
creating: createSelectTable
creating: createCSubTable
creating: createAbs
{'LookupTablea861dbe2': {'weight': array([[-0.15823296,  1.15599239, -1.10228109,  0.97598147, -0.87271488],
       [-1.45211852,  0.22384943, -0.42040601, -1.52288234,  0.36869732],
       [ 0.2179819 , -0.58092207,  0.15613583, -0.77772647, -1.09658873],
       ..., 
       [-1.43756568,  0.89641643,  0.1706786 , -0.18898763,  0.32030115],
       [ 1.05666268,  1.40475869,  1.54771531, -0.62869328, -0.36038619],
       [ 0.00263797, -0.10111605, -0.42646685, -0.22277924,  1.02462721]], dtype=float32), 'gradWeight': array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       ..., 
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]], dtype=float32)}}
creating: createAbsCriterion
creating: createDefault
creating: createSGD
creating: createMaxEpoch
creating: createDistriOptimizer
disableCheckSingleton is deprecated. Please use bigdl.check.singleton instead
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
2019-07-13 12:46:09 WARN  BlockManager:66 - Asked to remove block test_0weights0, which does not exist
2019-07-13 12:46:09 WARN  BlockManager:66 - Asked to remove block test_0gradients0, which does not exist
2019-07-13 12:46:09 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:09 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:09 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:09 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:09 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:09 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:10 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:11 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:12 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:13 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:14 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
2019-07-13 12:46:15 WARN  DistriOptimizer$:230 - Warning: for better training speed, total batch size is recommended to be at least two times of core number4, please tune your batch size accordingly
{'LookupTablea861dbe2': {'weight': array([[-0.1578801 ,  1.15341461, -1.09982288,  0.97380501, -0.87076885],
       [-1.44888031,  0.22335026, -0.41946855, -1.51948643,  0.36787516],
       [ 0.21749584, -0.57962668,  0.15578763, -0.77599216, -1.09414315],
       ..., 
       [-1.43436003,  0.89441758,  0.17029798, -0.18856621,  0.31958687],
       [ 1.05430651,  1.40162599,  1.54426384, -0.62729144, -0.35958251],
       [ 0.00263208, -0.10089056, -0.4255158 , -0.22228245,  1.02234209]], dtype=float32), 'gradWeight': array([[ -1.57881368e-04,   1.15342380e-03,  -1.09983177e-03,
          9.73812887e-04,  -8.70775839e-04],
       [ -1.44889194e-03,   2.23352050e-04,  -4.19471937e-04,
         -1.51949865e-03,   3.67878121e-04],
       [  2.17497596e-04,  -5.79631364e-04,   1.55788890e-04,
         -7.75998400e-04,  -1.09415187e-03],
       ..., 
       [ -1.43437157e-03,   8.94424797e-04,   1.70299361e-04,
         -1.88567719e-04,   3.19589453e-04],
       [  1.05431501e-03,   1.40163722e-03,   1.54427637e-03,
         -6.27296453e-04,  -3.59585421e-04],
       [  2.63210313e-06,  -1.00891368e-04,  -4.25519218e-04,
         -2.22284245e-04,   1.02235039e-03]], dtype=float32)}}
Result [array(0.13628673553466797, dtype=float32)]

Process finished with exit code 0
"""
