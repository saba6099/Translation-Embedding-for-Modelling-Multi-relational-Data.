import numpy as np
from bigdl.nn.layer import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *

from pyspark import SparkContext, SparkConf
from bigdl.nn.keras.layer import Merge, InputLayer
# layer = LookupTable(9, 4, 2.0, 0.1, 2.0, True)
# input = np.array([5.0, 2.0, 6.0, 9.0, 4.0]).astype("float32")

# layer = LookupTable(4, 2)
# input = np.array([0.002, 2.0, 3.0, 3.0, 4.0])
# layer.set_weights(np.array([0.2,1,1,1,1,1,1,1]))
#
# output = layer.forward(input)
# print(output)

conf = SparkConf().setAppName('test').setMaster('spark://saba-Aspire-VN7-591G:7077')
sc = SparkContext.getOrCreate(conf)
# model = Linear(2, 4)

# # model = Sequential()
# # samples = [
# #   Sample.from_ndarray(np.random.rand(2,2,2).reshape(2,4).squeeze(), np.array([1])),
# #   Sample.from_ndarray(np.random.rand(2,2,2).reshape(2,4).squeeze(), np.array([1])),
# #   Sample.from_ndarray(np.random.rand(2,2,2).reshape(2,4).squeeze(), np.array([1])),
# #   Sample.from_ndarray(np.random.rand(2,2,2).reshape(2,4).squeeze(), np.array([1]))
# # ]
# # train_data = sc.parallelize(samples)
# # print("Sample count : ",train_data.count())
# # init_engine()
# # optimizer = Optimizer(model, train_data, MSECriterion(), MaxIteration(10), 4)
# # optimizer.optimize()
# # # model.get_weights()[0]
# sample = np.array([[[[ 1],
#          [ 2],
#          [ 3 ]],
#         [[ 4],
#          [ 5],
#          [ 6]]],
#        [[[ 7],
#          [ 8],
#          [ 9]],
#         [[ 10],
#          [ 11],
#          [ 12]]],
#        [[[ 13],
#          [ 14],
#          [ 15]],
#         [[ 16],
#          [ 17],
#          [ 18]]],
#        [[[ 19],
#          [ 20],
#          [ 21]],
#         [[ 22],
#          [ 23],
#          [ 24]]],
#        [[[ 25],
#          [ 26],
#          [ 27]],
#         [[ 28  ],
#          [ 29],
#          [ 30]]],
#        [[[ 31],
#          [ 32],
#          [ 33]],
#         [[ 34],
#          [ 35],
#          [ 36]]],
#        [[[ 37],
#          [ 38],
#          [ 39]],
#         [[ 40],
#          [ 41],
#          [42]]],
#        [[[ 43],
#          [ 44],
#          [ 45]],
#         [[ 46],
#          [ 47],
#          [ 48]]]])



# sample=np.array([[1,2,3,1,2,4],[2,3,4,6,3,4],[4,5,6,4,5,1],[5,6,7,2,6,7]])
# sample=[[[ 1 , 2 ]
#   [ 2. , 1.]
#   [ 3. , 1.]
#   [ 1. , 1.]
#   [ 2. , 1.]
#   [ 4. , 1.]]
#
#  [[ 1. , 1.]
#   [ 1. , 1.]
#   [ 1. , 1.]
#   [ 1. , 1.]
#   [ 1. , 1.]
#   [ 1. , 1.]]]
# a=np.ones((2,6,2))
#print(a)


# sample_rdd = sc.parallelize(sample)
# labels = np.zeros(8)
# # labels=np.array([[1],[1],[1],[1]])
# labels = sc.parallelize(labels)
# record = sample_rdd.zip(labels)
# train_data = record.map(lambda t: Sample.from_ndarray(t[0], t[1]))

# print(train_data.collect())




#### JTensor
sample = np.array([[[[ 1],
         [ 2],
         [ 3 ]],
        [[ 4],
         [ 5],
         [ 6]]]])
# train_data = JTensor.from_ndarray(sample)
model =Sequential()
model.add(Reshape([6]))
embedding = LookupTable(49,2)
model.add(embedding)
model.add(Reshape([2,3,1,2])).add(Squeeze(1))
# print(model.forward((train_data)))
model.add(SplitTable(1))

branches = ParallelTable()
branch1 = Sequential()
pos_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
pos_add= pos_h_l.add(CAddTable())
pos_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
triplepos_meta = Sequential().add(ConcatTable().add(pos_add).add(pos_t))
triplepos_dist = triplepos_meta.add(CAddTable()).add(Abs())
triplepos_score = triplepos_dist.add(Unsqueeze(1)).add(Mean(3,1)).add(MulConstant(2.0))
branch1.add(triplepos_score)#.add(AddConstant(1.0))#.add(Unsqueeze(1))
# pos_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# a = Sequential().add(Narrow(1,1,2)).add(SplitTable(2))
# c = Sequential().add(CAveTable())
# b = Sequential().add(Select(1,2))

# branch1 = Sequential().add(CAveTable()).add(MulConstant(1.0))
# branch1 = Sequential().add(c)

branch2 = Sequential()
neg_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
neg_add= neg_h_l.add(CAddTable())
neg_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
tripleneg_meta= Sequential().add(ConcatTable().add(neg_add).add(neg_t))
tripleneg_dist = tripleneg_meta.add(CAddTable()).add(Abs())
tripleneg_score = tripleneg_dist.add(Unsqueeze(1)).add(Mean(3,1)).add(MulConstant(2.0))
branch2.add(tripleneg_score)#.add(Unsqueeze(1))
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
pos_plus_margin=Sequential().add(SelectTable(1)).add(AddConstant(1.0))

model.add(ConcatTable().add(pos_plus_margin).add(SelectTable(2))).add(CSubTable()).add(Abs())

result = model.predict(sample)
# print("Result",result.take(5))

# output = model.forward(train_data)
# print(output)
#



#### Numpy
# sample = np.random.rand(2,3,5)
# sample=np.array([[[ 20,  21,  23, 24, 25],
#                     [ 3,  4,  5, 6, 7],
#                     [ 6,  7,  8, 18, 19]],
#
#                     [[ 9, 10, 11,20, 21],
#                     [12, 13, 14, 22, 23],
#                     [15, 16, 17, 24, 25]]])
# model = Sequential()
# model.add(SplitTable(1))
# # model.add(Select(2,2))
# branches = ParallelTable()
# branch1 = Sequential().add(Narrow(1, 2)).add(Narrow(1,1)).add(Squeeze(1)).add(Squeeze(1)).add(Squeeze(1))
# branch2 = Sequential().add(Narrow(1, 1)).add(Narrow(1,1)).add(Squeeze(1)).add(Squeeze(1)).add(Squeeze(1))
# branches.add(branch1).add(branch2)
# model.add(branches)
#
# output = model.forward(sample)
# print(output)




##### MODEL
# branch1 = Sequential()
# pos_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
# pos_add= pos_h_l.add(CAddTable())
# pos_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# triplepos_meta = Sequential().add(ConcatTable().add(pos_add).add(pos_t))
# triplepos_dist = triplepos_meta.add(CAddTable()).add(Abs())
# triplepos_score = triplepos_dist.add(Unsqueeze(1)).add(Mean(3,1)).add(MulConstant(3.0))
# branch1.add(triplepos_score)
#
# branch2 = Sequential()
# neg_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
# neg_add= neg_h_l.add(CAddTable())
# neg_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# tripleneg_meta= Sequential().add(ConcatTable().add(neg_add).add(neg_t))
# tripleneg_dist = tripleneg_meta.add(CAddTable()).add(Abs())
# tripleneg_score = tripleneg_dist.add(Unsqueeze(1)).add(Mean(3,1)).add(MulConstant(3.0))
# branch2.add(tripleneg_score)
# branches.add(branch1).add(branch2)
# model.add(branches)


#
optimizer = Optimizer(
    model=model,
    training_rdd=train_data,
    criterion=AbsCriterion(False),
    optim_method=SGD(learningrate=0.01,learningrate_decay=0.0002),
    end_trigger=MaxEpoch(4),
    batch_size=4)


optimizer.optimize()

#
#
#
#
# # 4D numpy array, Dimensions are: 1:triples, 2:embedding dimensions 3:head/tail/relation 4:true/corrupted triples
# # sample=(np.array([[[[ 20,  21,  23],
# #                     [ 3,  4,  5],
# #                     [ 6,  7,  8]],
# #
# #                     [[ 9, 10, 11],
# #                     [512, 13, 14],
# #                     [15, 16, 17]],
# #
# #                    [[18, 19, 20],
# #                     [21, 22, 23],
# #                     [24, 25, 26]]],
# #
# #                     [[[ 30,  31,  32],
# #                     [ 33,  34,  35],
# #                     [ 36,  37,  38]],
# #
# #                    [[ 39, 310, 311],
# #                     [312, 313, 314],
# #                     [315, 316, 317]],
# #
# #                    [[318, 319, 320],
# #                     [321, 322, 323],
# #                     [324, 325, 326]]]]))
#
# # sample=[Sample.from_ndarray(np.array([1,3,4,5,6,1])),np.array([1,3,4,5,6,1]),np.array([1,3,4,5,6,1]),np.array([1,3,4,5,6,1])]
# # label=[np.ones(4)]
# # #label =list(1,1)
# # print(label)
# # sample_input = Sample.from_ndarray(sample, label)
# # print(type(sample_input))
# # print("Sample feature : ",sample_input.feature)
# # # Retrieve feature and label from a Sample
# # # sample_input.feature
# # # sample_input.label
# # print("")
# # model = Linear(2, 1)
# # sample_rdd = sc.parallelize([sample_input])
# # print("hello")
# # print(sample_rdd.count())
# #
# # init_engine()
# # optimizer = Optimizer(model, sample_rdd, MSECriterion(), MaxIteration(10), 1)
# # optimizer.optimize()
# # model.get_weights()[0]
# # print(sample_rdd.collect())
# # print("**Sample type###",type(sample_rdd))
#
#
# # model = Sequential()
# # #Splitting the Input and putting in parralel branches for true and corrupted inputs
# # model.add(SplitTable(1))
# # branches = ParallelTable()
# # #branch 1 works with true inputs
# # branch1 = Sequential()
# # #performs addition of heads and relations
# # pos_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
# # pos_add= pos_h_l.add(CAddTable())
# # #performs negation of tails
# # pos_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# # #calculates distance between them
# # triplepos_meta = Sequential().add(ConcatTable().add(pos_add).add(pos_t))
# # triplepos_dist = triplepos_meta.add(CAddTable()).add(Abs())
# # triplepos_score = triplepos_dist.add(Unsqueeze(1)).add(Mean(3,1)).add(MulConstant(3.0))
# # branch1.add(triplepos_score)
# # # Further branch2 operations are similar
# #
# # branch2 = Sequential()
# # neg_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
# # neg_add= neg_h_l.add(CAddTable())
# # neg_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# # tripleneg_meta= Sequential().add(ConcatTable().add(neg_add).add(neg_t))
# # tripleneg_dist = tripleneg_meta.add(CAddTable()).add(Abs())
# # tripleneg_score = tripleneg_dist.add(Unsqueeze(1)).add(Mean(3,1)).add(MulConstant(3.0))
# # branch2.add(tripleneg_score)
# # branches.add(branch1).add(branch2)
# # model.add(branches)
# # output = model.forward(sample)
# # print(output)
#
# #
# #
#
#
#
# trained_mnist_model = optimizer.optimize()
#
#
#
#
#
#
#
#
# # print(output)
#
# # from bigdl.nn.layer import *
# # from bigdl.nn.criterion import *
# # import numpy as np
# # input = np.array([
# #           [1.0, 2.0],
# #           [3.0, 4.0]
# #         ])
# # scalar = 2.0
# # model = Sequential()
# # model.add(Select(1,2)).add(MulConstant(scalar))
# # # model.add(MulConstant(scalar))
# # output = model.forward(input)
# # output
#
# # from bigdl.nn.layer import *
# # import numpy as np
# #
# # mlp = Concat(2)
# # mlp.add(Sum(2))
# # print(mlp.forward(np.array([[1, 2, 3], [4, 5, 6]])))
# #
# #
# # from bigdl.nn.layer import *
# # import numpy as np
# #
# # mlp = Sequential()
# # mlp.add(ConcatTable().add(Identity()).add(Identity()))
# # mlp.add(CAddTable())
# #
# # print(mlp.forward(np.array([[1, 2, 3], [4, 5, 6]])))
# #######
# # model = Sequential()
# # model.add(SplitTable(1))
# # branches = ParallelTable()
# # branch1 = Sequential()
# # branch1.add(Select(1,2)).add(MulConstant(-1.0))
# # branch2 = Sequential()
# # branch2.add(Select(1,2)).add(MulConstant(-1.0))
# # branches.add(branch1).add(branch2)
# # model.add(branches)
# # # model.add(SelectTable(1))module
# # output = model.forward(sample)
# #######
#
# #######
# # print(sample)
# # print(type(sample))
# # model = Sequential()
# # model.add(SplitTable(1))
# # branches = ParallelTable()
# # branch1 = Sequential()
# # pos_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
# # pos_add= pos_h_l.add(CAddTable())
# # pos_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# # branch1.add(ConcatTable().add(pos_add).add(pos_t))
# # branch1.add(CAddTable())
# #
# # branch2 = Sequential()
# # neg_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
# # neg_add= neg_h_l.add(CAddTable())
# # neg_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# # branch2.add(ConcatTable().add(neg_add).add(neg_t))
# # branch2.add(CAddTable())
# #
# # branches.add(branch1).add(branch2)
# # model.add(branches)
# # output = model.forward(sample)
# #
# # print(output)
#######


#######
# from bigdl.nn.layer import *
# from bigdl.nn.criterion import *
# from bigdl.optim.optimizer import *
# from bigdl.util.common import *
# mse = MarginRankingCriterion(margin=1.0, size_average=False)
# input1 = np.array([3, 7, 2, 18, 311]).astype("float32")
# input2 = np.array([4, 9, 1, 9, 3124]).astype("float32")
# input = [input1, input2]
# target1 = np.array([1, 1, 1, 1, 1]).astype("float32")
# target = [target1, target1]
# output = mse.forward(input, target)
# creating: createMarginRankingCriterion
# output
# 2819.0
#####