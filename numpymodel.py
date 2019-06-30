from bigdl.nn.layer import *
from bigdl.nn.keras.layer import Merge, InputLayer
# layer = LookupTable(9, 4, 2.0, 0.1, 2.0, True)
# input = np.array([5.0, 2.0, 6.0, 9.0, 4.0]).astype("float32")
sample=np.array([[[[ 20,  21,  23],
        [ 3,  4,  5],
        [ 6,  7,  8]],
       [[ 9, 10, 11],
        [512, 13, 14],
        [15, 16, 17]],
       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]],
        [[[ 30,  31,  32],
        [ 33,  34,  35],
        [ 36,  37,  38]],
       [[ 39, 310, 311],
        [312, 313, 314],
        [315, 316, 317]],
       [[318, 319, 320],
        [321, 322, 323],
        [324, 325, 326]]]])
print(sample)
print(type(sample))
model = Sequential()
model.add(SplitTable(1))
branches = ParallelTable()
branch1 = Sequential()
pos_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
pos_add= pos_h_l.add(CAddTable())
pos_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
triplepos_meta = Sequential().add(ConcatTable().add(pos_add).add(pos_t))
triplepos_dist = triplepos_meta.add(CAddTable()).add(Abs())
triplepos_score = triplepos_dist.add(Unsqueeze(1)).add(Mean(3,1)).add(MulConstant(3.0))

branch1.add(triplepos_score)

branch2 = Sequential()
neg_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
neg_add= neg_h_l.add(CAddTable())
neg_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
tripleneg_meta= Sequential().add(ConcatTable().add(neg_add).add(neg_t))
tripleneg_dist = tripleneg_meta.add(CAddTable()).add(Abs())
tripleneg_score = tripleneg_dist.add(Unsqueeze(1)).add(Mean(3,1)).add(MulConstant(3.0))
branch2.add(tripleneg_score)

branches.add(branch1).add(branch2)
model.add(branches)
output = model.forward(sample)

print(output)

# from bigdl.nn.layer import *
# from bigdl.nn.criterion import *
# import numpy as np
# input = np.array([
#           [1.0, 2.0],
#           [3.0, 4.0]
#         ])
# scalar = 2.0
# model = Sequential()
# model.add(Select(1,2)).add(MulConstant(scalar))
# # model.add(MulConstant(scalar))
# output = model.forward(input)
# output

# from bigdl.nn.layer import *
# import numpy as np
#
# mlp = Concat(2)
# mlp.add(Sum(2))
# print(mlp.forward(np.array([[1, 2, 3], [4, 5, 6]])))
#
#
# from bigdl.nn.layer import *
# import numpy as np
#
# mlp = Sequential()
# mlp.add(ConcatTable().add(Identity()).add(Identity()))
# mlp.add(CAddTable())
#
# print(mlp.forward(np.array([[1, 2, 3], [4, 5, 6]])))
#######
# model = Sequential()
# model.add(SplitTable(1))
# branches = ParallelTable()
# branch1 = Sequential()
# branch1.add(Select(1,2)).add(MulConstant(-1.0))
# branch2 = Sequential()
# branch2.add(Select(1,2)).add(MulConstant(-1.0))
# branches.add(branch1).add(branch2)
# model.add(branches)
# # model.add(SelectTable(1))module
# output = model.forward(sample)
#######

#######
# print(sample)
# print(type(sample))
# model = Sequential()
# model.add(SplitTable(1))
# branches = ParallelTable()
# branch1 = Sequential()
# pos_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
# pos_add= pos_h_l.add(CAddTable())
# pos_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# branch1.add(ConcatTable().add(pos_add).add(pos_t))
# branch1.add(CAddTable())
#
# branch2 = Sequential()
# neg_h_l = Sequential().add(ConcatTable().add(Select(1,1)).add(Select(1,3)))
# neg_add= neg_h_l.add(CAddTable())
# neg_t= Sequential().add(Select(1,2)).add(MulConstant(-1.0))
# branch2.add(ConcatTable().add(neg_add).add(neg_t))
# branch2.add(CAddTable())
#
# branches.add(branch1).add(branch2)
# model.add(branches)
# output = model.forward(sample)
#
# print(output)
#######