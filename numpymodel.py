from bigdl.nn.layer import *

# layer = LookupTable(9, 4, 2.0, 0.1, 2.0, True)
# input = np.array([5.0, 2.0, 6.0, 9.0, 4.0]).astype("float32")
sample=np.array([[[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],
       [[ 9, 10, 11],
        [12, 13, 14],
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
model.add(SplitTable(1)).add(SelectTable(1))
# branches = ParallelTable()
# branch1 = Sequential().add(SelectTable(1))
# branch2 = Sequential().add(SelectTable(2))
# branches.add(branch1).add(branch2)
# model.add(branches)
# model.add(SelectTable(1))
output = model.forward(sample)

print(output)