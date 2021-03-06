# Translation Embedding for modelling Multi-Relational Data
For further details, please read [Project Report](https://github.com/saba6099/Distributed-Big-Data-Lab-Project/blob/master/Translation_Embedding_for_modelling_multi_relational_data.pdf)
 
## Pre-requisites
Following packages must be installed:

[Pyspark 2.2.3](https://pypi.org/project/pyspark/)

[BigDl 0.7](https://pypi.org/project/BigDL/)

[Python 3.6](https://www.python.org/downloads/)

## Datasets used
1. Freebase
2. WordNet

## To run the code
   1. Change the file paths for entity2id, relation2id, training data and testing data files. 
   2. Give the path to save and load the model in training and testing modules respectively.
   3. Set the path for train_summary.
   4. Run the TransE file with python3.

## Project module description


### Main method: code driver.
The file paths have to be changed in main method itself. It loads the data and calls other modules.

### create_model: TransE score model.
This module calculates the scores of positive and corrupted triples of a training batch. It includes the embedding layer and other BigDL math layers that perform the operation.

### generate_training_corrupted_triplets: 
This module generates a corrupted triple for every training triple by either replacing the head or the tail with a random entity from set of all the entities.

 ### generate_test_corrupted_triplets
This module generates a set of testing corrupted triples by first replacing the head with all the possible entities and then replacing the tail in a similar fashion.

### training
This module generates an RDD of samples for the training set(h, t, l, h', t', l') of all the training triples and then runs BigDL Optimizer on the RDD in a batch of given size. It then plots the training error and saves the trained model at a file location.

### testing
This module generates an RDD of samples for a true test triple and it's corresponding corrupted triples and then predicts the results using the pre-trained model 

 We have used tensorboard to generate training error graphs.
