from vad import Encoder, Backwards, Attention, Decoder, Inference, Prior, CBOW, trainIteration
from vad_utils import loss_function, plotBatchLoss, batchData, loadDataset

import random
import numpy as np
import torch
import torch.nn as nn
seed = 1337

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    print("Loading parameters..", end=" ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        parameters = {
            'hiddenSize'			: 64,
            'latentSize'			: 32,
            'batchSize'				: 32,
            'iterations'			: 5,
            'learningRate'			: 0.0001,
            'gradientClip'			: 5,
            'useBOW'				: False,
            'bidirectionalEncoder'	: True,
            'reduction'             : 512
        }
    else:
        parameters = {
            'hiddenSize'			: 512,
            'latentSize'			: 400,
            'batchSize'				: 64,
            'iterations'			: 5,
            'learningRate'			: 0.0001,
            'gradientClip'			: 5,
            'useBOW'				: False,
            'bidirectionalEncoder'	: True,
            'reduction'             : 1
        }

    hiddenSize = parameters['hiddenSize']
    latentSize = parameters['latentSize']
    batchSize = parameters['batchSize']
    iterations = parameters['iterations']
    learningRate = parameters['learningRate']
    gradientClip = parameters['gradientClip']
    useBOW = parameters['useBOW']
    bidirectionalEncoder = parameters['bidirectionalEncoder']
    reduction = parameters['reduction']

    print(parameters)
    print("Done.")

    print("Loading dataset..", end=" ")
    dataset = loadDataset()
    # setup store parameters
    id2word = dataset['id2word']
    word2id = dataset['word2id']
    weightMatrix = dataset['weights']
    train = dataset['train']
    validation = dataset['validation']
    cutoff = dataset['cutoff']
    paddingID = word2id['<pad>']
    print("Done.")

    print("Converting dataset weights into tensors..", end=" ")
    # convert dataset into tensors
    weightMatrix = torch.tensor(weightMatrix, dtype=torch.float)
    print("Done.")

    # batching data
    print("Batching Data..", end=" ")

    # shuffle data rows
    random.shuffle(train)
    random.shuffle(validation)

    trainx = [x[0] for x in train[::reduction]]
    trainy = [x[1] for x in train[::reduction]]
#     valx = [x[0] for x in validation]
#     valy = [x[1] for x in validation]

    # shuffle data row

    trainx = batchData(trainx, paddingID, device, batchSize, cutoff)
    trainy = batchData(trainy, paddingID, device, batchSize, cutoff)
    # trainx = batchData(valx, paddingID, batchSize, cutoff)
    # trainy = batchData(valy, paddingID, batchSize, cutoff)

    traindata = (trainx, trainy)
    # valdata= (valx, valy)
    print("Done.")

    # setup variables for model components initialisation
    maxReviewLength = cutoff
    vocabularySize = len(id2word)
    embeddingDim = weightMatrix.shape[1]
    embedding_shape = weightMatrix.shape

    print("Initialising model components..", end=" ")

    modelEncoder = Encoder(weightMatrix, vocabularySize,
                           paddingID, hiddenSize, bidirectionalEncoder).to(device)

    modelAttention = Attention(hiddenSize, bidirectionalEncoder).to(device)
    modelBackwards = Backwards(weightMatrix, vocabularySize,
                               paddingID, hiddenSize, bidirectionalEncoder).to(device)
    modelInference = Inference(
        hiddenSize, latentSize, bidirectionalEncoder).to(device)
    modelPrior = Prior(hiddenSize, latentSize, bidirectionalEncoder).to(device)
    modelDecoder = Decoder(weightMatrix, vocabularySize,
                           paddingID, batchSize, maxReviewLength, hiddenSize, latentSize, bidirectionalEncoder).to(device)
    modelBOW = CBOW(vocabularySize, latentSize).to(device)

    criterion_r = nn.NLLLoss(ignore_index=paddingID)
    criterion_bow = nn.BCEWithLogitsLoss()

    print("Done.")

    trainIteration(device,
                   traindata,
                   modelEncoder,
                   modelAttention,
                   modelBackwards,
                   modelInference,
                   modelPrior,
                   modelDecoder,
                   modelBOW,
                   iterations,
                   word2id,
                   criterion_r,
                   criterion_bow,
                   learningRate,
                   gradientClip,
                   useBOW,
                   printEvery=1000)

    saveModels(modelEncoder, modelBackwards, modelAttention,
               modelInference, modelPrior, modelDecoder, modelBOW)

    # GOAL
    # Batch: 0 Loss: 26731.2604 LL: 22.8447 KL: 26707.8917 AUX: 0.5253
