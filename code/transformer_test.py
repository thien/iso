import torch
from transformer import Transformer
import random
import numpy as np
import pickle
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

"""
Set seed #
"""
seed = 1337

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def loadDataset(path = '../Datasets/Reviews/dataset_ready.pkl'):
    return pickle.load(open(path, 'rb'))

def batchData(dataset, eos, batchsize=32):
    """
    Splits the dataset into batches.
    Each batch needs to be sorted by 
    the length of their sequence in order
    for `pack_padded_sequence` to be used.
    """
    datasize = len(dataset)
    batches = []
    # split data into batches.
    for i in range(0, datasize, batchsize):
        batches.append(dataset[i:i+batchsize])
    # within each batch, sort the entries.
    for i in range(len(batches)):
        batch = batches[i]
        # get lengths of each review in the batch
        # based on the postion of the EOS tag.
        lengths = (batch==eos).nonzero()[:,1]
        # sort the lengths
        ordered = torch.argsort(lengths, descending=True)
        # get the reviews based on the sorted batch lengths
        reviews = batch[ordered]
        # re-allocate values.
        batches[i] = (reviews, lengths[ordered].cpu().numpy())
    return batches

def trainModel(x,y,xLength,yLength,model,optimiser,criterion):
    # set up zero gradients
    optimiser.zero_grad()
    # evaluate on transformer
    y_predicted, _, _, _ = model(x,xLength,y,yLength)
    # calculate the loss
    loss = criterion(y_predicted, y)
    # update gradients
    loss.backward()
    optimiser.step()
    return loss/y[0].shape[0]

def train(model, dataset, epochs, optimiser, criterion):
    for e in range(epochs):
        for batch in dataset:
            entry, length = batch
            x, y = entry, entry
            xLength, yLength = length, length

            loss = trainModel(x,y,xLength,yLength,model,optimiser,criterion)
            
    
if __name__ == "__main__":
    print("Loading parameters..", end=" ")
    hiddenSize = 128
    epochs = 10
    batchSize = 32
    learningRate = 0.001
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("Done.")

    print("Loading dataset..", end=" ")
    dataset = loadDataset()
    # setup store parameters
    id2word = dataset['id2word']
    word2id = dataset['word2id']
    weightMatrix = dataset['weights']
    dataset = dataset['reviews']
    print("Done.")

    print("Converting dataset weights into tensors..", end=" ")
    # convert dataset into tensors
    dataset = torch.tensor(dataset, dtype=torch.long, device=device)
    weightMatrix = torch.tensor(weightMatrix, dtype=torch.float)
    print("Done.")

    # batching data
    print("Batching Data..",end=" ")
    dataset = batchData(dataset, word2id['<eos>'], batchSize)
    print("Done.")
    
    # setup variables for model components initialisation
    maxReviewLength = dataset[0][0].shape[1]
    vocabularySize = len(id2word)
    embeddingDim = weightMatrix.shape[1]
    embedding_shape = weightMatrix.shape
    paddingID = word2id['<pad>']

    # setup Transformer
    print("Initialising Transformer..", end=" ")
    model = Transformer(vocabularySize, maxReviewLength, vocabularySize, maxReviewLength).to(device)
    optimiser = optim.Adam(model.parameters(), lr=learningRate)
    criterion = nn.NLLLoss()
    print("Done.")

    # train model
    train(model, dataset, epochs, optimiser, criterion)