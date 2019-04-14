from vad import VAD
from vad_utils import loss_function, plotBatchLoss, batchData, loadDataset, saveModel, saveLossMeasurements, initiateDirectory, printParameters, copyComponentFile, saveEvalOutputs, responseID2Word

import os
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import json
from shutil import copyfile


def defaultParameters():
    """
    Initiates default parameters if none is present.
    """

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
            'reduction'             : 512,
            'device'                : "cpu",
            'useLatent'             : True,
            'teacherTrainingP'      : 0.3
        }
    else:
        parameters = {
            'hiddenSize'			: 512,
            'latentSize'			: 400,
            'batchSize'				: 32,
            'iterations'			: 50,
            'learningRate'			: 0.0001,
            'gradientClip'			: 1,
            'useBOW'				: True,
            'bidirectionalEncoder'	: True,
            'reduction'             : 8,
            'device'                : "cuda",
            'useLatent'             : True,
            'teacherTrainingP'      : 1.0
        }

    return parameters

def initiate(parameters, model_base_dir="models"):
    
    # by default we set the folder_path folder to the current datetime
    folder_path = datetime.datetime.now().strftime("%Y%m%d %H-%M-%S")
    folder_path = os.path.join(model_base_dir, folder_path)

    # initiate parameter variables
    hiddenSize = parameters['hiddenSize']
    latentSize = parameters['latentSize']
    batchSize = parameters['batchSize']
    iterations = parameters['iterations']
    learningRate = parameters['learningRate']
    gradientClip = parameters['gradientClip']
    useBOW = parameters['useBOW']
    bidirectionalEncoder = parameters['bidirectionalEncoder']
    reduction = parameters['reduction']
    teacherTrainingP = parameters['teacherTrainingP']
    useLatent = parameters['useLatent']
    device = torch.device(parameters['device'])

    printParameters(parameters)

    folder_path = initiateDirectory(folder_path)
    # copy dataset parameters
    copyComponentFile(folder_path)

    # copy the other files.
    backup_dir = os.path.join(folder_path, "backups")
    backup_dir = initiateDirectory(backup_dir)

    component_files = ['train_vad.py', 'vad.py', 'train_vad_new.py', 'vad_utils.py', 'vad_evaluate.ipynb', 'Data Preprocessing.ipynb', 'Dataset.ipynb']
    for component in component_files:
        copyComponentFile(backup_dir, component)

    # save model parameters
    param_jsonpath = os.path.join(folder_path, "model_parameters.json")
    with open(param_jsonpath, 'w') as outfile:
        json.dump(parameters, outfile)

    print("Loading dataset..", end=" ")
    dataset = loadDataset()
    # setup store parameters
    id2word = dataset['id2word']
    word2id = dataset['word2id']
    weightMatrix = dataset['weights']
    train = dataset['train']
    val = dataset['validation']
    cutoff = dataset['cutoff']
    paddingID = word2id['<pad>']
    sosID = word2id['<sos>']

    print("Converting dataset weights into tensors..", end=" ")
    # convert dataset into tensors
    weightMatrix = torch.tensor(weightMatrix, dtype=torch.float)
    print("Done.")

    # batching data
    print("Batching Data..", end=" ")

    # shuffle data rows and split data s.t they can be processed.
    random.shuffle(train)
    raw_x, raw_y = [x[0] for x in train[::reduction]], [x[1] for x in train[::reduction]]
    # batchify data.
    trainx = batchData(raw_x, paddingID, device, batchSize, cutoff)
    trainy = batchData(raw_y, paddingID, device, batchSize, cutoff)
    # the unpadded sequence is used for the backwards rnn.
    trainy_back = batchData(raw_y, paddingID, device, batchSize, cutoff, backwards=True)

    traindata = (trainx, trainy, trainy_back)

    val_x, val_y = [x[0] for x in val[::reduction]], [x[1] for x in val[::reduction]]

    valx = batchData(val_x, paddingID, device, batchSize, cutoff)
    valy = batchData(val_y, paddingID, device, batchSize, cutoff)
    valdata = (valx, valy)
    print("Done.")

    # setup variables for model components initialisation
    maxReviewLength = cutoff
    vocabularySize = len(id2word)
    embeddingDim = weightMatrix.shape[1]
    embedding_shape = weightMatrix.shape

    print("Initialising model components..", end=" ")

    embedding = nn.Embedding(
            num_embeddings=vocabularySize,
            embedding_dim=embeddingDim,
            padding_idx=paddingID,
        _weight=weightMatrix
        ).to(device) 

    model = VAD(embedding, 
            paddingID,
            sosID,
            hiddenSize,
            vocabularySize,
            latentSize,
            useLatent,
            maxReviewLength,
            bidirectionalEncoder,
            teacherTrainingP,
            useBOW=useBOW).to(device)
    model.device = device

    criterion_r   = nn.CrossEntropyLoss(ignore_index=paddingID)
    criterion_bow = nn.BCEWithLogitsLoss()

    # setup model optimiser
    optimiser = optim.Adam(model.parameters(), lr=learningRate)
    print("Done.")

    trainModel(device,
               traindata,
               valdata,
               model,
               optimiser,
               iterations,
               learningRate,
               word2id,
               id2word,
               criterion_r,
               criterion_bow,
               gradientClip,
               folder_path
            )

    saveModel(model, folder_path)

def trainModel(device,
               traindata,
               valdata,
               model,
               optimiser,
               numEpochs,
               learningRate,
               word2id,
               id2word,
               criterion_r,
               criterion_bow,
               gradientClip,
               folder_path):
    
    numTrainingBatches = len(traindata[0])
    numEvalBatches = len(valdata[0])
    model.numBatches = numTrainingBatches

    for epoch in tqdm(range(0,numEpochs)):

        model.train()
        model.epoch = epoch
        # create random indexes s.t we shuffle the training batches.
        indexes = [i for i in range(numTrainingBatches)]
        random.shuffle(indexes)
        # setup loss containers.
        losses, ll_losses, kl_losses, aux_losses = [], [], [], []
        # iterate through the training batches
        for n in tqdm(range(len(indexes))):
            model.batchNum = n
            # load batch number based on the shuffled list
            batch_number = indexes[n]
            # each batch is composed of the reviews, and a sentence length.
            
            x,   xLength = traindata[0][n][0], traindata[0][n][1]
            y,   yLength = traindata[1][n][0], traindata[1][n][1]
            yb, ybLength = traindata[2][n][0], traindata[2][n][1]
            # send to cuda (if warranted)
            x, y, yb = x.to(device), y.to(device), yb.to(device)

            # get output and loss values
            input_data = ((x,xLength), (y,yLength), (yb, ybLength))
            _, loss_container = model(input_data, loss_function)
            loss, ll, kl, aux = loss_container
            # gradient descent step
            optimiser.zero_grad()
            loss.backward()
            # gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradientClip)
            optimiser.step()
            # add losses.
            losses.append(loss.item())
            ll_losses.append(ll)
            kl_losses.append(kl)
            aux_losses.append(aux)
            
            # print("LOSS:",loss.item(), "LL:", ll, "KL:", kl, "AUX:",aux)

            if n % 10 == 0:
                plotBatchLoss(epoch, ll_losses, kl_losses, aux_losses, folder_path)
                saveLossMeasurements(epoch, folder_path, ll_losses, kl_losses, aux_losses)

        # clear memory
        del x,y,yb
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # ---------------------------------

        # save at the end of the training round.
        saveLossMeasurements(epoch, folder_path, ll_losses, kl_losses, aux_losses)
        saveModel(model, folder_path)

        # ---------------------------------

        # evaluate model
        model.eval()
        
        results = []
        for n in tqdm(range(0,numEvalBatches)):
            x,   xLength = valdata[1][n][0], valdata[1][n][1]
            y,   yLength = valdata[1][n][0], valdata[1][n][1]
            x, y = x.to(device), y.to(device)
            # get output and loss values
            input_data = ((x,xLength), (y,yLength))
            responses  = model(input_data)
            results.append(responses)

        results = [responseID2Word(id2word, resp) for resp in results]
        # write the responses to file.
        saveEvalOutputs(folder_path, results, epoch)

if __name__ == "__main__":
    # load default parameters if they don't exist.
    parameters = defaultParameters()

    initiate(parameters)