from vad import Encoder, Backwards, Attention, Decoder, Inference, Prior, CBOW
from vad_utils import loss_function, plotBatchLoss, batchData, loadDataset, saveModels

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

# setup default seeds so we can repeat the outcomes
seed = 1337
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def evalVAD(x,
             xLength,
             y,
             yLength,
             encoder,
             decoder,
             device,
             criterion,
             word2id
            ):
    
    loss = 0
    
    # initalise input and target lengths
    inputLength = x[0].size(0)
    targetLength = y[0].size(0)
    batchSize = x.shape[0]

    # set up encoder computation
    encoderHidden = encoder.initHidden(batchSize).to(device)
    
    # set up encoder outputs
    encoderOutputs, encoderHidden = encoder(x, encoderHidden, xLength)

    # set up the variables for decoder computation
    decoderInput = torch.tensor([word2id["<sos>"]] * batchSize, dtype=torch.long, device=device)
    
    decoderHidden = encoderHidden[-1]
    decoderOutput = None
    decoderOutputs = []
    
    # Run through the decoder one step at a time. This seems to be common practice across
    # all of the seq2seq based implementations I've come across on GitHub.
    for t in range(yLength[0]):
        # compute the output of each decoder state
        decoderOutput, decoderHidden = decoder(
            decoderInput, 
            encoderOutputs,  
            xLength, 
            decoderHidden, 
            device,
            back=None)

        decoderOutputs.append(decoderOutput)

        decoderInput = decoderOutput.argmax(1)
        decoderHidden = decoderHidden.squeeze(0)
        
    return decoderOutputs, loss

def trainVAD(
    device,
    epoch,
    batch_num,
    num_batches,
    x,
    y,
    yb,
    xLength,
    yLength,
    ybLength,
    encoder,
    backwards,
    decoder,
    encoderOpt,
    backwardsOpt,
    decoderOpt,
    word2id,
    criterion_r,
    criterion_bow,
    useBOW,
    gradientClip,
    use_latent
    ):
    """
    Represents a whole sequence iteration trained on a batch of reviews.
    """

    # initialise gradients
    encoderOpt.zero_grad()
    backwardsOpt.zero_grad()
    decoderOpt.zero_grad()

    # set default loss
    loss, ll_loss, kl_loss, aux_loss = 0, 0, 0, 0

    # initalise input and target lengths, and vocab size
    ySeqlength, batchSize = yLength[0], x.shape[0]
    vocabularySize = decoder.embedding.weight.shape[0]

    # set up encoder and backward hidden vectors
    encoderHidden = encoder.initHidden(batchSize).to(device)
    backwardHidden = backwards.initHidden(batchSize).to(device)

    # set up encoder outputs
    encoderOutputs, encoderHidden = encoder(x, encoderHidden, xLength)

    # compute backwards outputs
    backwardOutput, backwardHidden = backwards(yb, ybLength, backwardHidden)

    # set up the variables for decoder computation
    decoderInput = torch.tensor(
        [word2id["<sos>"]] * batchSize, dtype=torch.long, device=device)

    decoderHidden = torch.sum(encoderHidden, dim=0)
    decoderOutput = None

    # randomly determine teacher forcing with equal probability
    teacher_forcing = True if random.random() > 0.5 else False

    # Run through the decoder one step at a time. This seems to be common practice across
    # all of the seq2seq based implementations I've come across on GitHub.
    for t in range(ySeqlength):
        # compute the output of each decoder state
        out = decoder(decoderInput, 
                      encoderOutputs,
                      xLength,
                      decoderHidden, 
                      device,
                      back=backwardOutput[:, t]
                      )

        # update variables
        decoderOutput, decoderHidden, pred_bow, infer_mu, infer_logvar, prior_mu, prior_logvar = out

        # compute reference CBOW to compare our predictions to.
        ref_bow_mask = (y[:, t:] != word2id['<pad>']).float()

        # calculate the loss
        ll, kl, aux = loss_function(
            epoch,
            batch_num,
            num_batches,
            decoderOutput,
            y[:, t],
            infer_mu,
            infer_logvar,
            prior_mu,
            prior_logvar,
            y[:, t:],
            ref_bow_mask,
            pred_bow,
            criterion_r,
            criterion_bow,
            use_latent)

        loss += ll + kl + aux
        ll_loss += ll
        kl_loss += kl
        aux_loss += aux

        if teacher_forcing:
            # feed this output to the next input
            decoderInput = y[:, t]
        else:
            decoderInput = torch.argmax(decoderOutput, dim=1)
        decoderHidden = decoderHidden.squeeze(0)

    # normalise loss
    avg_loss = loss/ySeqlength

    # calculate gradients
    avg_loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(encoder.parameters(),   gradientClip)
    torch.nn.utils.clip_grad_norm_(backwards.parameters(), gradientClip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(),   gradientClip)

    # gradient descent
    decoderOpt.step()
    backwardsOpt.step()
    encoderOpt.step()

    # return avg losses for plotting.
    avg_llloss = ll_loss.item()/ySeqlength
    avg_klloss, avg_auxloss = 0,0
    if use_latent:
        avg_klloss = kl_loss.item()/ySeqlength
        avg_auxloss = aux_loss.item()/ySeqlength

    return avg_loss.item(), avg_llloss, avg_klloss, avg_auxloss

def trainIteration(
        device,
        dataset,
        encoder,
        backwards,
        decoder,
        iterations,
        word2id,
        criterion_r,
        criterion_bow,
        learningRate,
        gradientClip,
        useBOW,
        folder_path,
        use_latent,
        printEvery=10,
        plotEvery=100):

    plotLosses = []
    printLossTotal = 0
    plotLossTotal = 0

    encoderOpt   = optim.Adam(encoder.parameters(),   lr=learningRate)
    backwardsOpt = optim.Adam(backwards.parameters(), lr=learningRate)
    decoderOpt   = optim.Adam(decoder.parameters(),   lr=learningRate)

    numBatches = len(dataset[0])

    for j in range(1, iterations + 1):
        print("Epoch", j)
        # set up variables needed for training.
        n = -1

        # get random indexes
        indexes = [i for i in range(numBatches)]
        random.shuffle(indexes)

        # note that the data entries in each batch are sorted!
        # we're shuffling the batches.

        losses = []
        ll_losses = []
        kl_losses = []
        aux_losses = []
        for batch_num in tqdm(indexes):
            n += 1
            # each batch is composed of the
            # reviews, and a sentence length.
            x,   xLength = dataset[0][batch_num][0], dataset[0][batch_num][1]
            y,   yLength = dataset[1][batch_num][0], dataset[1][batch_num][1]
            yb, ybLength = dataset[2][batch_num][0], dataset[2][batch_num][1]

            # send to cuda.
            x, y, yb = x.to(device), y.to(device), yb.to(device)
            # calculate loss.
            loss, ll, kl, aux = trainVAD(
                device,
                j,
                n,
                numBatches,
                x, y, yb, 
                xLength,
                yLength,
                ybLength,
                encoder,
                backwards,
                decoder,
                encoderOpt,
                backwardsOpt,
                decoderOpt,
                word2id,
                criterion_r,
                criterion_bow,
                useBOW,
                gradientClip,
                use_latent
            )

            print("Batch:", n, "Loss:", round(loss, 4), "LL:", round(
                ll, 4), "KL:", round(kl, 4), "AUX:", round(aux, 4))

            # increment our print and plot.
            printLossTotal += loss
            plotLossTotal += loss

            losses.append(loss)
            ll_losses.append(ll)
            kl_losses.append(kl)
            aux_losses.append(aux)

            if batch_num % 10 == 0:
                plotBatchLoss(j, ll_losses, kl_losses, aux_losses, folder_path)

        saveModels(encoder, backwards, decoder, folder_path)

def printParameters(parameters):
    """
    Pretty print parameters in the cli.
    """
    maxLen = max([len(k) for k in parameters])
    print("Parameters:")
    for key in parameters:
        padding = " ".join(["" for _ in range(maxLen - len(key) + 5)])
        print(key + padding, parameters[key])

def initiateDirectory(folder_path):
    # create directory as it does not exist yet.
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

def copyDatasetParams(folder_path, dataset_parameters_file="dataset_parameters.json"):
    # copy the dataset parameters into the model directory so we have an idea on
    # what dataset the model parameters are trained with.
    copyfile(dataset_parameters_file, os.path.join(folder_path, dataset_parameters_file))

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
        }
    else:
        parameters = {
            'hiddenSize'			: 350,
            'latentSize'			: 300,
            'batchSize'				: 64,
            'iterations'			: 200,
            'learningRate'			: 0.0001,
            'gradientClip'			: 1,
            'useBOW'				: True,
            'bidirectionalEncoder'	: True,
            'reduction'             : 8,
            'device'                : "cuda",
            'useLatent'             : True,
        }
    return parameters

def initiate(parameters=None, model_base_dir="models"):
    """
    Loads model parameters, setup file directories and trains
    model with specified params.
    """

    # load default parameters if they don't exist.
    if parameters is None:
        parameters = defaultParameters()

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
    device = torch.device(parameters['device'])

    printParameters(parameters)

    folder_path = initiateDirectory(folder_path)
    # copy dataset parameters
    copyDatasetParams(folder_path)
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
    cutoff = dataset['cutoff']
    paddingID = word2id['<pad>']


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

    modelEncoder = Encoder(embedding, vocabularySize,
                           paddingID, hiddenSize, bidirectionalEncoder).to(device)

    modelBackwards = Backwards(embedding, vocabularySize,
                               paddingID, hiddenSize, bidirectionalEncoder).to(device)

    # https://discuss.pytorch.org/t/confusion-about-nested-modules-and-shared-parameters/26212/2
    modelDecoder = Decoder(embedding, vocabularySize,
                           paddingID, batchSize, maxReviewLength, hiddenSize, latentSize, bidirectionalEncoder).to(device)

    criterion_r   = nn.CrossEntropyLoss(ignore_index=paddingID)
    criterion_bow = nn.BCEWithLogitsLoss()

    print("Done.")

    trainIteration(device,
                   traindata,
                   modelEncoder,
                   modelBackwards,
                   modelDecoder,
                   iterations,
                   word2id,
                   criterion_r,
                   criterion_bow,
                   learningRate,
                   gradientClip,
                   useBOW,
                   folder_path,
                   parameters['useLatent'],
                   printEvery=1000)

    saveModels(modelEncoder, modelBackwards, modelDecoder, folder_path)
    
if __name__ == "__main__":
    initiate()