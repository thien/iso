import os
from io import open
import torch
import pickle
import numpy as np

from tempfile import mkdtemp
from torch.autograd import Variable
import torch.nn.functional as F

# plotting
import matplotlib
matplotlib.use('Agg')

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from shutil import copyfile

import csv

def copyComponentFile(folder_path, dataset_parameters_file="dataset_parameters.json"):
    # copy the dataset parameters into the model directory so we have an idea on
    # what dataset the model parameters are trained with.
    copyfile(dataset_parameters_file, os.path.join(folder_path, dataset_parameters_file))

def printParameters(parameters):
    """
    Pretty print parameters in the cli.
    """
    maxLen = max([len(k) for k in parameters])
    for key in parameters:
        padding = " ".join(["" for _ in range(maxLen - len(key) + 5)])
        print(key + padding, parameters[key])

def initiateDirectory(folder_path):
    # create directory as it does not exist yet.
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

def saveLossMeasurements(epoch, folder_path, reconstruction, kl, aux, filename="results"):
    # create container of losses.
    container = np.array([reconstruction, kl, aux])
    container = np.transpose(container)
    path = os.path.join(folder_path, filename)
    path = initiateDirectory(path)
    # setup filename 
    filename = "epoch_" + str(epoch) + filename + ".csv"
    filepath = os.path.join(path, filename)
    # dump
    np.savetxt(filepath, container, delimiter=", ",  fmt="%.4f")

def loadDataset(path='../Datasets/Reviews/dataset_ready.pkl'):
    return pickle.load(open(path, 'rb'))

def sequence_mask(sequence_length, max_len=None):
    """
    Caution: Input and Return are VARIABLE.
    """
    sequence_length = torch.tensor(sequence_length)
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand
    
    return mask

def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
        
    The code is same as:
    
    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # Note: mask need to bed casted to float!
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()
    
    # (batch_size * max_tgt_len,)
    pred_flat = log_probs_flat.max(1)[1]
    # (batch_size * max_tgt_len,) => (batch_size, max_tgt_len) => (max_tgt_len, batch_size)
    pred_seqs = pred_flat.view(*target.size()).transpose(0,1).contiguous()
    # (batch_size, max_len) => (batch_size * max_tgt_len,)
    mask_flat = mask.view(-1)
    
    # `.float()` IS VERY IMPORTANT !!!
    # https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = length.data.sum()

    return loss, pred_seqs, num_corrects, num_words

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    mu_1, var_1 = recog_mu, recog_logvar
    mu_2, var_2 = prior_mu, prior_logvar
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    kld = -0.5 * torch.sum(1 + (var_1 - var_2)
                           - torch.div(torch.pow(mu_2 - mu_1,
                                                 2), torch.exp(var_2))
                           - torch.div(torch.exp(var_1), torch.exp(var_2)), 1)
    return kld

def bow_loss(future_y_labels, y_mask, pred_bow):
    # cvae implementation
    bow_guess = pred_bow.gather(1, future_y_labels)
    bow_guess = bow_guess * y_mask
    bow_loss = torch.sum(bow_guess, 1)
    avg_bow_loss = torch.mean(bow_loss)
    return avg_bow_loss
    # else:
    #     bow_loss = ref_bow * pred_bow
    #     bow_loss = torch.sum(bow_loss, dim=1)
    #     avg_bow_loss = torch.mean(bow_loss)
    #     return avg_bow_loss

def recon_loss(y_predicted, y):
    y_onehot = torch.tensor(batch_size, num_classes).zero_()
    return None

def loss_function(epoch,
                  batch_num,
                  num_batches,
                  y_predicted,
                  y,
                  inference_mu,
                  inference_logvar,
                  prior_mu,
                  prior_logvar,
                  future_y_labels,
                  ref_bow_t_mask,
                  ref_bow_mask,
                  pred_bow,
                  use_latent=True):

    # compute reconstruction loss
    # ll_loss = criterion_r(y_predicted, y)
    ll_loss = F.cross_entropy(y_predicted.view(-1, y_predicted.size(-1)), y.reshape(-1), reduction='none').view(y_predicted.size()[:-1])
    # print(ll_loss.shape, ref_bow_t_mask.shape)
    ll_loss = torch.mean(ll_loss * ref_bow_t_mask)
    # compute mean
    # ll_loss = ll_loss.mean()

    # compute KLD
    kl_loss = 0
    if use_latent:
        kl_loss = gaussian_kld(inference_mu, inference_logvar, prior_mu, prior_logvar)
        kl_loss = torch.mean(kl_loss)
        # KL Annealing
        if epoch < 2:
            kl_cap = 1000
            # print(epoch, num_batches, batch_num)
            kl_weight_count = max((epoch-1)*num_batches,0) + batch_num
            # print("KL_WEIGHT CAP:", kl_weight_count)
            kl_weight = kl_weight_count / kl_cap
            kl_loss *= kl_weight

    aux_loss = 0
    if use_latent:
        # compute auxillary loss
        aux_loss = bow_loss(future_y_labels, ref_bow_mask, pred_bow)
        # weight auxillary loss
        alpha = 10
        aux_loss *= alpha

    return ll_loss, kl_loss, aux_loss

def plotBatchLoss(iteration, losses, kl, aux, folder_path):
    x = [i for i in range(1, len(losses)+1)]
    labels = ["LL", "KL", "Auxiliary"]

    plt.stackplot(x, losses, kl, aux, labels=labels)
    plt.legend()
    # plt.ylim(top=15)
    title = 'Learning Loss during Iteration ' + str(iteration)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Batch Number')

    filetype = "png"
    directory = "charts"
    filename = title + "." + filetype
    directory = os.path.join(folder_path, directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def padSeq(row, maxlength, padID, backwards=False):
    currentLength = len(row)
    difference = maxlength - currentLength
    if backwards:
        return row[::-1] + [padID for _ in range(difference)]
    else:
        return row + [padID for _ in range(difference)]

def batchData(dataset, padID, device, batchsize=32, cutoff=50, backwards=False):
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
        lengths = [len(seq) for seq in batch]
        indexes = [x for x in range(len(lengths))]
        sortedindexes = sorted(list(zip(lengths, indexes)), reverse=True)

        # since sentences are split by period, the period itself acts
        # the token to identify that the sentence has ended.
        # i.e. we don't need another token identifying the end of the subsequence.

        # get the reviews based on the sorted batch lengths

        if not backwards:
            reviews = [padSeq(batch[i[1]], cutoff, padID) for i in sortedindexes]
        else:
            reviews = [padSeq(batch[i[1]], cutoff, padID, backwards) for i in sortedindexes]

        reviews = torch.tensor(reviews, dtype=torch.long)
        # re-allocate values.
        batches[i] = (reviews, [i[0] for i in sortedindexes])
    return batches

def saveModels(encoder, backwards, decoder, filepath):
    print("Saving models..", end=" ")
    torch.save(encoder.state_dict(),  os.path.join(filepath, 'encoder.pth'))
    torch.save(backwards.state_dict(),  os.path.join(filepath, 'backwards.pth'))
    torch.save(decoder.state_dict(),  os.path.join(filepath, 'decoder.pth'))
    print("Done.")

def saveModel(vad, filepath):
    print("Saving model..", end=" ")
    torch.save(vad.state_dict(), os.path.join(filepath, 'vad.pth'))
    print("Done.")

def saveEvalOutputs(folder_path, results, epochs, folder_name="outputs"):
    output_dir = os.path.join(folder_path, folder_name)
    output_dir = initiateDirectory(output_dir)
    filename = "epoch_" + str(epochs) + ".csv"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as myfile:
        wr = csv.writer(myfile, delimiter='\t', quoting=csv.QUOTE_ALL)
        wr.writerows(results)

def responseID2Word(id2word, outputs):
    entries = []
    for batch_line in outputs:
        entry = [torch.argmax(batch_line[i]).cpu().item() for i in range(len(batch_line))]
        entries.append([id2word[i] for i in entry])
     
    words = []
    for i in range(len(outputs[0])):
        tokens = [entries[j][i] for j in range(len(entries))]
        # find the eos token
        try:
            tokenpos = tokens.index("<eos>")
        except:
            tokenpos = len(tokens)
        # remove extra eos tokens and padding values if they exist.
        words.append(" ".join(tokens[:tokenpos+1]))
    return words