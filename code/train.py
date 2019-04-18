from vad import VAD
from vad_utils import *
from vad_statistics import Statistics
import argparse

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
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

class Trainer:
    def __init__(self, args, model_base_dir="models"):
        self.args = args
        self.args.model_parent_dir = model_base_dir

        self.traindata = None
        self.valdata = None
        self.vocabularySize = None
        self.word2id = None
        self.id2word = None
        self.eosID = None
        self.sosID = None
        self.paddingID = None
        self.unkID = None
        self.embeddingDim = None
        self.cutoff = None # maximum length of a sequence
        self.device = None

    def setup(self):
        self.args.folder_name = self.args.dataset + " " + self.args.name
        self.folder_path = os.path.join(self.args.model_parent_dir, self.args.folder_name)
        self.device = torch.device(self.args.device)
        printParameters(vars(self.args))
     
        if self.args.save:
            initiateDirectory(self.folder_path)
            self.backup()

        self.setupDataset()
        self.setupModel()
        self.train()

        if self.args.save:
            saveModel(self.model, self.folder_path)
            self.runStats()

    def setupDataset(self):
        if self.args.dataset == "penn":
            self.loadPenn()
        else:
            self.loadSet()

    def setupModel(self):
        # setup embedding
        embedding = nn.Embedding(
            num_embeddings=self.vocabularySize,
            embedding_dim=self.embeddingDim,
            padding_idx=self.paddingID,
        ).to(self.device) 
        if self.args.pretrained_weights and (self.args.dataset != "penn"):
            embedding.load_state_dict({'weight': self.embeddingWeights})
        
        if self.args.model == "vad":
            self.model = VAD(embedding,
                    self.paddingID,
                    self.sosID,
                    self.args.hidden_size,
                    self.vocabularySize,
                    self.args.latent_size,
                    self.args.use_latent,
                    self.cutoff,
                    True, # use bidirectional encoder
                    self.args.teacher_training_p,
                    useBOW=self.args.useBOW).to(self.device)
            self.model.device = self.device
            self.criterion_r = None #(could also be criterion_r = nn.CrossEntropyLoss(ignore_index=paddingID))
            self.criterion_bow = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("You didn't implement other models.")
        # setup loss functions and optimiser
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    
    def backup(self):
        """
        Copy parameter json and files for backup
        """
        # copy dataset parameters
        if self.args.dataset != "penn":
            if self.args.dataset == "amazon":
                filename = "dataset_parameters.json"
                directory = self.args.amazon_path
            elif self.args.dataset == "subtitles":
                filename = "subtitles_parameters.json"
                directory = self.args.subtitles_path
    
            copyComponentFile(self.folder_path, filename, directory)

        # copy the other files.
        backup_dir = os.path.join(self.folder_path, "backups")
        backup_dir = initiateDirectory(backup_dir)

        component_files = ['vad.py', 'train.py', 'vad_utils.py', 'vad_evaluate.ipynb', 'Data Preprocessing.ipynb', 'Dataset.ipynb']
        for component in component_files:
            copyComponentFile(backup_dir, component)

        # save model parameters
        param_jsonpath = os.path.join(self.folder_path, "model_parameters.json")
        with open(param_jsonpath, 'w') as outfile:
            json.dump(vars(self.args), outfile)
   
    def train(self):
        if self.args.dataset != "penn":
            numTrainingBatches = len(self.traindata[0])
            numEvalBatches = len(self.valdata[0])
        else:
            numTrainingBatches = len(self.traindata)
            numEvalBatches = len(self.valdata)

        self.model.numBatches = numTrainingBatches
        # setup indexes for each epoch
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            self.model.epoch = epoch
            train_loader = DataLoader(
                dataset=self.traindata,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=cpu_count()-1,
                pin_memory=torch.cuda.is_available()
            )
            # setup loss containers
            losses, ll_losses, kl_losses, aux_losses = [], [], [], []
            # iterate through the training batches
            for n, batch in enumerate(tqdm(train_loader)):
                self.model.batchNum = n

                # print(batch['input'])

                batch['input']   = batch['input'].to(self.device)
                batch['target']  = batch['target'].to(self.device)
                batch['reverse'] = batch['reverse'].to(self.device)
        
                # get output and loss values
                _, loss_container = self.model(batch, loss_function, self.criterion_r)
                loss, ll, kl, aux = loss_container
                # gradient descent step
                self.optimiser.zero_grad()
                loss.backward()
                # gradient clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                self.optimiser.step()
                # add losses.
                losses.append(loss.item())
                ll_losses.append(ll)
                kl_losses.append(kl)
                aux_losses.append(aux)
                
                if self.args.save and (n % 10 == 0):
                    plotBatchLoss(epoch, ll_losses, kl_losses, aux_losses, self.folder_path)
                    saveLossMeasurements(epoch, self.folder_path, ll_losses, kl_losses, aux_losses)
            
            # ---------------------------------
            # save at the end of the training round.
            if self.args.save:
                saveLossMeasurements(epoch, self.folder_path, ll_losses, kl_losses, aux_losses)
                saveModel(self.model, self.folder_path)

            # ---------------------------------

            val_loader = DataLoader(
                dataset=self.valdata,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=cpu_count()-1,
                pin_memory=torch.cuda.is_available()
            )

            # evaluate model
            self.model.eval()
            
            results = []
            with torch.no_grad():
                for n, batch in enumerate(tqdm(val_loader)):
                    self.batchNum = n
                    batch['input'] = batch['input'].to(self.device)
                    # get output and loss values
                    responses = self.model(batch)
                    responses = [entry.detach().cpu() for entry in responses]
                    responses = responseID2Word(self.id2word, responses)
                    results.append(responses)

            if self.args.save:
                saveEvalOutputs(self.folder_path, results, epoch)

            if self.model.device.type == "cuda":
                torch.cuda.empty_cache()

    """
    data loaders.
    """
    def loadPenn(self):
        """
        setup penn dataset loader.
        """

        from penn_utils import to_var, idx2word, expierment_name, PTB

        # setup variables for model components initialisation
        self.cutoff = 60

        self.traindata = PTB(
            data_dir=self.args.penn_path,
            split='train',
            create_data='store_true',
            max_sequence_length=self.cutoff,
            min_occ=1
        )

        self.valdata = PTB(
            data_dir=self.args.penn_path,
            split='valid',
            create_data='store_true',
            max_sequence_length=self.cutoff,
            min_occ=1
        )

        with open(os.path.join(self.args.penn_path,'ptb.vocab.json'), 'r') as file:
            vocab = json.load(file)

        self.word2id, id2word = vocab['w2i'], vocab['i2w']
        self.id2word = {int(key) : id2word[key] for key in id2word.keys()}
        self.vocabularySize = self.traindata.vocab_size
        self.sosID = self.traindata.sos_idx
        self.eosID = self.traindata.eos_idx
        self.paddingID = self.traindata.pad_idx
        self.unkID = self.traindata.unk_idx
        self.embeddingDim = 50

    def loadSet(self):
        """
        Handles the Amazon and the Subtitles dataset.
        """
        print("Loading",self.args.dataset,"dataset..", end=" ")
        if self.args.dataset == "subtitles":
            dataset = loadDataset(os.path.join(self.args.subtitles_path,"dataset_ready.pkl"))
        else:
            dataset = loadDataset(os.path.join(self.args.amazon_path,"dataset_ready.pkl"))
        
        # setup store parameters
        self.id2word = dataset['id2word']
        self.word2id = dataset['word2id']
        weightMatrix = dataset['weights']
        train = dataset['train']
        val = dataset['validation']
        self.cutoff = dataset['cutoff']
        self.paddingID = self.word2id['<pad>']
        self.sosID = self.word2id['<sos>']

        # convert dataset into tensors
        self.embeddingWeights = torch.tensor(weightMatrix, dtype=torch.float)
        self.vocabularySize = self.embeddingWeights.shape[0]
        self.embeddingDim = self.embeddingWeights.shape[1]
     
        # batching data
        self.traindata = prepDataset(train, self.paddingID, self.cutoff, train=True, step=self.args.reduction)
        self.valdata   = prepDataset(val,   self.paddingID, self.cutoff, train=False, step=self.args.reduction)
        print("Done.")

    """
    Statistic handlers.
    """
    def runStats(self):
        # once we're done with training we can do some stats!
        stat = Statistics(self.args.folder_name)
        stat.loadModelParameters()
        stat.loadResults()
        stat.val_loader = DataLoader(
            dataset=self.valdata,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=cpu_count()-1
        )
        stat.id2word = self.id2word
        stat.parseDataLoader()
        stat.loadModelOutputs()
        stat.plotChart(step=50, ylim=5)
        statdump = stat.batchComputeBLEUROUGE()
        stat.chartBLEUROUGE(statdump)
        stat.dumpStats(statdump)

def defaultModelName():
    return datetime.datetime.now().strftime("%Y%m%d %H-%M-%S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--name','-n', type=str, default=defaultModelName())
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--model', type=str, default='vad')
    parser.add_argument('--hidden_size', "-hs", type=int, default=512)
    parser.add_argument('--latent_size', "-ls", type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs','-ep',  type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--gradient_clip', '-cl', type=int, default=1)
    parser.add_argument('--useBOW','-bow', type=bool, default=True)
    parser.add_argument('--reduction', '-r', type=int, default=12)
    parser.add_argument('--device', '-d', type=str, default="cuda")
    parser.add_argument('--use_latent', '-ul', type=bool, default=True)
    parser.add_argument('--teacher_training_p', '-tp', type=float, default=1.0)
    parser.add_argument('--pretrained_weights', '-w', type=bool, default=True)
    parser.add_argument('--save', '-s', type=bool, default=True)
    parser.add_argument('--penn_path', type=str, default= '../Datasets/Penn')
    parser.add_argument('--amazon_path', type=str, default= '../Datasets/Reviews')
    parser.add_argument('--subtitles_path', type=str, default= '../Datasets/OpenSubtitles')

    args = parser.parse_args()

    assert args.model   in ['vad', 'seq2seq', 'bowman']
    assert args.dataset in ['amazon', 'penn', 'subtitles']
 
    brock = Trainer(args)
    brock.setup()
    brock.train()