from vad import VAD
from vad_utils import *
from vad_statistics import Statistics
import svae
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
        self.args.folder_name = self.args.name + " " + self.args.dataset + " " + self.args.model
        self.folder_path = os.path.join(self.args.model_parent_dir, self.args.folder_name)
        self.device = torch.device(self.args.device)
        printParameters(vars(self.args))
     
        if self.args.save:
            initiateDirectory(self.folder_path)
            self.backup()

        self.setupDataset()
        self.setupModel()

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
            self.criterion_r = None
            self.criterion_bow = nn.BCEWithLogitsLoss()
        elif self.args.model == "seq2seq":
            # since the VAD collapses into a seq2seq with attention
            # when the KL vanishes, we can force the latent variable to
            # learn nothing.
            self.model = VAD(embedding,
                    self.paddingID,
                    self.sosID,
                    self.args.hidden_size,
                    self.vocabularySize,
                    1,
                    False,
                    self.cutoff,
                    True, # use bidirectional encoder
                    self.args.teacher_training_p,
                    useBOW=False).to(self.device)
            self.model.device = self.device
            self.criterion_r = None
            self.criterion_bow = nn.BCEWithLogitsLoss()
        elif self.args.model == "bowman":
            # setup Sentence VAE model described by bowman.
            self.model = svae.SentenceVAE(
                embedding,
                self.vocabularySize,
                self.embeddingDim,
                self.args.hidden_size,
                self.args.latent_size,
                0,
                self.sosID,
                self.eosID,
                self.paddingID,
                self.unkID,
                self.cutoff
            ).to(self.device)
            self.model.device = self.device
        else:
            raise ValueError("You didn't implement other models.")
        self.model.k = self.args.k
        self.model.x0 = self.args.x0
        # setup loss functions and optimiser
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    
    def loadModel(self):
        # look up model parameters
        self.folder_path = os.path.join(self.args.model_parent_dir, self.args.name)
        if not os.path.isdir(self.folder_path):
            raise Exception("This model folder does not exist.")
        # load model parameters
        param_path = os.path.join(self.folder_path,"model_parameters.json")
        with open(param_path) as json_file:  
            params = json.load(json_file)
        # iterate through the params and load them into the arguments.
        for parameter in params:
            if parameter not in ['save', 'test']:
                setattr(self.args, parameter, params[parameter])
        # setup the model.
        self.device = torch.device(self.args.device)
        printParameters(vars(self.args))
        self.setupDataset()
        self.setupModel()
        # load model weights
        self.model.load_state_dict(torch.load(os.path.join(self.folder_path,'vad.pth')))

    def backup(self):
        """
        Copy parameter json and files for backup
        """
        # copy dataset parameters
        if self.args.dataset != "penn":
            filename = "dataset_parameters.json"
            if self.args.dataset == "amazon":
                directory = self.args.amazon_path
            elif self.args.dataset == "subtitles":
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
        # get number of batches in each epoch based on the type of dataset
        if self.args.dataset != "penn":
            numTrainingBatches = len(self.traindata[0])
            numEvalBatches = len(self.valdata[0])
        else:
            numTrainingBatches = len(self.traindata)
            numEvalBatches = len(self.valdata)

        if self.args.model != "bowman":
            self.model.decoder.doInference = True
        self.model.numBatches = numTrainingBatches
        # setup kl step
        step = 0
        # setup inner loop for the outer loop
        datasets = [("train", self.traindata), ("val", self.valdata)]

        # we return this value in the event the model in question is
        # being used for hyperparameter optimisation. (We keep the
        # loss at the end of the epochs.)
        hyperparam_loss = 0

        # setup indexes for each epoch
        for epoch in tqdm(range(self.args.epochs), desc="epochs"):
            # swap between train and eval models.
            for datatype in datasets:
                label_type, dataset = datatype
            
                if label_type == "train":
                    self.model.train()
                else:
                    self.model.eval()
                    results = []      

                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True if (label_type == "train") else False,
                    num_workers=cpu_count()-1,
                    pin_memory=torch.cuda.is_available()
                )
                # setup loss containers
                losses, ll_losses, kl_losses, aux_losses = [], [], [], []
                # iterate through the training batches
                for n, batch in enumerate(tqdm(dataloader, desc=label_type)):
                    if label_type == "train":
                        step += 1
                    self.model.batchNum = n

                    batch['input']   = batch['input'].to(self.device)
                    batch['target']  = batch['target'].to(self.device)
                    batch['reverse'] = batch['reverse'].to(self.device)
            
                    # get output and loss values
                    if self.args.model == "bowman":
                        responses, mean, logv, _ = self.model(batch['input'], batch['input_length'])
                        # calculate loss
                        ll = svae.recon_loss(responses, batch['target'], batch['input_length'], self.model.nll)
                        y_mask = (batch['target'] != self.paddingID).detach().float()
                        ll = torch.mean(ll * y_mask)
                        
                        kl, kl_weight = svae.KL(mean, logv, 'linear', step, self.args.k, self.args.x0)
                        aux = 0.0
                        loss = (ll + kl_weight * kl)/self.args.batch_size
                    else:
                        self.model.step = step
                        responses, loss_container = self.model(batch, loss_function, self.criterion_r)
                        loss, ll, kl, aux = loss_container

                    if label_type == "train":
                        # gradient descent step
                        self.optimiser.zero_grad()
                        loss.backward()
                        # gradient clip
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                        self.optimiser.step()

                    # compute mean loss (by the individual output) for bowman
                    if self.args.model == "bowman":
                        loss = loss/self.cutoff
                        ll = (ll.item()/self.args.batch_size)/self.cutoff
                        kl = (kl.item()/self.args.batch_size)/self.cutoff

                    # interpret outputs
                    if label_type != "train":
                        responses = [entry.detach().cpu() for entry in responses]
                        responses = responseID2Word(self.id2word, responses)
                        results.append(responses)

                    # add losses.
                    losses.append(loss.item())
                    ll_losses.append(ll)
                    kl_losses.append(kl)
                    aux_losses.append(aux)
                    
                    if label_type == "train":
                        if self.args.save and (n % 10 == 0):
                            plotBatchLoss(epoch, ll_losses, kl_losses, aux_losses, self.folder_path)
                            saveLossMeasurements(epoch, self.folder_path, ll_losses, kl_losses, aux_losses)
                
                # save at the end of the training round.
                if self.args.save:
                    if label_type == "train":
                        saveLossMeasurements(epoch, self.folder_path, ll_losses, kl_losses, aux_losses)
                        saveModel(self.model, self.folder_path)
                    else:
                        saveOutputs(self.folder_path, results, epoch)

                if label_type == "train" and self.args.hyp_opt:
                    # compute average loss
                    hyperparam_loss = sum(losses)/len(losses)

            # clear cache
            if self.model.device.type == "cuda":
                torch.cuda.empty_cache()

            if self.args.hyp_opt:
                if hyperparam_loss <= self.args.hyp_opt_loss_thresh:
                    return hyperparam_loss
        return hyperparam_loss

    def test(self):
        if self.args.model != "bowman":
            self.model.decoder.doInference = False
        self.model.eval()
        with torch.no_grad():
            test_loader = DataLoader(
                dataset=self.testdata,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=cpu_count()-1,
                pin_memory=torch.cuda.is_available()
            ) 

            for i in range(1,11):
                results = []
                for n, batch in enumerate(tqdm(test_loader)):
                    self.batchNum = n
                    batch['input'] = batch['input'].to(self.device)
                    # get output and loss values
                    
                    if self.args.model == "bowman":
                        responses, _, _, _ = self.model(batch['input'], batch['input_length'])
                    else:
                        responses = self.model(batch)
                    
                    responses = [entry.detach().cpu() for entry in responses]
                    responses = responseID2Word(self.id2word, responses)
                    results.append(responses)
                    
                saveOutputs(self.folder_path, results, i, folder_name="test_outputs")
      
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

        self.valdata = PTB(
            data_dir=self.args.penn_path,
            split='test',
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
        test = dataset['test']
        self.cutoff = dataset['cutoff']
        self.paddingID = self.word2id['<pad>']
        self.sosID = self.word2id['<sos>']
        self.eosID = self.word2id['<eos>']
        self.unkID = self.word2id['<unk>']

        # convert dataset into tensors
        self.embeddingWeights = torch.tensor(weightMatrix, dtype=torch.float)
        self.vocabularySize = self.embeddingWeights.shape[0]
        self.embeddingDim = self.embeddingWeights.shape[1]
     
        # batching data
        self.traindata = prepDataset(train, self.paddingID, self.cutoff, train=True, step=self.args.reduction)
        self.valdata   = prepDataset(val,   self.paddingID, self.cutoff, train=True, step=self.args.reduction)
        self.testdata  = prepDataset(test,  self.paddingID, self.cutoff, train=False, step=self.args.reduction)
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

def loadDefaultArgs():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
            
    parser = argparse.ArgumentParser()

    default_name = defaultModelName()
    parser.add_argument('--name','-n', type=str, default=default_name)
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--model', type=str, default='vad')
    parser.add_argument('--hidden_size', "-hs", type=int, default=512)
    parser.add_argument('--latent_size', "-ls", type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs','-ep',  type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--gradient_clip', '-cl', type=int, default=1)
    parser.add_argument('--useBOW','-bow', type=str2bool, default=True)
    parser.add_argument('--reduction', '-r', type=int, default=16)
    parser.add_argument('--device', '-d', type=str, default="cuda")
    parser.add_argument('--use_latent', '-ul', type=str2bool, default=True)
    parser.add_argument('--teacher_training_p', '-tp', type=float, default=1.0)
    parser.add_argument('--pretrained_weights', '-w', type=str2bool, default=True)
    parser.add_argument('--save', '-s', type=str2bool, default=True)
    parser.add_argument('--penn_path', type=str, default= '../Datasets/Penn')
    parser.add_argument('--amazon_path', type=str, default= '../Datasets/Reviews')
    parser.add_argument('--subtitles_path', type=str, default= '../Datasets/OpenSubtitles')
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--test', type=str2bool, default=False)
    # kl parameters
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)
    # svae parameters
    parser.add_argument('-svae_word_dropout_rate', type=float, default=0.0)
    parser.add_argument('-hyp_opt', type=str2bool, default=False)
    parser.add_argument('-hyp_opt_loss_thresh', type=float, default=1.0)
    # beats annoying error when called in jupyter
    parser.add_argument('-f', default="?")

    args = parser.parse_args()

    args.default_name = default_name

    assert args.model   in ['vad', 'seq2seq', 'bowman']
    assert args.dataset in ['amazon', 'penn', 'subtitles']
    # make sure that if we're loading a model, then the default_name can't be used (as it wouldn't exist)
    if args.load_model:
        assert args.name != default_name

    return args

if __name__ == "__main__":
    args = loadDefaultArgs()
    brock = Trainer(args)
    if args.load_model:
        brock.loadModel()
    else:
        brock.setup()
        brock.train()
        if brock.args.save:
            saveModel(brock.model, brock.folder_path)
            brock.runStats()
    if args.test:
        brock.test()