class Statistics:
    def __init__(self, model_folder):
        self.model_folder = model_folder

    # load the csv containing the results
    def loadResults(model_folder, models_parent_folder="models", results_folder="results"):
        """
        Loads csvs of all epochs from a given model_folder name.
        Returns a concatenation of results.
        """
        # setup filepath
        folder = os.path.join(models_parent_folder, model_folder, results_folder)
        files = os.listdir(folder)
        
        # retrieve .csv files.
        if len(files) < 1: return
        parts = re.match(r"([a-z]+_)([0-9]+)([a-z]+.csv)", files[0], re.I).groups()
        files = [i for i in files if (i[-4:].lower() == ".csv")]
        # sort by epochs (increasing order)
        numbs = sorted([int(re.findall('\d+', i )[0]) for i in files])
        
        results = None
        for number in numbs:
            # create filename
            file = parts[0] + str(number) + parts[2]
            filepath = os.path.join(folder, file)
            
            ents = np.loadtxt(filepath, delimiter=", ")
            if results is None:
                results = ents
            else:
                results = np.vstack((results,ents))
        return results

    def plotChart(results, step=100, ylim=6):
        recon = results[:,0]
        x = np.arange((recon.shape[0]))
        best_fit = np.poly1d(np.polyfit(x, recon, 4))
        plt.plot(results[::step, :])
        plt.plot(best_fit(x)[::step], label="Reconstruction Loss Best Fit")
        plt.ylim(0,ylim)
        plt.legend()
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.show()

    # now we're looking at loading the dataset from each model.
    def loadDatasetFromModel(model_folder):
        # we want to load the model parameters
        filepath = os.path.join("models",model_folder, "model_parameters.json")
        with open(filepath, "r") as f:
            model_parameters = json.load(f)

        device = torch.device("cpu")
        batchSize = model_parameters['batchSize']
        reduction = model_parameters['reduction']

        # load dataset parameters so we know the location of the datasets.
        filepath = os.path.join("models",model_folder, "dataset_parameters.json")
        with open(filepath, "r") as f:
            dataset_parameters = json.load(f)

        # load dataset
        dataset_filepath = dataset_parameters['datasetFile']
        dataset = loadDataset(path=dataset_filepath)

        word2id, id2word = dataset['word2id'], dataset['id2word']
        paddingID, sosID = word2id['<pad>'], word2id['<sos>']
        cutoff = dataset['cutoff']

        # load validation data.
        val = dataset['validation']

        val_x, val_y = [x[0] for x in val[::reduction]], [x[1] for x in val[::reduction]]
        valx = batchData(val_x, paddingID, device, batchSize, cutoff)
        valy = batchData(val_y, paddingID, device, batchSize, cutoff)
        valx_tokens = [convertRealID2Word(id2word, i[0], toString=False) for i in valx]
        valy_tokens = [convertRealID2Word(id2word, i[0], toString=False) for i in valy]
        
        return (valx_tokens, valy_tokens)

    # now we're looking at loading the dataset from each model.
    def loadDatasetFromModel(model_folder):
        # we want to load the model parameters
        filepath = os.path.join("models",model_folder, "model_parameters.json")
        with open(filepath, "r") as f:
            model_parameters = json.load(f)

        device = torch.device("cpu")
        batchSize = model_parameters['batchSize']
        reduction = model_parameters['reduction']

        # load dataset parameters so we know the location of the datasets.
        filepath = os.path.join("models",model_folder, "dataset_parameters.json")
        with open(filepath, "r") as f:
            dataset_parameters = json.load(f)

        # load dataset
        dataset_filepath = dataset_parameters['datasetFile']
        dataset = loadDataset(path=dataset_filepath)

        word2id, id2word = dataset['word2id'], dataset['id2word']
        paddingID, sosID = word2id['<pad>'], word2id['<sos>']
        cutoff = dataset['cutoff']

        # load validation data.
        val = dataset['validation']

        val_x, val_y = [x[0] for x in val[::reduction]], [x[1] for x in val[::reduction]]
        valx = batchData(val_x, paddingID, device, batchSize, cutoff)
        valy = batchData(val_y, paddingID, device, batchSize, cutoff)
        valx_tokens = [convertRealID2Word(id2word, i[0], toString=False) for i in valx]
        valy_tokens = [convertRealID2Word(id2word, i[0], toString=False) for i in valy]
        
        return (valx_tokens, valy_tokens)

    # Rouge Calculation
    def calcRouge(act, pred, n=2):
        # create n-grams
        ngram_act  = [tuple(act[i:i+n]) for i in range(len(act)-(n-1))]
        ngram_pred = [tuple(pred[i:i+n]) for i in range(len(pred)-(n-1))]
        # create dictionary of reference n-grams to count against.
        ref = {}
        for gram in ngram_act:
            if gram not in ref:
                ref[gram] = 0
            ref[gram] += 1
        # compute rouge (recall)
        count = sum([1 if gram in ref else 0 for gram in ngram_pred])
        if count > 0:
            return count / len(ngram_act)
        return count

    # Mean averaged rouge with 1-3 grams
    def mean_rouge(act,pred, n_min=1, n_max=3):
        rouges = [rouge(act,pred,n) for n in range(n_min, n_max+1)]
        return sum(rouges)/len(rouges)

    # calculate bleu and rouge scores.
    def processPredictions(valx, valy, preds, epoch=-1):
        bleus = []
        rouges_1 = []
        rouges_2 = []
        rouges_3 = []
        for batch_num in range(len(valx)-1):
            for seq_num in range(len(valx[0])):
                condition = valx[batch_num][seq_num]
                actual    = valy[batch_num][seq_num]
                predicted = preds[epoch][batch_num][seq_num].split(" ")
                bleu = sentence_bleu([actual], predicted, weights=[1])
                rouge1 = calcRouge(actual, predicted, n=1)
                rouge2 = calcRouge(actual, predicted, n=2)
                rouge3 = calcRouge(actual, predicted, n=3)
                bleus.append(bleu)
                rouges_1.append(rouge1)
                rouges_2.append(rouge2)
                rouges_3.append(rouge3)

        return {
            'bleu' : np.mean(bleus),
            'rouge_1': np.mean(rouges_1),
            'rouge_2': np.mean(rouges_2),
            'rouge_3': np.mean(rouges_3)
        }

    def chartBLEUROUGE(self):
        bloos = [i['bleu'] for i in results]
        rouges_1 = [i['rouge_1'] for i in results]
        rouges_2 = [i['rouge_2'] for i in results]
        rouges_3 = [i['rouge_3'] for i in results]

        plt.plot(bloos, label="BLEU")
        plt.plot(rouges_1, label="ROUGE 1")
        plt.plot(rouges_2, label="ROUGE 2")
        plt.plot(rouges_3, label="ROUGE 3")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Percentage")
        plt.title("Emperical Measurements over Epochs")
        plt.show()