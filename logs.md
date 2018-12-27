# Logs

These logs are used to indicate progress of the project as I work on it through the year. It also serves as a scratchpad for things that may be useful/relevant for the ISO.

## To-Do

- Blitz through [this 60 minute tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) on PyTorch.
- Create notes on Goldberg's NLP Paper
- Summary of Baselines:
    - Concise summary of seq2seq
    - Concise summary of Transformer paper
- Cover Controllable Story Generation paper
- Buy storage so you can fit the amazon dataset
- Browse around for variational autoencoder implementations?
- Model the implemented `seq2seq` code around the ISO problem so we can use it as a benchmark.

----

### 2018/12/27 (Thursday)

- Initial writeup for the theorical aspects of the paper. Lots of more reading is needed.
- The difference between VAEs and CVAEs does not look huge. I'll want to look into this in more detail.
- I'll need to make diagrams for regular autoencoders, LSTMs, RNNs, GRUs, Seq2seq, CVAEs, and VADs.

### 2018/12/26 (Wednesday)

- Retrieved large dataset from Julain McAuley (julian.mcauley@gmail.com).
- Talked to Julia about the most appropiate method for structuring the theoretical part of the ISO writeup.

### 2018/12/21 (Friday)

- Reviewed [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU) video.
- Used a guide to implement `seq2seq` in PyTorch that translates english to french, which can be found [here](Samples/seq2seq.ipynb). This model can be adapted to our problem, which we can use as our baseline. It will also help show the problems with this particular model as described in the variational autoregressive decoders paper.
- *Note to self:* I'll want to have a further look at different PyTorch tutorials. It's a lot more hands on than Keras (you define your own training mechanism, which I guess makes it more appropiate for this kind of task).
- The keras implementation for autoencoders becomes less useful now.
- I kind of understand GRUs but I'd want to look into why it's more efficent than LSTMS. In Quoc's seminar he used a LSTM but GRUs were introduced a few months before the `seq2seq` paper (according to the ArXiv dates).

#### Reading Material

- This pretty good article on [Variational Autoencoders](http://kvfrans.com/variational-autoencoders-explained/).
- This [implementation of a Variational Autoencoder](https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/) in PyTorch. (The code is most likely outdated considering it was published over a year ago but the structure should be similar.)
- The authors of the Variational Autoregressive decoders have a [recorded seminar on the paper](https://vimeo.com/305926196) I'm looking into.
- There is [some code on the GAN approach](https://github.com/ZhenYangIACAS/NMT_GAN) I'm looking into.

### 2018/12/20 (Thursday)

- Initial completion of Goldberg's NLP paper walkthrough. There is some content that is ignored based on the premise of the ISO (e.g Recursive Neural Networks), and the more relevant contents should be covered in more detail.
- Brief cover of implementing Autoencoders in Keras. However this is on a different problem (working on the MNIST dataset) but it provided a good idea on how to whip up autoencoders using machine learning libraries.
- Covered the Seq2Seq primer on [youtube](https://www.youtube.com/watch?v=oF0Rboc4IJw). This did not provide sufficent information (it was too shallow and was mostly preliminary knowledge).
- Covered Quoc Le's Seminar on `seq2seq`. It's a lot more thorough and provided a lot of relevant detail. Notes are [here](Notes/seq2seq.md).

### 2018/12/19 (Wednesday)

- Initial readthrough of Goldberg's NLP Primer to refresh understanding for task at hand. Currently at p46; Recurrent Neural Networks.
- Reviewed [Embedding Layers](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12).
- Reviewed [Word Embeddings](https://www.youtube.com/watch?v=5PL0TmQhItY) video.
- Reviewed the dataset, relatively fast storage will be needed to store the relevant dataset ([Amazon product reviews on Electronics](http://jmcauley.ucsd.edu/data/amazon/)) as it is quite large; I will also need to contact Julain McAuley (julian.mcauley@gmail.com) to obtain a link for the whole review data as you can only retrieve a small sample on the website.

#### Reading Material
- Undergraduates at Imperial College wrote a brief but strong primer on autoencoders and NLP [here](https://www.doc.ic.ac.uk/~js4416/163/website/nlp/).
- An okay primer of `seq2seq` in the form of a [youtube video](https://www.youtube.com/watch?v=oF0Rboc4IJw).
- A tutorial on [Implementing Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html).
- Quoc Le's [seminar on seq2seq](https://www.youtube.com/watch?v=G5RY_SUJih4).
- Google's [seq2seq documentation](https://google.github.io/seq2seq/).
- [Youtube summary of the Transformer paper](https://www.youtube.com/watch?v=iDulhoQ2pro) (Attention Is All You Need).
- Andrew Ng's Video on calculating [Derivatives with Computation Graphs](https://www.youtube.com/watch?v=nJyUyKN-XBQ).


### 2018/12/18 (Tuesday)

- Changed schedule meeting to 2019/01/02 due to unforseen changes to University opening times.
- Expectations of pre-assignment have changed; they are now to present the following for the meeting:
    - To have a plan of the report ready to show
    - To start planning and writing the theoretical aspects of the report (around 10 pages for this section)
    - To provide some code indicating the understanding of the problem at hand.

### 2018/12/14-17

- Well deserved break.

### 2018/12/13 (Thursday)

- Reviewed [Variational Autoenecoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) video.