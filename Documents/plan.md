<!-- Assessment is via a 20-30 page report plus viva.

– Dissertation (20-30 pages)
- Oral presentation followed by questions 
– Submitted at beginning of Summer term

In the second case, I suggest you to already have a plan of the report ready and start writing the theoretical part of your report (all in all about 10 pages for this part). Again feel free to send me any intermediate versions (google doc or overleaf if you prefer Latex). 

So I would suggest go straight to the point: starting with describing the problem first, positioning it in the range of the related problems (as you mentioned yourself), introducing in details just the Seq2seq solution with some necessary background (as your baseline), mention the limitations of Seq2seq for your problem and ways they could be addressed (VAEs, etc.), describe the VAE approach and your motivation to pick it, introduce the VAE paper you work with in details.

Depending on the resulting volume you can vary the neural network primer part in Seq2seq, or you maybe could do a separate mini section somewhere (we can see later).

I am not sure if you work in Latex, if you do I would prefer Overleaf. 

Keep me posted and good luck,

Julia
-->


- Introduction
	- Discussing about text generation
	- Text generation is one of 
- Literature Survey
	- Theory
		- Neural Networks
			- Neural Networks are non-linear statistical models that generate complex relationships between input and output vectors. Note that the input and output vectors are of a fixed dimension, which becomes a problem for our task at hand.
			- Backpropagation
		- Autoencoders
			Autoencoders are a specialised form of MLPs where the model attempts to recreate the inputs on the output. Autoencoders typically have a neural network layer in the model where its dimension is smaller than the input space, therefore representing a type of compression in the data. Autoencoders are composed of two different networks, an encoder and a decoder. The two networks are trained together in a manner that allows them to preserve the input as much as possible. Autoencoders are popularised through their use in Machine Translation, Word Embeddings, and document clustering.
		- Recurrent Neural Networks
			- Recurrent neural networks (CNNs) are a particular class of neural networks such that the outputs are not necessarily restricted and discrete (as opposed to the MLP). CNNs essentially operate over a sequence of vectors, making them popular in contempoary NLP problems. Given a sequence of inputs (x_1, ..., x_T), a standard RNN computes a sequence of outputs (y_1, ..., y_T) by iterating the following equation:
			

	- Modern Work
		- seq2seq
			- Seq2Seq is a modern interpretation of the encoder model, by providing an attention mechanism. The attention mechanism looks at all of the inputs from the hidden states of the encoders so far. 
		Seq2Seq describes 
		- Problems
			- Transformers
			- Variational Autoencoders
			- Variational Autoregressive Decoders
- Approach
	