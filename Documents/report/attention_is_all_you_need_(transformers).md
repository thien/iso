

attention is a mechanism 

- go back and look at at particular inputs and see what is relevant
- attend to hidden states of the input sentence
- the decoder state can look directly at a particular encoder transitions, directly to the input vector.
- the way the decoder decides how to look at which state to look at:
	- in each step
	- it'll output a bunch of keys k1,...,kn
	- these keys will index the hidden states via a softmax architecture
	- (you can avoid going through the previous states entirely; so therefore..)
- you can ditch RNNs entirely.

- @ each step, the model is autoregressive (consuming the previously genearted symbols as additional input when generating the next).

## encoder:
- it's a stack of identical layers, each of which composes of 2 sublayers: a multi-head self-attention mechanism and the otherwise is a positionwise MLP. 
- There's a residual connection around the sublayers, followed by a normalisation.

## decoder:
- also a stack of identical layers, each of which has 3 sublayers: the same as the encoder but the third sublayer performs multi-head attention over the output of the encoder stack.

## attention
- described well in the paper.

- uses positional encoding (which essentially represent binary values using a combination of sinusoidal waves).
- Feed inputs directly
- determines the most appropiate word to use based on t