## Installation

This requires `Python 3.9.1` and it is recommended to install using a virtual environment. I have personally used `pyenv` to simplify things, but it's not necessary.

	python -m venv env
	source env/bin/activate
	pip install -U pip wheel
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	
Scripts to download the datasets and assets imply that you're using some form of Unix, otherwise you can read the files and grab the URLs.
		
## Fetch Data

Here we'll be downloading the OpenSubtitles Dataset, and the Amazon Dataset. This has been neatly simplified into a single script which can be run using `./get_datasets.sh`. 

Downloading relevant assets for the models are bundled separately through `./get_assets.sh` (such as the embeddings and so on).

## Data Preprocessing

It's all bundled into the following:

	python preprocess_data.py -D open_subtitles
	python preprocess_data.py -D amazon

## Train Models

Train Bowman's VAE, CVAE, and the VAD.

## Compare Results



