# ema
A library for comparing protein embedding spaces for different models

## Colab Notebook Examples

| Description | Link |
|---------|-------------|
| HCN1 variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pia-francesca/ema/blob/main/colab_notebooks/ema_HCN1_variants_example.ipynb) |
| Ion Channels and Families | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pia-francesca/ema/blob/main/colab_notebooks/ema_ion_channel_proteins_example.ipynb) |


## Local installation

You can install the ema library through pip, or access examples locally by cloning the github repo.

### Installing the ema library
```
pip install ema-emb
```

### Cloning the ema repo
```
git clone https://github/pia-francesca/ema

cd ema                         # enter project directory
pip3 install .                 # install dependencies
jupyter lab colab_notebooks    # open notebook examples in jupyter for local exploration
```

## Usage
ema allows you to compare embedding spaces from different protein language models. Uses include:
- Unsupervised clustering in the embedded space
- Visualizing clusters with PCA, UMAP, and t-SNE plots of embedding spaces for each model
- Scatterplot distances between sequences, compared across embedding spaces (For example, seq 1 and seq 2 have distance x in embedding space 1 and distance y in embedding space 2)

## Default/example embedding spaces provided
- esm1b
- esm1v
- esm2
- ProstT5
