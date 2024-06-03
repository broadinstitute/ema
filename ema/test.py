import numpy as np
import pandas as pd

from ema import EmbeddingHandler

DATA_DIR = "examples/HCN1-variants/"
FP_METADATA = DATA_DIR + "metadata.csv"
FP_EMB_ESM1b = DATA_DIR + "esm1b_t33_650M_UR50S-embeddings.npy"
FP_EMB_ESM2 = DATA_DIR + "esm2_t33_650M_UR50D-embeddings.npy"

# load metadata and embeddings

metadata = pd.read_csv(FP_METADATA)
emb_esm1b = np.load(FP_EMB_ESM1b)
emb_esm2 = np.load(FP_EMB_ESM2)

# initialize embedding handler
emb_handler = EmbeddingHandler(metadata)

# add embeddings to the handler
emb_handler.add_emb_space(embeddings=emb_esm1b, emb_space_name="esm1b")
emb_handler.add_emb_space(embeddings=emb_esm2, emb_space_name="esm2")

fig = emb_handler.plot_emb_dis_scatter(
    emb_space_name_1="esm1b",
    emb_space_name_2="esm2",
    distance_metric="euclidean",
    colour_group="binary_disorder_prediction",
    colour_value_1="False",
)

print()
