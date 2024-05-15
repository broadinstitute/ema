import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
from scripts.SpaceP import EMBEDDING_HANDLER
from ema.ema import EmbeddingHandler

SPLIT_ID = 1
AGG_ID = 1
MODEL_NAME = "esm2_t30_150M_UR50D"
FP_AGG_EMB_0 = (
    f"data/aggregated-embeddings/{MODEL_NAME}/split-0/agg-{AGG_ID}.npy"
)
FP_AGG_EMB_1 = f"data/aggregated-embeddings/{MODEL_NAME}/split-{SPLIT_ID}/agg-{AGG_ID}.npy"
FP_AGG_VARIANT_META_0 = (
    f"data/aggregated-embeddings/{MODEL_NAME}/split-0/agg-{AGG_ID}-meta.csv"
)
FP_AGG_VARIANT_META_1 = f"data/aggregated-embeddings/{MODEL_NAME}/split-{SPLIT_ID}/agg-{AGG_ID}-meta.csv"
# FP_GENE_TABLE = "data/genes.csv"


def main():

    variant_names_0 = pd.read_csv(FP_AGG_VARIANT_META_0).iloc[:, 0].tolist()
    variant_names_1 = pd.read_csv(FP_AGG_VARIANT_META_1).iloc[:, 0].tolist()
    assert variant_names_0 == variant_names_1
    variant_names = variant_names_0.copy()
    del variant_names_0, variant_names_1

    emb_0 = np.load(FP_AGG_EMB_0)
    emb_1 = np.load(FP_AGG_EMB_1)
    assert emb_0.shape == emb_1.shape
    assert emb_0.shape[0] == len(variant_names)
    df_meta_data = pd.read_csv(FP_AGG_VARIANT_META_0)

    emb = EmbeddingHandler(sample_meta_data=df_meta_data)
    emb.add_emb_space(embeddings=emb_0, emb_space_name="full_length")
    emb.add_emb_space(embeddings=emb_1, emb_space_name="chopped")

    # visualise embedding values
    fig = emb.plot_emb_hist()
    fig = emb.plot_emb_box(group="sample")
    fig = emb.plot_emb_box(group="gene")

    fig = emb.visualise_emb_tsne(emb_space_name="full_length", colour="family")
    fig = emb.visualise_emb_tsne(emb_space_name="chopped", colour="family")

    # visualise embedding distances
    fig = emb.plot_emb_dist_heatmap(
        emb_space_name="full_length",
        group="family",
        distance_metric="cosine",
    )
    fig = emb.plot_emb_dist_heatmap(
        emb_space_name="chopped",
        group="family",
        distance_metric="cosine",
    )

    fig = emb.plot_emb_dist_box(group="gene", distance_metric="cosine")

    # explore difference between similarity distances between two embedding spaces
    fig = emb.plot_emb_dis_dif_heatmap(
        emb_space_name_1="full_length",
        emb_space_name_2="chopped",
        distance_metric="cosine",
    )

    emb.plot_emb_dis_dif_box(
        emb_space_name_1="full_length",
        emb_space_name_2="chopped",
        distance_metric="cosine",
        group="family",
    )

    print()


if __name__ == "__main__":
    main()
