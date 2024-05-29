import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
from ema.ema import EmbeddingHandler

FP_META_DATA = "data/potassium-channels/meta_data.csv"
FP_EMB_ESM1v = "data/potassium-channels/esm1v_t33_650M_UR90S_1-embeddings.npy"
FP_EMB_ESM2 = "data/potassium-channels/esm2_t30_150M_UR50D-embeddings.npy"


def main():

    emb_esm1v = np.load(FP_EMB_ESM1v)
    emb_esm2 = np.load(FP_EMB_ESM2)
    df_meta_data = pd.read_csv(FP_META_DATA)

    emb = EmbeddingHandler(sample_meta_data=df_meta_data)

    emb.add_emb_space(embeddings=emb_esm1v, emb_space_name="ESM1v")
    emb.add_emb_space(embeddings=emb_esm2, emb_space_name="ESM2")
    fig = emb.plot_emb_dist_scatter(
        emb_space_name_1="ESM1v",
        emb_space_name_2="ESM2",
        distance_metric="adjusted_cosine",
        colour_group="family",
        colour_value_1="Kir",
        colour_value_2="Kv",
    )

    fig = emb.plot_emb_dist_dif_percentiles(
        emb_space_name_1="ESM1v",
        emb_space_name_2="ESM2",
        distance_metric="adjusted_cosine",
        subset_group="family",
        subset_group_value="Kir",
    )

    fig = emb.plot_distances_per_rank(
        emb_space_name="ESM1v", distance_metric="cosine"
    )

    fig = emb.plot_emb_dist_dif_percentiles(
        emb_space_name_1="ESM1v",
        emb_space_name_2="ESM2",
        distance_metric="adjusted_cosine",
    )

    fig = emb.visualise_emb_pca("ESM1v")

    # An unsupervised clustering can be performed using the KMeans algorithm
    # The number of clusters can be determined by the user
    # The overlap between the clusters and the groups defined in the meta data can be visualised
    # in this bar plot

    fig = emb.plot_emb_hist()
    fig = emb.plot_emb_box(group="sample")
    fig = emb.plot_emb_box(group="gene")

    fig = emb.visualise_emb_tsne(emb_space_name="ESM1v", colour="family")
    fig = emb.visualise_emb_tsne(emb_space_name="ESM2", colour="family")

    # visualise embedding distances
    fig = emb.plot_emb_dist_dif_percentiles(
        emb_space_name_1="full_length",
        emb_space_name_2="chopped",
        distance_metric="cityblock",
        subset_group="family",
        subset_group_value="Kir",
    )

    fig = emb.plot_emb_dist_hist(
        emb_space_name_1="full_length",
        emb_space_name_2="chopped",
        distance_metric="euclidean",
    )

    fig = emb.plot_emb_dist_heatmap(
        emb_space_name="full_length",
        group="family",
        distance_metric="cosine",
        order_x="gene",
        order_y="gene",
    )

    fig = emb.plot_emb_dist_box(group="gene", distance_metric="cosine")

    # explore difference between similarity distances between two embedding spaces
    fig = emb.plot_emb_dis_dif_heatmap(
        emb_space_name_1="full_length",
        emb_space_name_2="chopped",
        distance_metric="cosine",
    )

    print()


if __name__ == "__main__":
    main()
