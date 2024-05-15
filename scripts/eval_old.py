import numpy as np
import plotly.express as px
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from scripts.SpaceP import EMBEDDING_HANDLER

SPLIT_ID = 1
AGG_ID = 1
MODEL_NAME = "esm2_t30_150M_UR50D"
FP_AGG_EMB_0 = (
    f"data/aggregated-embeddings/{MODEL_NAME}/split-0/agg-{AGG_ID}.npy"
)
FP_AGG_EMB = f"data/aggregated-embeddings/{MODEL_NAME}/split-{SPLIT_ID}/agg-{AGG_ID}.npy"
FP_AGG_VARIANT_NAME_0 = f"data/aggregated-embeddings/{MODEL_NAME}/split-0/agg-{AGG_ID}-variant-names.csv"
FP_AGG_VARIANT_NAME = f"data/aggregated-embeddings/{MODEL_NAME}/split-{SPLIT_ID}/agg-{AGG_ID}-variant-names.csv"
FP_GENE_TABLE = "data/genes.csv"


def visualise_embeddings(
    embedding_matrix, variant_names, variant_color, color_map=None
):
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    X_2d = tsne.fit_transform(embedding_matrix)

    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=variant_color,
        labels={"color": "Cluster"},
        title=f"t-SNE visualization of variant embeddings of {SPLIT_ID}",
        size=[8] * len(variant_names),
        hover_data={
            "Variant": variant_names,
        },
        color_discrete_map=color_map,
    )
    fig.show()
    return


def get_edm(X):
    """Calculate Euclidean Distance Matrix.

    Args:
        X (np.array): Matrix with samples
            for which the EDM should be calculated.

    Returns:
        np.array: Euclidean Distance Matrix.
    """
    p1 = np.sum(X**2, axis=1)[:, np.newaxis]
    p2 = np.sum(X**2, axis=1)
    p3 = -2 * np.dot(X, X.T)
    edm = np.round(np.sqrt(p1 + p2 + p3), 2)
    np.nan_to_num(edm, copy=False)  # replace NaN with 0
    return edm


def normalise_by_median(X):
    """Normalise matrix by median per row.

    Args:
        X (np.array): Matrix to be normalised.

    Returns:
        np.array: Normalised matrix.
    """
    X = X / np.median(X, axis=1, keepdims=True)  # [:, np.newaxis]
    return X


def normalise_by_sum(X):
    """Normalise matrix by sum per row.

    Args:
        X (np.array): Matrix to be normalised.

    Returns:
        np.array: Normalised matrix.
    """
    X = X / np.sum(X, axis=1, keepdims=True)  # [:, np.newaxis]
    return X


emb = np.load(FP_AGG_EMB)
emb_0 = np.load(FP_AGG_EMB_0)
variant_names = list(pd.read_csv(FP_AGG_VARIANT_NAME, header=None)[0])
gene_table = pd.read_csv(FP_GENE_TABLE)

# check that FP_AGG_VARIANT_NAME_0 is the same as FP_AGG_VARIANT_NAME
variant_names_0 = list(pd.read_csv(FP_AGG_VARIANT_NAME_0, header=None)[0])
assert variant_names == variant_names_0

variant_genes = []
variant_families = []
for variant in variant_names:
    if ":" in variant:
        gene_name = variant.split(":")[0]
        variant_genes.append(gene_name + "_variant")
        gene_family = (
            gene_table[gene_table["gene_name"] == gene_name]["family"].values[
                0
            ]
            + "_variant"
        )
        variant_families.append(gene_family)
    else:
        variant_genes.append(variant)
        gene_family = gene_table[gene_table["gene_name"] == variant][
            "family"
        ].values[0]
        variant_families.append(gene_family)

# generate colour mappping for gene families
gene_families = list(set(variant_families))
# sort
gene_families = sorted(gene_families)
gene_family_colors = px.colors.qualitative.Set1
gene_family_color_map = {
    gene_family: gene_family_colors[i]
    for i, gene_family in enumerate(gene_families)
}

visualise_embeddings(
    emb, variant_families, variant_families, gene_family_color_map
)
visualise_embeddings(
    emb_0, variant_families, variant_families, gene_family_color_map
)

# calculate Euclidean distance matrix
edm = get_edm(emb)
edm_0 = get_edm(emb_0)

emb_handler = EMBEDDING_HANDLER(
    sample_list=variant_names,
    emb1_fp=FP_AGG_EMB_0,
    emb2_fp=FP_AGG_EMB,
    emb1_name="full_length",
    emb2_name="chopped",
)

# visualise embedding distances within the families
fig = px.imshow(
    edm,
    title=f"Euclidean Distance Matrix of {SPLIT_ID}",
    labels=dict(x="Variant", y="Variant"),
    x=variant_names,
    y=variant_names,
    color_continuous_scale="Blues",
)
fig.show()

df_distances_between_families = pd.DataFrame(
    columns=["family", "variant_id", "distance", "other_family"]
)

for family in gene_families:
    family_indices = [
        i
        for i, family_name in enumerate(variant_families)
        if family_name == family
    ]
    non_family_indices = [
        i
        for i, family_name in enumerate(variant_families)
        if family_name != family
    ]
    edm_family = np.median(edm[family_indices][:, family_indices], axis=1)
    edm_other_families = np.median(
        edm[family_indices][:, non_family_indices], axis=1
    )
    df_distances_between_families = pd.concat(
        [
            df_distances_between_families,
            pd.DataFrame(
                {
                    "family": [family] * len(family_indices),
                    "variant_id": [variant_names[i] for i in family_indices],
                    "distance": edm_family,
                    "family": "same family",
                }
            ),
        ]
    )
    df_distances_between_families = pd.concat(
        [
            df_distances_between_families,
            pd.DataFrame(
                {
                    "family": [family] * len(family_indices),
                    "variant_id": [variant_names[i] for i in family_indices],
                    "distance": edm_other_families,
                    "family": "other family",
                }
            ),
        ]
    )

# visualise mean Euclidean distance between variants within families
# and to other families in a bar plot
fig = px.bar(
    df_distances_between_families,
    x="variant_id",
    y="distance",
    barmode="group",
    color="family",
    labels={"distance": "Mean Euclidean Distance"},
    title="Mean Euclidean Distance within Families",
)
fig.update_layout(
    template="plotly_white",
    font=dict(family="Arial", size=18),
)
fig.show()

# normalise EDMs by median
edm_norm = normalise_by_sum(edm)
edm_0_norm = normalise_by_sum(edm_0)

# visualise embedding distances
fig = px.imshow(
    edm_norm,
    title=f"Normalised Euclidean Distance Matrix of {SPLIT_ID}",
    labels=dict(x="Variant", y="Variant"),
    x=variant_names,
    y=variant_names,
    color_continuous_scale="Blues",
)
fig.show()

# calculate absolute difference between normalised EDMs
edm_abs_dif = np.abs(edm_norm - edm_0_norm)

# visualise emb_abs_diff diagnonal matrix values
# variant_names on x-axis and y-axis
fig = px.imshow(
    edm_abs_dif,
    title=f"Absolute difference between embeddings of {SPLIT_ID} and 0",
    labels=dict(x="Variant", y="Variant"),
    x=variant_names,
    y=variant_names,
    color_continuous_scale="Blues",
)
fig.show()

# generate numpy array with 20 samples of each 4 features
# from a random distribution with mean 0 and standard deviation 1
test_emb = np.random.randn(40).reshape(20, 2)

# rotate all 4 features by 45 degrees
rotation_matrix = np.array(
    [
        [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
        [np.sin(np.pi / 4), np.cos(np.pi / 4)],
    ]
)
test_emb_0 = np.dot(test_emb, rotation_matrix)

# scale all features by 2
test_emb_0 = test_emb_0 * 2

# add some noise to the second and fourth sample
test_emb_0[1] = test_emb_0[1] + 3
test_emb_0[3] = test_emb_0[3] - 1

visualise_embeddings(
    test_emb, variant_names[:20], variant_families[:20], gene_family_color_map
)
visualise_embeddings(
    test_emb_0,
    variant_names[:20],
    variant_families[:20],
    gene_family_color_map,
)

# visualise embeddings in 2d scatter plot
fig = px.scatter(
    x=test_emb[:, 0],
    y=test_emb[:, 1],
    color=variant_families[:20],
    labels={"color": "Cluster"},
    title="t-SNE visualization of variant embeddings",
    size=[8] * 20,
    hover_data={
        "Variant": variant_names[:20],
    },
    color_discrete_map=gene_family_color_map,
)
fig.show()

fig = px.scatter(
    x=test_emb_0[:, 0],
    y=test_emb_0[:, 1],
    color=variant_families[:20],
    labels={"color": "Cluster"},
    title="t-SNE visualization of variant embeddings",
    size=[8] * 20,
    hover_data={
        "Variant": variant_names[:20],
    },
    color_discrete_map=gene_family_color_map,
)
fig.show()

# calculate Euclidean distance matrix
edm = get_edm(test_emb)
edm_0 = get_edm(test_emb_0)

# normalise EDMs by median
edm_norm = normalise_by_sum(edm)
edm_0_norm = normalise_by_sum(edm_0)

# calculate absolute difference between normalised EDMs
edm_abs_dif = np.abs(edm_norm - edm_0_norm)

# visualise emb_abs_diff diagnonal matrix values
# variant_names on x-axis and y-axis
fig = px.imshow(
    edm_abs_dif,
    title=f"Absolute difference between embeddings of {SPLIT_ID} and 0",
    labels=dict(x="Variant", y="Variant"),
    x=variant_names[:20],
    y=variant_names[:20],
    color_continuous_scale="Blues",
)
fig.show()

print()
