import numpy as np
import plotly.express as px
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

SPLIT_ID = 1
AGG_ID = 1
MODEL_NAME = "esm2_t30_150M_UR50D"
FP_AGG_EMB = f"data/aggregated-embeddings/{MODEL_NAME}/split-{SPLIT_ID}/agg-{AGG_ID}.npy"
FP_AGG_VARIANT_NAME = f"data/aggregated-embeddings/{MODEL_NAME}/split-{SPLIT_ID}/agg-{AGG_ID}-variant-names.csv"
FP_GENE_TABLE = "data/genes.csv"

emb = np.load(FP_AGG_EMB)
variant_names = list(pd.read_csv(FP_AGG_VARIANT_NAME, header=None)[0])
gene_table = pd.read_csv(FP_GENE_TABLE)

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

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
X_2d = tsne.fit_transform(emb)

fig = px.scatter(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    color=variant_families,
    labels={"color": "Cluster"},
    title=f"t-SNE visualization of variant embeddings of {SPLIT_ID}",
    size=[8] * len(variant_names),
    hover_data={
        "Variant": variant_names,
        "Gene": variant_genes,
    },
    color_discrete_map=gene_family_color_map,
)
fig.show()

print()
