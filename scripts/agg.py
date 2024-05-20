import json
import numpy as np
import os
import torch
import pathlib
import pandas as pd

AGG_ID = "1"

SPLIT_ID = "0"
MODEL_NAME = "esm2_t33_650M_UR50D"  # "esm2_t30_150M_UR50D"

FP_AGG_PARAMS = "configs/agg_ids.json"
FP_GENE_TABLE = "data/genes.csv"

DIR_EMB = f"data/embeddings/{MODEL_NAME}/split-{SPLIT_ID}"
FP_OUT_AGG_EMB = f"data/aggregated-embeddings/{MODEL_NAME}/split-{SPLIT_ID}/agg-{AGG_ID}.npy"

esm_representation_layer_per_model = {
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
    "esm1v_t33_650M_UR90S_1": 33,
}


def main():

    # import AGG_ID params
    with open(FP_AGG_PARAMS, "r") as f:
        agg_params = json.load(f)

    for key, value in agg_params[AGG_ID].items():
        globals()[key] = value

    # find all files in the DIR_EMB
    files = os.listdir(DIR_EMB)
    variant_embs = [file.split(".pt")[0] for file in files]
    variants = list(
        set([variant_emb.split("-")[0] for variant_emb in variant_embs])
    )

    # sort variants
    variants.sort()

    for variant in variants:

        # find all versions
        embedding_files = [
            embedding_file
            for embedding_file in files
            if embedding_file.split("-")[0] == variant
        ]

        for i in range(0, len(embedding_files)):

            embedding_result = torch.load(f"{DIR_EMB}/{variant}-{i}.pt")
            embedding = embedding_result["mean_representations"][
                esm_representation_layer_per_model[MODEL_NAME]
            ].numpy()

            if variants.index(variant) == 0:
                numpy_embeddings = np.empty(
                    (len(variants), embedding.shape[0])
                )

            if i == 0:
                variant_numpy_embeddings = np.empty(
                    (len(embedding_files), embedding.shape[0])
                )

            variant_numpy_embeddings[i] = embedding

        # aggregate embedding
        if aggregation_method == "mean":
            numpy_embeddings[variants.index(variant)] = np.mean(
                variant_numpy_embeddings, axis=0
            )
        else:
            raise ValueError(
                f"Aggregation method {aggregation_method} not implemented."
            )

    # write embeddings into numpy file
    pathlib.Path(FP_OUT_AGG_EMB).parents[0].mkdir(parents=True, exist_ok=True)
    np.save(FP_OUT_AGG_EMB, numpy_embeddings)

    # get meta data for variants

    gene_table = pd.read_csv(FP_GENE_TABLE)

    # generate dataframe for gene families
    variant_genes = []
    variant_families = []
    for variant in variants:
        if ":" in variant:
            gene_name = variant.split(":")[0]
            variant_genes.append(gene_name + "_variant")
            variant_families.append(
                gene_table[gene_table["gene_name"] == gene_name][
                    "family"
                ].values[0]
                + "_variant"
            )
        else:
            variant_genes.append(variant)
            variant_families.append(
                gene_table[gene_table["gene_name"] == variant][
                    "family"
                ].values[0]
            )

    df_meta_data = pd.DataFrame(
        {
            "variant": variants,
            "gene": variant_genes,
            "family": variant_families,
        }
    )

    df_meta_data.to_csv(
        str(pathlib.Path(FP_OUT_AGG_EMB).parents[0])
        + f"/agg-{AGG_ID}-meta.csv",
        index=False,
    )

    print()


if __name__ == "__main__":
    main()
