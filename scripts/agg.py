import numpy as np
import os
import torch
import pathlib

AGG_ID = 1

SPLIT_ID = 1
MODEL_NAME = "esm2_t30_150M_UR50D"
DIR_EMB = f"data/embeddings/{MODEL_NAME}/split-{SPLIT_ID}"
FP_OUT_AGG_EMB = f"data/aggregated-embeddings/{MODEL_NAME}/split-{SPLIT_ID}/agg-{AGG_ID}.npy"

esm_representation_layer_per_model = {
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
}


def main():

    # find all files in the DIR_EMB
    files = os.listdir(DIR_EMB)
    variant_embs = [file.split(".pt")[0] for file in files]
    variants = list(
        set([variant_emb.split("-")[0] for variant_emb in variant_embs])
    )

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
        numpy_embeddings[variants.index(variant)] = np.mean(
            variant_numpy_embeddings, axis=0
        )

    # write embeddings into numpy file
    pathlib.Path(FP_OUT_AGG_EMB).parents[0].mkdir(parents=True, exist_ok=True)
    np.save(FP_OUT_AGG_EMB, numpy_embeddings)

    # save variant order in txt
    with open(
        str(pathlib.Path(FP_OUT_AGG_EMB).parents[0])
        + f"/agg-{AGG_ID}-variant-names.csv",
        "w",
    ) as f:
        for variant in variants:
            f.write(f"{variant}\n")

    print()


if __name__ == "__main__":
    main()
