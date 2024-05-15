import os
import pathlib
import torch

from esm import FastaBatchedDataset, pretrained

# code gratefully borrowed from
# https://www.kaggle.com/code/viktorfairuschin/extracting-esm-2-embeddings-from-fasta-files

SPLIT_ID = "1"
MODEL_NAME = "esm2_t30_150M_UR50D"
FP_SEQUENCES = f"data/splits/split-{SPLIT_ID}.fasta"
FP_OUT_EMBEDDINGS = f"data/embeddings/{MODEL_NAME}/split-{SPLIT_ID}/"

esm_representation_layer_per_model = {
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
}


def extract_embeddings(
    model_name,
    fasta_file,
    output_dir,
    repr_layers,
    tokens_per_batch=4096,
    seq_length=1022,
):

    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f"Processing batch {batch_idx + 1} of {len(batches)}")

            existing_files_counter = 0
            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                filename = output_dir / f"{entry_id}.pt"

                if os.path.exists(filename):
                    existing_files_counter += 1

            if existing_files_counter == len(labels):
                print("All files already exist. Skipping batch.")
                continue

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu")
                for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                entry_id = label.split()[0]

                filename = output_dir / f"{entry_id}.pt"

                truncate_len = min(seq_length, len(strs[i]))

                result = {"entry_id": entry_id}
                result["mean_representations"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }

                torch.save(result, filename)


def main():

    repr_layers = [esm_representation_layer_per_model[MODEL_NAME]]

    fasta_file = pathlib.Path(FP_SEQUENCES)
    output_dir = pathlib.Path(FP_OUT_EMBEDDINGS)

    extract_embeddings(
        model_name=MODEL_NAME,
        fasta_file=fasta_file,
        output_dir=output_dir,
        repr_layers=repr_layers,
    )


if __name__ == "__main__":
    main()
