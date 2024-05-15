import json
import pandas as pd

from scripts.utils import read_in_fasta

FP_SPLIT_PARAMS = "configs/split_ids.json"
FP_SEQ = "data/sequences_nav_cav.fasta"
FP_LABELS = "data/variants.csv"

# Parameters
SPLIT_ID = "2"
FP_OUT_FASTA = f"data/splits/split-{SPLIT_ID}.fasta"


def split_seq_into_chunks_of_equal_size(
    length_chunk, length_overlap, sequence
):
    """Split a sequence into chunks with overlap.
    Splitting starts from both ends of the sequence and moves towards the center.

    Args:
        length_chunk (int): The length of the chunks
        length_overlap (int): The length of the overlap
        sequence (str): The sequence to be split

    Returns:
        list: A list of sequences
    """

    chunks = []

    # determine number of chunks
    n_chunks = int(len(sequence) / (length_chunk - length_overlap)) - 1

    # add first sequences
    for i in range(0, int(n_chunks / 2)):
        chunks.append(
            sequence[
                i
                * (length_chunk - length_overlap) : i
                * (length_chunk - length_overlap)
                + length_chunk
            ][::1]
        )

    # add middle sequence
    if n_chunks % 2 != 0:
        # add chunk around the middle of the sequence
        chunks.append(
            sequence[
                int(len(sequence) / 2)
                - int(length_chunk / 2) : int(len(sequence) / 2)
                - int(length_chunk / 2)
                + length_chunk
            ]
        )

    # add last sequences
    seq_inverted = sequence[::-1]
    end_chunks = []
    for i in range(0, int(n_chunks / 2)):
        end_chunks.append(
            seq_inverted[
                i
                * (length_chunk - length_overlap) : i
                * (length_chunk - length_overlap)
                + length_chunk
            ][::-1]
        )
    print(end_chunks)
    chunks = chunks + end_chunks[::-1]

    return chunks


def split_seq_by_n_chunks():
    pass


def main():

    # read in the parameters for the split
    with open(FP_SPLIT_PARAMS, "r") as f:
        split_params = json.load(f)

    for key, value in split_params[SPLIT_ID].items():
        globals()[key] = value

    dict_seq = read_in_fasta(FP_SEQ, index="gene")

    if SPLIT_ID == 0:
        # copy fasta file into output file without any further changes
        with open(FP_OUT_FASTA, "w") as f:
            for key, seq in dict_seq.items():
                f.write(f">{key}-0\n{seq}\n")
        return

    if (n_chunks != 0) and (chunk_size != 0):
        raise ValueError("Both n_chunks and chunk_size cannot be non-zero.")

    if n_chunks != 0:
        # TODO: split_seq_by_n_chunks()
        raise ValueError("Method not implemented yet.")

    if chunk_size != 0:
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        chopped_variants = []
        new_sequences = dict()
        for variant, variant_sequence in dict_seq.items():
            chunks = split_seq_into_chunks_of_equal_size(
                length_chunk=chunk_size,
                length_overlap=overlap,
                sequence=variant_sequence,
            )
            for i, chunk in enumerate(chunks):
                new_sequences[f"{variant}-{i}"] = chunk
            chopped_variants.append(variant)

        for variant in chopped_variants:
            del dict_seq[variant]

        dict_seq.update(new_sequences)

    else:
        raise ValueError("Neither chunk_size or n_chunks > 0.")

    with open(FP_OUT_FASTA, "w") as f:
        for key, seq in dict_seq.items():
            f.write(f">{key}\n{seq}\n")


if __name__ == "__main__":
    main()
