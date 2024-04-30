def read_in_fasta(fp: str, index: str):
    """Read in fasta file.

    Parameters:
        fp (str): filepath to fasta file.
        index (str): index of sequences in dict. \
            Options: "isoform", or "gene".

    Returns:
        seq_dict (dict): dictionary of isoform:sequence pairs.        \
    """

    index_mapping = {"gene": 0, "isoform": 1}

    if index not in index_mapping:
        raise ValueError("Index " + index + " is not valid.")

    seq_dict = {}
    with open(fp, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(">"):
                seq_index = line.split("|")[index_mapping[index]].strip()
                seq_index = seq_index.replace(">", "")
                seq = lines[lines.index(line) + 1].strip()
                seq_dict[seq_index] = seq

    return seq_dict