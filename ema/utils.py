import numpy as np
import plotly.express as px

from scipy.spatial.distance import squareform


def generate_unique_pairs(indices):
    pairs = set()
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pairs.add((indices[i], indices[j]))
    return list(pairs)


def generate_cross_list_pairs(indices_1, indices_2):
    pairs = set()
    for i in range(len(indices_1)):
        for j in range(len(indices_2)):
            pair = tuple(sorted([indices_1[i], indices_2[j]]))
            pairs.add(pair)
    return list(pairs)


def percentile_rank(row):
    """
    Replace each value in the row with its percentile rank.

    Parameters:
    row (numpy.ndarray): Input 1D array (row).

    Returns:
    numpy.ndarray: Row with values replaced by their percentile ranks.
    """
    sorted_indices = np.argsort(row)
    ranks = np.argsort(sorted_indices)
    percentiles = (ranks / (len(row) - 1)) * 100
    percentiles_rounded_up = np.ceil(percentiles / 10) * 10
    return percentiles_rounded_up


def row_percentiles(arr):
    """
    Replace each value in the array with its percentile rank relative to the row.

    Parameters:
    arr (numpy.ndarray): Input 2D array.

    Returns:
    numpy.ndarray: Array with the same shape where each value is replaced by its percentile rank.
    """

    # Apply the percentile_rank function to each row
    percentile_array = np.apply_along_axis(percentile_rank, 1, arr)
    return percentile_array


def global_percentiles(arr):
    """
    Replace each value in the array with its global percentile rank, rounded up to the next multiple of ten.

    Parameters:
    arr (numpy.ndarray): Input 2D array.

    Returns:
    numpy.ndarray: Array with the same shape where each value is replaced by its global percentile rank.
    """
    flattened = squareform(arr)
    sorted_indices = np.argsort(flattened)
    ranks = np.argsort(sorted_indices)
    percentiles = (ranks / (len(flattened) - 1)) * 100
    percentiles_rounded_up = np.ceil(percentiles / 10) * 10
    percentile_array = squareform(percentiles_rounded_up)
    return percentile_array


def get_scatter_plot(
    emb_object: dict,
    emb_space_name: str,
    colour: str,
    X_2d: np.array,
    method: str,
):
    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=emb_object.meta_data[colour],
        labels={"color": "Cluster"},
        title=f"{method} visualization of variant embeddings of {emb_space_name} embeddings",
        hover_data={
            "Sample": emb_object.sample_names,
        },
        opacity=0.5,
        color_discrete_map=emb_object.colour_map[colour],
    )

    # make dots proportional to number of samples
    fig.update_traces(
        marker=dict(size=(1 / len(emb_object.sample_names)) * 400)
    )
    fig = update_fig_layout(fig)
    return fig


def update_fig_layout(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="black"),
    )
    # show line at y=0 and x=0
    fig.update_xaxes(showline=True, linecolor="black", linewidth=2)
    fig.update_yaxes(showline=True, linecolor="black", linewidth=2)
    # hide gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def convert_to_1d_array(matrix):
    """
    Convert a 2D symmetric distance matrix to a 1D array of distances.

    Parameters:
    matrix (numpy.ndarray): 2D symmetric distance matrix.

    Returns:
    numpy.ndarray: 1D array of distances.
    """
    # Get the size of the matrix
    n = matrix.shape[0]
    m = matrix.shape[1]

    # Initialize an empty array to store distances
    distances = []

    # Extract upper triangular part of the matrix (excluding diagonal)
    for i in range(n):
        for j in range(i + 1, m):
            distances.append(matrix[i, j])

    return np.array(distances)
