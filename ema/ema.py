import pandas as pd
import plotly.express as px
import numpy as np
import itertools
import umap
import math
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from statistics import mean


emb_space_colours = ["#56638A", "#BAD9B5", "#D77A61", "#D77A61", "#40531B"]
distance_metric_aliases = {
    "euclidean": "Euclidean",
    "cityblock": "Manhattan",
    "cosine": "Cosine",
    "sqeuclidean": "Standardised Euclidean",
    "sequclidean_normalised": "Normalised Standardised Euclidean",
    "euclidean_normalised": "Normalised Euclidean",  # replace with "Normalised Euclidean"
    "cityblock_normalised": "Normalised Manhattan",
    "adjusted_cosine": "Adjusted Cosine",
    "knn": "K-Nearest Neighbours",
    "mahalanobis": "Mahalanobis",
}


class EmbeddingHandler:
    def __init__(self, sample_meta_data: pd.DataFrame):
        """Initialise EmbeddingHandler object.

        Parameters:
        sample_meta_data (pd.DataFrame): Meta data for samples.
            Should have sample names in the first column.
        """
        self.meta_data = sample_meta_data.astype(str)
        self.sample_names = self.meta_data.iloc[:, 0].tolist()
        self.colour_map = self.__get_colour_map_for_features__()
        self.emb = dict()
        print(f"{len(self.sample_names)} samples loaded.")
        print(f"Meta data columns: {self.meta_data.columns}")
        return

    def __get_colour_map_for_features__(self) -> dict:
        """Generate colour map for features in meta_data.

        Returns:
        dict: Colour map for features in meta_data.
        """
        if len(self.meta_data.columns) == 1:
            print("No meta data provided. Cannot generate colour map.")
            return None
        colour_map = dict()
        for column in self.meta_data.columns[1:]:
            column_values = self.meta_data[column].unique()
            if len(column_values) > 15:
                print(
                    f"Column {column} has more than 15 unique values. \
                        Skipping."
                )
                continue
            colour_map[column] = dict()
            colours = px.colors.qualitative.Set2 + px.colors.qualitative.Set3

            # Check if the colours list is empty
            if not colours:
                raise ValueError("The colours list is empty.")

            for i, value in enumerate(column_values):

                # Check if the index i is within the range of the colours list
                if i >= len(colours):
                    raise IndexError(
                        f"The index i is out of range for {column_values}"
                    )
                colour_map[column][value] = colours[i]
        return colour_map

    def __check_for_emb_space__(self, emb_space_name: str) -> None:
        """Check if emb_space_name is present in emb.

        Parameters:
        emb_space_name (str): Name of the embedding space.

        Raises:
        ValueError: If emb_space_name is not present in emb.
        """
        if emb_space_name not in self.emb.keys():
            raise ValueError  # Add error message
        else:
            return True

    def __check_col_in_meta_data__(self, col) -> None:
        """Check if col is present in meta_data.

        Parameters:
        col (str): Column name.

        Raises:
        ValueError: If col is not present in meta_data.
        """
        if col not in self.meta_data.columns:
            raise ValueError(f"Column {col} not found in meta data.")

    def __sample_indices_to_groups__(self, group: str) -> dict:
        """Convert sample indices to groups based on a column in meta_data.

        Parameters:
        group (str): Column name in meta_data.

        Returns:
        dict: Dictionary with group names as keys and sample indices as values.
        """
        self.__check_col_in_meta_data__(group)
        group_indices = dict()
        for group_name in self.meta_data[group].unique():
            group_indices[group_name] = self.meta_data[
                self.meta_data[group] == group_name
            ].index.tolist()
        return group_indices

    def __calculate_clusters__(self, emb_space_name: str, n_clusters: int):
        """Calculate clusters using KMeans clustering algorithm.
        If n_clusters is not provided, the number of clusters is calculated
        based on the number of unique values in the meta_data columns.
        The number of clusters is the average of the number of unique values
        in the columns of the meta_data, excluding the first column.
        If only one column is present, the number of clusters is set to 5.
        If the number of clusters is less than 2, it is set to 3.
        If the number of clusters is more than 15, it is set to 15.
        """
        self.__check_for_emb_space__(emb_space_name)

        if n_clusters is None:
            if len(self.meta_data.columns) > 1:
                columns = [
                    col
                    for col in self.meta_data.columns
                    if "cluster_" not in col
                ]
                n_clusters = math.floor(
                    mean(
                        [self.meta_data[col].nunique() for col in columns[1:]]
                    )
                )
                if n_clusters < 2:
                    n_clusters = 3
                if n_clusters > 15:
                    n_clusters = 15
            else:
                n_clusters = 5
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb[emb_space_name]["emb"])
        km.predict(self.emb[emb_space_name]["emb"])

        # add cluster labels to meta_data
        self.meta_data["cluster_" + emb_space_name] = km.labels_.astype(str)
        self.colour_map = self.__get_colour_map_for_features__()
        print(f"{n_clusters} clusters calculated for {emb_space_name}.")
        return

    def __calculate_k_closest_neighbours__(self, emb_space_name: str, k: int):
        self.__check_for_emb_space__(emb_space_name)
        emb_pwd = self.get_sample_distance(emb_space_name, "euclidean")
        k_closest_neighbours = dict()
        for i, sample in enumerate(self.sample_names):
            k_closest_neighbours[sample] = [
                self.sample_names[j]
                for j in np.argsort(emb_pwd[i, :])[1 : k + 1]
            ]
        # mask the k_closest_neighbours in the distance matrix with 0 and 1
        emb_pwd_masked = np.zeros(emb_pwd.shape)
        emb_pwd_masked[emb_pwd < np.sort(emb_pwd, axis=1)[:, k][:, None]] = 1
        return emb_pwd_masked

    def add_emb_space(self, embeddings: np.array, emb_space_name: str) -> None:
        """Add embedding space to emb.

        Parameters:
        embeddings (np.array): Embedding space. Embeddings need to be
        in the shape (n_samples, n_features). The order of samples should
        match the order of samples in the meta_data.
        emb_space_name (str): Name of the embedding space. Can be any string.
        """
        if emb_space_name in self.emb.keys():
            raise ValueError(
                f"Embedding space {emb_space_name} already exists."
            )
        if embeddings.shape[0] != len(self.sample_names):
            raise ValueError(
                f"Number of samples in embeddings ({embeddings.shape[0]}) \
                    does not match the number of samples in meta_data \
                    ({len(self.sample_names)})"
            )
        self.emb[emb_space_name] = dict()
        self.emb[emb_space_name]["emb"] = embeddings
        self.__calculate_clusters__(emb_space_name, n_clusters=None)
        if len(self.emb.keys()) > len(emb_space_colours):
            # choose random color
            self.emb[emb_space_name]["colour"] = px.colors.qualitative.Set3[
                len(self.emb.keys()) - len(emb_space_colours)
            ]
        else:
            self.emb[emb_space_name]["colour"] = emb_space_colours[
                len(self.emb.keys()) - 1
            ]
        return

    def remove_emb_space(self, emb_space_name: str) -> None:
        """Remove embedding space from emb.

        Parameters:
        emb_space_name (str): Name of the embedding space to be removed.
        """
        self.__check_for_emb_space__(emb_space_name)
        del self.emb[emb_space_name]
        return

    def get_emb(self, emb_space_name: str) -> np.array:
        """Return embedding space.

        Parameters:
        emb_space_name (str): Name of the embedding space.

        Returns:
        np.array: Embedding space.
        """
        self.__check_for_emb_space__(emb_space_name)
        return self.emb[emb_space_name]["emb"]

    def get_groups(self) -> list:
        """Return list of columns in meta_data.

        Returns:
            list: List of columns in meta_data.
        """
        return self.meta_data.columns.tolist()

    def get_value_per_group(self, group: str) -> list:
        """Return unique values in a column of meta_data.

        Args:
            group (str): Column name in meta_data.

        Returns:
            list: Unique values in the column ordered alphabetically.
        """
        self.__check_col_in_meta_data__(group)
        group_values = self.meta_data[group].unique().tolist()
        group_values = sorted(group_values)
        return group_values

    def __calculate_pwd__(self, emb_space_name: str, metric: str):

        if metric == "sqeuclidean_normalised":
            # divide each row by its norm
            emb = self.emb[emb_space_name]["emb"]
            emb_norm = np.linalg.norm(emb, axis=1)
            emb = emb / emb_norm[:, None]  # divide each row by its norm
            emb_pwd = squareform(pdist(emb, metric="sqeuclidean"))
            return emb_pwd

        elif metric == "euclidean_normalised":

            # divide each row of the emb by its norm
            emb = self.emb[emb_space_name]["emb"]
            emb_norm = np.linalg.norm(emb, axis=1)
            emb = emb / emb_norm[:, None]  # divide each row by its norm
            emb_pwd = squareform(pdist(emb, metric="euclidean"))
            return emb_pwd

        elif metric == "cityblock_normalised":
            emb_pwd = squareform(
                pdist(self.emb[emb_space_name]["emb"], metric="cityblock")
            )
            emb_pwd = emb_pwd / len(self.emb[emb_space_name]["emb"][1])
            return emb_pwd

        elif metric == "adjusted_cosine":
            # substract the mean of each column from each value
            emb = self.emb[emb_space_name]["emb"]
            emb = emb - np.median(emb, axis=0)  # emb.median(axis=0)
            emb_pwd = squareform(pdist(emb, metric="cosine"))
            return emb_pwd

        elif metric == "knn":
            k_neighbours = math.floor(len(self.sample_names) / 5)
            if k_neighbours < 1:
                k_neighbours = 3
            elif k_neighbours > 100:
                k_neighbours = 100
            print(
                f"Calculating {k_neighbours} nearest neighbours for each sample."
            )
            emb_pwd = self.__calculate_k_closest_neighbours__(
                emb_space_name=emb_space_name, k=k_neighbours
            )
            return emb_pwd

        emb_pwd = squareform(
            pdist(self.emb[emb_space_name]["emb"], metric=metric)
        )
        return emb_pwd

    def get_distance_percentiles(self, emb_space_name, distance_metric):
        self.__check_for_emb_space__(emb_space_name)
        emb_pwd = self.get_sample_distance(emb_space_name, distance_metric)
        percentiles = global_percentiles(emb_pwd)
        return percentiles

    def get_sample_distance(
        self, emb_space_name: str, metric: str, rank: bool = False
    ) -> np.array:
        """Calculate pairwise distance between samples in the embedding space.

        Parameters:
        emb_space_name (str): Name of the embedding space.
        metric (str): Distance metric. Can be one of the following:
            - "euclidean"
            - "cityblock"
            - "cosine"
            - "sequclidean"
            - "euclidean_normalised"
            - "cityblock_normalised"
            - "adjusted_cosine"
            - "knn"

        Returns:
        np.array: Pairwise distance matrix.
        """
        self.__check_for_emb_space__(emb_space_name)
        # check if distance matrix is already calculated

        if "distance" not in self.emb[emb_space_name].keys():
            self.emb[emb_space_name]["distance"] = dict()
        if metric in self.emb[emb_space_name]["distance"].keys():
            pwd = self.emb[emb_space_name]["distance"][metric]
        else:
            pwd = self.__calculate_pwd__(emb_space_name, metric)
            self.emb[emb_space_name]["distance"][metric] = pwd
        if rank:
            pwd = self.get_distance_percentiles(emb_space_name, metric)
        return pwd

    def get_sample_distance_difference(
        self,
        emb_space_name_1: str,
        emb_space_name_2: str,
        distance_metric: str,
        rank: bool = False,
    ):
        for emb_space_name in [emb_space_name_1, emb_space_name_2]:
            self.__check_for_emb_space__(emb_space_name)

        pwd_1 = self.get_sample_distance(
            emb_space_name_1, distance_metric, rank
        )
        pwd_2 = self.get_sample_distance(
            emb_space_name_2, distance_metric, rank
        )

        delta_emb_pwd = pwd_1 - pwd_2

        return delta_emb_pwd

    def plot_distances_per_rank(
        self, emb_space_name: str, distance_metric: str
    ):
        self.__check_for_emb_space__(emb_space_name)
        emb_pwd = self.get_sample_distance(
            emb_space_name=emb_space_name,
            metric=distance_metric,
        )
        df = global_percentiles_bins(emb_pwd)
        fig = px.line(
            df,
            x="values",
            y=[0] * len(df),
            color="percentile",
            line_shape="hv",
            title="",
            labels={
                "values": f"{distance_metric_aliases[distance_metric]} distance in {emb_space_name}",
            },
            # alternate between light and dark grey
            color_discrete_sequence=["lightgrey", "darkgrey"] * 50,
        )
        # add percentile as text
        for percentile in df["percentile"].unique():
            if percentile == 0:
                continue
            # y to be 2 if even percentile, 1 if odd percentile
            if percentile / 10 % 2 == 0:
                y_value = 1.5
            else:
                y_value = 1
            # get first row of the percentile
            x_value = df[df["percentile"] == percentile]["values"].min()
            fig.add_annotation(
                x=x_value,
                y=y_value,
                text=f"{percentile}th",
                showarrow=False,
            )
        fig.update_traces(line=dict(width=20))
        fig = update_fig_layout(fig)
        fig.update_layout(
            yaxis=dict(visible=False),  # remove y axis
            showlegend=False,  # remove legend
        )
        # adjust size to be very narrow
        fig.update_layout(
            width=1000,
            height=200,
        )
        return fig

    def visualise_emb_pca(self, emb_space_name: str, colour: str = None):
        self.__check_for_emb_space__(emb_space_name)
        if not colour:
            self.__calculate_clusters__(emb_space_name, n_clusters=None)
            colour = "cluster_" + emb_space_name

        self.__check_col_in_meta_data__(colour)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self.emb[emb_space_name]["emb"])

        fig = get_scatter_plot(
            emb_object=self,
            emb_space_name=emb_space_name,
            colour=colour,
            X_2d=X_2d,
            method="PCA",
        )
        return fig

    def visualise_emb_umap(self, emb_space_name: str, colour: str = None):
        self.__check_for_emb_space__(emb_space_name)
        if not colour:
            self.__calculate_clusters__(emb_space_name, n_clusters=None)
            colour = "cluster_" + emb_space_name
        self.__check_col_in_meta_data__(colour)

        umap_data = umap.UMAP(n_components=2, random_state=8)
        X_2d = umap_data.fit_transform(self.emb[emb_space_name]["emb"])

        fig = get_scatter_plot(
            emb_object=self,
            emb_space_name=emb_space_name,
            colour=colour,
            X_2d=X_2d,
            method="UMAP",
        )
        return fig

    def visualise_emb_tsne(
        self, emb_space_name: str, colour: str = None, perplexity=10
    ):
        self.__check_for_emb_space__(emb_space_name)
        if not colour:
            self.__calculate_clusters__(emb_space_name, n_clusters=None)
            colour = "cluster_" + emb_space_name
        self.__check_col_in_meta_data__(colour)

        tsne = TSNE(n_components=2, random_state=8, perplexity=perplexity)
        X_2d = tsne.fit_transform(self.emb[emb_space_name]["emb"])

        fig = get_scatter_plot(
            emb_object=self,
            emb_space_name=emb_space_name,
            colour=colour,
            X_2d=X_2d,
            method="t-SNE",
        )
        return fig

    def plot_feature_cluster_overlap(self, emb_space_name: str, feature: str):
        self.__check_col_in_meta_data__(feature)
        self.__check_for_emb_space__(emb_space_name)

        # plot a bar plot with features on x-axis and number of samples on y-axis
        # coloured by the different clusters

        fig = px.bar(
            self.meta_data,
            x=feature,
            color="cluster_" + emb_space_name,
            title=f"Agreement between unsupervised clusters and {feature} clusters for {emb_space_name} embeddings",
            labels={
                "cluster_" + emb_space_name: "Unsupervised cluster",
                feature: feature.capitalize(),
                "count": "Number of samples",
            },
            color_discrete_map=self.colour_map["cluster_" + emb_space_name],
        )
        # order x-axis categories by category name
        fig.update_xaxes(categoryorder="category ascending")
        fig = update_fig_layout(fig)
        return fig

    def plot_emb_hist(
        self,
    ):
        # merge embedding data into one dataframe with emb_name as column
        df_emb = pd.DataFrame(columns=["emb_space", "emb_values"])
        for emb_space in self.emb.keys():
            emb_values = self.emb[emb_space]["emb"].flatten()
            emb_space_col = [emb_space] * len(emb_values)
            df_emb = pd.concat(
                [
                    df_emb,
                    pd.DataFrame(
                        {"emb_space": emb_space_col, "emb_values": emb_values}
                    ),
                ]
            )
        fig = px.histogram(
            df_emb,
            x="emb_values",
            color="emb_space",
            labels={"emb_values": "Embedding values"},
            title="Distribution of embedding values",
            color_discrete_map={
                emb: self.emb[emb]["colour"] for emb in self.emb.keys()
            },
            marginal="box",
            opacity=0.5,
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        fig = update_fig_layout(fig)
        return fig

    def plot_emb_box(self, group: str):
        # group: either "sample" or "meta_col"
        if group != "sample":
            self.__check_col_in_meta_data__(group)
        df_emb = pd.DataFrame(columns=["emb_space", "emb_group", "emb_values"])
        for emb_space in self.emb.keys():
            emb_values = self.emb[emb_space]["emb"].flatten()
            if group == "sample":
                emb_group = list(
                    itertools.chain.from_iterable(
                        itertools.repeat(
                            x, self.emb[emb_space]["emb"].shape[1]
                        )
                        for x in self.sample_names
                    )
                )
            else:
                emb_group = list(
                    itertools.chain.from_iterable(
                        itertools.repeat(
                            x, self.emb[emb_space]["emb"].shape[1]
                        )
                        for x in self.meta_data[group].tolist()
                    )
                )
            emb_space_col = [emb_space] * len(emb_values)
            df_emb = pd.concat(
                [
                    df_emb,
                    pd.DataFrame(
                        {
                            "emb_space": emb_space_col,
                            "emb_group": emb_group,
                            "emb_values": emb_values,
                        }
                    ),
                ]
            )
        fig = px.box(
            df_emb,
            x="emb_group",
            y="emb_values",
            color="emb_space",
            title="Distribution of embedding values per {}".format(group),
            color_discrete_map={
                emb: self.emb[emb]["colour"] for emb in self.emb.keys()
            },
            labels={
                "emb_values": "Embedding values",
                "emb_group": group.capitalize(),
            },
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        fig.update_xaxes(categoryorder="category ascending")
        fig = update_fig_layout(fig)
        return fig

    def plot_emb_dist_heatmap(
        self,
        emb_space_name: str,
        distance_metric: str,
        order_x: str = None,
        order_y: str = None,
        rank: bool = False,
    ):
        self.__check_for_emb_space__(emb_space_name)
        if order_x is not None:
            self.__check_col_in_meta_data__(order_x)
        if order_y is not None:
            self.__check_col_in_meta_data__(order_y)

        emb_pwd = self.get_sample_distance(
            emb_space_name=emb_space_name,
            metric=distance_metric,
            rank=rank,
        )

        # find indices of samples for each order_x and order_y
        if order_x is not None:
            # find all sample ids for each order_x from meta_data
            order_x_indices = [
                self.meta_data[self.meta_data[order_x] == x].index.tolist()
                for x in self.meta_data[order_x].unique()
            ]
            order_x_indices = list(
                itertools.chain.from_iterable(order_x_indices)
            )
            emb_pwd = emb_pwd[order_x_indices, :]
        else:
            order_x_indices = list(range(len(self.sample_names)))

        if order_y is not None:
            # find all sample ids for each order_y from meta_data
            order_y_indices = [
                self.meta_data[self.meta_data[order_y] == y].index.tolist()
                for y in self.meta_data[order_y].unique()
            ]
            order_y_indices = list(
                itertools.chain.from_iterable(order_y_indices)
            )
            emb_pwd = emb_pwd[:, order_y_indices]
        else:
            order_y_indices = list(range(len(self.sample_names)))

        if rank:
            title = f"Rank of {distance_metric_aliases[distance_metric]} distance matrix of {emb_space_name} embedding space"
        else:
            title = f"{distance_metric_aliases[distance_metric]} distance matrix of {emb_space_name} embedding space"

        fig = px.imshow(
            emb_pwd,
            labels=dict(
                x="Sample",
                y="Sample",
                color=f"{distance_metric_aliases[distance_metric]} Distance",
            ),
            x=[self.sample_names[i] for i in order_x_indices],
            y=[self.sample_names[i] for i in order_y_indices],
            title=title,
            color_continuous_scale="Reds",
        )
        fig = update_fig_layout(fig)
        return fig

    def plot_emb_dist_box(
        self,
        group: str,
        distance_metric: str,
    ):
        # group: either "sample" or "meta_col"
        if group != "sample":
            self.__check_col_in_meta_data__(group)
        df_emb = pd.DataFrame(columns=["emb_space", "emb_group", "emb_values"])
        for emb_space in self.emb.keys():
            emb_dist_values = self.get_sample_distance(
                emb_space, distance_metric
            ).flatten()
            if group == "sample":
                emb_group = list(
                    itertools.chain.from_iterable(
                        itertools.repeat(
                            x, self.emb[emb_space]["emb"].shape[0]
                        )
                        for x in self.sample_names
                    )
                )
            else:
                emb_group = list(
                    itertools.chain.from_iterable(
                        itertools.repeat(
                            x, self.emb[emb_space]["emb"].shape[0]
                        )
                        for x in self.meta_data[group].tolist()
                    )
                )
            emb_space_col = [emb_space] * len(emb_dist_values)
            df_emb = pd.concat(
                [
                    df_emb,
                    pd.DataFrame(
                        {
                            "emb_space": emb_space_col,
                            "emb_group": emb_group,
                            "emb_values": emb_dist_values,
                        }
                    ),
                ]
            )
        fig = px.box(
            df_emb,
            x="emb_group",
            y="emb_values",
            color="emb_space",
            title=f"Distribution of {distance_metric_aliases[distance_metric]} distance values per {group}",
            color_discrete_map={
                emb: self.emb[emb]["colour"] for emb in self.emb.keys()
            },
            labels={
                "emb_values": f"{distance_metric_aliases[distance_metric]} distance",
                "emb_group": "",
            },
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        fig.update_xaxes(categoryorder="category ascending")
        fig = update_fig_layout(fig)
        return fig

    def plot_emb_dist_scatter(
        self,
        emb_space_name_1: str,
        emb_space_name_2: str,
        distance_metric: str,
        colour_group: str = None,
        colour_value_1: str = None,
        colour_value_2: str = None,
        rank: bool = False,
    ):
        self.__check_for_emb_space__(emb_space_name_1)
        self.__check_for_emb_space__(emb_space_name_2)

        if colour_group is not None:
            self.__check_col_in_meta_data__(colour_group)
            # check if subset_group_value is in the meta_data
            if colour_value_1 is not None:
                if colour_value_1 not in self.meta_data[colour_group].unique():
                    raise ValueError(
                        f"{colour_value_1} not found in {colour_group}"
                    )
                if colour_value_2 is not None:
                    if (
                        colour_value_2
                        not in self.meta_data[colour_group].unique()
                    ):
                        raise ValueError(
                            f"{colour_value_2} not found in {colour_group}"
                        )
            else:
                raise ValueError(
                    "Please provide a colour_value or set colour_group=None"
                )

        emb_pwd_1 = self.get_sample_distance(
            emb_space_name_1, distance_metric, rank
        )
        emb_pwd_2 = self.get_sample_distance(
            emb_space_name_2, distance_metric, rank
        )

        if colour_group is not None:
            if colour_value_2 is None:
                group_indices = self.meta_data[
                    self.meta_data[colour_group] == colour_value_1
                ].index.tolist()
                non_group_indices = list(
                    set(
                        set(range(0, len(self.sample_names)))
                        - set(group_indices)
                    )
                )

                colour = []

                for i in range(len(self.sample_names)):
                    for j in range(i + 1, len(self.sample_names)):
                        if i in group_indices and j in group_indices:
                            colour.append("group")
                        elif i in non_group_indices and j in non_group_indices:
                            colour.append("non_group")
                        else:
                            colour.append("mixed")

                    colour_map = {
                        "group": self.colour_map[colour_group][colour_value_1],
                        "non_group": "lightgray",
                        "mixed": "lightsteelblue",
                    }
            else:
                group_indices_1 = self.meta_data[
                    self.meta_data[colour_group] == colour_value_1
                ].index.tolist()
                group_indices_2 = self.meta_data[
                    self.meta_data[colour_group] == colour_value_2
                ].index.tolist()
                colour = []

                for i in range(len(self.sample_names)):
                    for j in range(i + 1, len(self.sample_names)):
                        if (i in group_indices_1 and j in group_indices_2) or (
                            i in group_indices_2 and j in group_indices_1
                        ):
                            colour.append("cross")
                        elif i in group_indices_1 and j in group_indices_1:
                            colour.append(f"{colour_value_1}")
                        elif i in group_indices_2 and j in group_indices_2:
                            colour.append(f"{colour_value_2}")
                        else:
                            colour.append("non_group")

                    colour_map = {
                        "cross": "steelblue",
                        f"{colour_value_1}": self.colour_map[colour_group][
                            colour_value_1
                        ],
                        f"{colour_value_2}": self.colour_map[colour_group][
                            colour_value_2
                        ],
                        "non_group": "lightgray",
                    }

        else:
            colour = None
            colour_map = None

        sample_names = []
        # add sample names
        for i in range(len(self.sample_names)):
            for j in range(i + 1, len(self.sample_names)):
                sample_names.append(
                    f"{self.sample_names[i]} - {self.sample_names[j]}"
                )

        fig = px.scatter(
            x=convert_to_1d_array(emb_pwd_1),
            y=convert_to_1d_array(emb_pwd_2),
            labels={
                "x": f"{emb_space_name_1} {distance_metric_aliases[distance_metric]} distance",
                "y": f"{emb_space_name_2} {distance_metric_aliases[distance_metric]} distance",
            },
            title=f"Scatter plot of {distance_metric_aliases[distance_metric]} distance values between {emb_space_name_1} and {emb_space_name_2}",
            opacity=0.4,
            color=colour,
            color_discrete_map=colour_map,
            hover_data={"Sample pair": sample_names},
            hover_name=sample_names,
        )
        # add line at 45 degrees dashed
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max(emb_pwd_1.max(), emb_pwd_2.max()),
            y1=max(emb_pwd_1.max(), emb_pwd_2.max()),
            line=dict(color="black", width=2, dash="dash"),
        )
        # adjust x and y axis to be the same scale
        fig.update_xaxes(
            range=[0, max(emb_pwd_1.max() * 1.1, emb_pwd_2.max() * 1.1)],
            title=f"{emb_space_name_1} {distance_metric_aliases[distance_metric]} distance",
        )
        fig.update_yaxes(
            range=[0, max(emb_pwd_1.max() * 1.1, emb_pwd_2.max() * 1.1)],
            title=f"{emb_space_name_2} {distance_metric_aliases[distance_metric]} distance",
        )
        fig.update_layout(width=800, height=800)
        fig = update_fig_layout(fig)

        return fig

    def plot_emb_dist_hist(self, distance_metric):
        for emb_space_name in self.emb.keys():
            if "distance" not in self.emb[emb_space_name].keys():
                self.emb[emb_space_name]["distance"] = dict()
            if (
                distance_metric
                not in self.emb[emb_space_name]["distance"].keys()
            ):
                self.emb[emb_space_name]["distance"][distance_metric] = (
                    self.get_sample_distance(emb_space_name, distance_metric)
                )

        fig = go.Figure()
        for emb_space_name in self.emb.keys():
            fig.add_trace(
                go.Histogram(
                    x=squareform(
                        self.emb[emb_space_name]["distance"][distance_metric]
                    ),
                    name=emb_space_name,
                    opacity=0.8,
                    marker_color=self.emb[emb_space_name]["colour"],
                )
            )
        fig.update_layout(
            title=f"Distribution of {distance_metric_aliases[distance_metric]} distance values for {len(self.emb.keys())} embedding spaces",
            xaxis_title=f"{distance_metric_aliases[distance_metric]} distance",
            yaxis_title="Number of samples",
            barmode="overlay",
        )
        fig = update_fig_layout(fig)
        return fig

    def plot_emb_dis_dif_heatmap(
        self,
        emb_space_name_1: str,
        emb_space_name_2: str,
        distance_metric: str,
        rank: bool = False,
    ):
        for emb_space_name in [emb_space_name_1, emb_space_name_2]:
            self.__check_for_emb_space__(emb_space_name)
        emb_pwd_diff = self.get_sample_distance_difference(
            emb_space_name_1=emb_space_name_1,
            emb_space_name_2=emb_space_name_2,
            distance_metric=distance_metric,
            rank=rank,
        )
        if rank:
            title = f"Rank of {distance_metric_aliases[distance_metric]} distance difference matrix from {emb_space_name_1} to {emb_space_name_2} embedding space"
        else:
            title = f"{distance_metric_aliases[distance_metric]} distance difference matrix from {emb_space_name_1} to {emb_space_name_2} embedding space"
        fig = px.imshow(
            emb_pwd_diff,
            labels=dict(
                x="Sample",
                y="Sample",
                color="",
            ),
            x=self.sample_names,
            y=self.sample_names,
            title=title,
            color_continuous_scale="rdbu",
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12, color="black"),
        )
        return fig

    def plot_emb_dis_dif_box(
        self, emb_space_name_1, emb_space_name_2, group, distance_metric
    ):
        self.__check_col_in_meta_data__(group)

        group_ids = self.__sample_indices_to_groups__(group=group)
        delta_emb_pwd = self.get_sample_distance_difference(
            emb_space_name_1=emb_space_name_1,
            emb_space_name_2=emb_space_name_2,
            distance_metric=distance_metric,
        )

        delta_emb_pwd_per_group = {
            group_name: dict() for group_name in group_ids.keys()
        }
        for group_name, indices in group_ids.items():
            other_indices = list(
                set(list(range(len(self.sample_names)))) - set(indices)
            )
            pairs_within_group = generate_unique_pairs(indices)
            pairs_with_outside_group = generate_cross_list_pairs(
                indices_1=indices, indices_2=other_indices
            )
            # get delta_emb_pwd for pairs_within_group
            delta_emb_pwd_per_group[group_name]["within_group"] = [
                delta_emb_pwd[pair] for pair in pairs_within_group
            ]
            delta_emb_pwd_per_group[group_name]["outside_group"] = [
                delta_emb_pwd[pair] for pair in pairs_with_outside_group
            ]
        df = pd.DataFrame(columns=["group", "distance", "pair_type"])
        for group_name, delta_emb_pwd in delta_emb_pwd_per_group.items():
            for pair_type, delta_emb_pwd_values in delta_emb_pwd.items():
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "group": [group_name]
                                * len(delta_emb_pwd_values),
                                "distance": delta_emb_pwd_values,
                                "pair_type": [pair_type]
                                * len(delta_emb_pwd_values),
                            }
                        ),
                    ]
                )
        fig = px.box(
            df,
            x="group",
            y="distance",
            color="pair_type",
            title=f"Distribution of {distance_metric_aliases[distance_metric]} distance difference values per {group}",
            color_discrete_map={
                "within_group": "slategray",
                "outside_group": "lightsteelblue",
            },
            labels={
                "distance": f"{distance_metric_aliases[distance_metric]} distance difference",
                "group": "",
            },
        )
        # rename legend with more descriptive names
        fig.for_each_trace(
            lambda trace: trace.update(
                name=(
                    "Within group"
                    if trace.name == "within_group"
                    else "Outside group"
                )
            )
        )
        # hide legend title
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12),
            legend=dict(
                title=None,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        fig.update_xaxes(categoryorder="category ascending")
        fig = update_fig_layout(fig)
        return fig

    def plot_emb_dist_dif_percentiles(
        self,
        emb_space_name_1,
        emb_space_name_2,
        distance_metric,
        subset_group=None,
        subset_group_value=None,
        compare_subset_to=None,
    ):
        percentiles_1 = self.get_distance_percentiles(
            emb_space_name_1, distance_metric
        )

        percentiles_2 = self.get_distance_percentiles(
            emb_space_name_2, distance_metric
        )
        # set all values on the diagonal to 0
        np.fill_diagonal(percentiles_1, 0)
        np.fill_diagonal(percentiles_2, 0)

        # if subset is provided, only use the subset
        if subset_group is not None:
            self.__check_col_in_meta_data__(subset_group)
            # check if subset_group_value is in the meta_data
            if subset_group_value is not None:
                if (
                    subset_group_value
                    not in self.meta_data[subset_group].unique()
                ):
                    raise ValueError(
                        f"{subset_group_value} not found in {subset_group}"
                    )

            # get indices of the subset
            subset_indices = self.meta_data[
                self.meta_data[subset_group] == subset_group_value
            ].index.tolist()

            if compare_subset_to is not None:
                if compare_subset_to == "within_group":
                    percentiles_1 = percentiles_1[subset_indices, :][
                        :, subset_indices
                    ]
                    percentiles_2 = percentiles_2[subset_indices, :][
                        :, subset_indices
                    ]
                elif compare_subset_to == "outside_group":
                    other_indices = list(
                        set(
                            set(range(len(self.sample_names)))
                            - set(subset_indices)
                        )
                    )
                    percentiles_1 = percentiles_1[subset_indices, :][
                        :, other_indices
                    ]
                    percentiles_2 = percentiles_2[subset_indices, :][
                        :, other_indices
                    ]
                else:
                    raise ValueError(
                        f"Invalid value for compare_subset_to: {compare_subset_to}. \
                            Must be either 'within_group' or 'outside_group'"
                    )
            else:
                percentiles_1 = percentiles_1[subset_indices, :]
                percentiles_2 = percentiles_2[subset_indices, :]

        percentiles_1 = convert_to_1d_array(percentiles_1)
        percentiles_2 = convert_to_1d_array(percentiles_2)

        # get uniqiue percentiles
        percentiles = np.unique(np.concatenate((percentiles_1, percentiles_2)))
        percentiles.sort()

        # create a mapping from percentile to index
        percentile_to_index = {
            percentile: i for i, percentile in enumerate(percentiles)
        }

        # initialise a count matrix
        count_matrix = np.zeros((len(percentiles), len(percentiles)))

        # populate the count matrix
        for i in range(len(percentiles_1)):
            count_matrix[
                percentile_to_index[percentiles_1[i]],
                percentile_to_index[percentiles_2[i]],
            ] += 1

        # Prepare Sankey diagram data
        source = []
        target = []
        value = []

        for i in range(len(percentiles)):
            for j in range(len(percentiles)):
                if count_matrix[i, j] > 0:
                    source.append(i)
                    target.append(
                        j + len(percentiles)
                    )  # Offset target indices for better separation
                    value.append(count_matrix[i, j])

        df_links = pd.DataFrame(
            {"source": source, "target": target, "value": value}
        )
        df_links.sort_values(["source", "target"], inplace=True)
        source = df_links["source"].tolist()
        target = df_links["target"].tolist()
        value = df_links["value"].tolist()

        # define node labelss
        labels = [
            f"{percentile} {emb_space_name_1}" for percentile in percentiles
        ]
        labels += [
            f"{percentile} {emb_space_name_2}" for percentile in percentiles
        ]
        # colours of the respective embedding_space
        colors = [
            self.emb[emb_space_name_1]["colour"]
            for _ in range(len(percentiles))
        ] + [
            self.emb[emb_space_name_2]["colour"]
            for _ in range(len(percentiles))
        ]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color=colors,
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=["lightsteelblue" for _ in value],
                    ),
                    textfont=dict(color="black", family="Arial", size=12),
                )
            ]
        )
        # set dimensions
        fig.update_layout(
            width=800,
            height=800,
            autosize=False,
        )
        fig = update_fig_layout(fig)

        return fig


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
    # check that at least 11 entries in the array
    if len(arr) < 11:
        raise ValueError("Array must have at least 11 entries.")
    flattened = squareform(arr)
    sorted_indices = np.argsort(
        flattened
    )  # returns the indices that would sort the array
    ranks = np.argsort(sorted_indices)  # get the ranks of the sorted indices
    percentiles = (ranks / (len(flattened) - 1)) * 100  # calculate percentiles
    percentiles_rounded_up = np.ceil(percentiles / 10) * 10
    percentiles_rounded_up[percentiles_rounded_up == 0] = (
        10  # replace 0 with 10
    )
    percentile_array = squareform(percentiles_rounded_up)
    return percentile_array


def global_percentiles_bins(arr):
    if len(arr) < 11:
        raise ValueError(
            "Array must have at least 11 entries in order to compute ranks."
        )
    flattened = squareform(arr)
    sorted_indices = np.argsort(flattened)
    ranks = np.argsort(sorted_indices)
    percentiles = (ranks / (len(flattened) - 1)) * 100
    percentiles_rounded_up = np.ceil(percentiles / 10) * 10
    percentile_df = pd.DataFrame(columns=["percentile", "values"])
    for i in range(0, 110, 10):
        values = flattened[percentiles_rounded_up == i]
        percentile_df = pd.concat(
            [
                percentile_df,
                pd.DataFrame(
                    {"percentile": [i] * len(values), "values": values}
                ),
            ]
        )
    return percentile_df


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

    # make square aspect ratio
    fig.update_layout(
        width=800,
        height=800,
        autosize=False,
    )

    # make dots proportional to number of samples
    fig.update_traces(
        marker=dict(size=max(10, (1 / len(emb_object.sample_names)) * 400))
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
