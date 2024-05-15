import numpy as np
import pandas as pd
import plotly.express as px
import umap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from scipy import stats


"""
This script contains the PROTEIN HANDLER class which is used to handle
and compare different protein representations. The class compares
embeddings from different models and provides means to calculate
similarities and distances between them.

"""


def flatten_list(xss):
    return [x for xs in xss for x in xs]


class EMBEDDING_HANDLER:
    def __init__(
        self,
        sample_meta: pd.DataFrame,
        emb1_fp: str,
        emb2_fp: str,
        emb1_name: str = "emb1",
        emb2_name: str = "emb2",
    ):
        self.meta_data = pd.read_csv(sample_meta)
        self.sample_list = self.meta_data.iloc[:, 0].tolist()
        self.emb = dict()
        self.emb_names = []
        self.add_embedding(emb1_name, emb1_fp)
        self.add_embedding(emb2_name, emb2_fp)
        self.colour_map = self.__get_colour_map_for_features__()

    def __check_for_sample__(self, sample) -> None:
        if sample not in self.sample_list:
            raise ValueError(f"Sample {sample} not found.")

    def __check_for_embedding__(self, emb_name) -> None:
        if emb_name not in self.emb:
            raise ValueError(f"Embedding {emb_name} not found.")

    def __check_col_in_meta_data__(self, col) -> None:
        if col not in self.meta_data.columns:
            raise ValueError(f"Column {col} not found in meta data.")

    def __get_colour_map_for_features__(self) -> dict:
        if len(self.meta_data.columns) == 1:
            print("No meta data provided. Cannot generate colour map.")
            return None
        colour_map = dict()
        for column in self.meta_data.columns[1:]:
            column_values = self.meta_data[column].unique()
            if len(column_values) > 20:
                print(
                    f"Column {column} has more than 20 unique values. Skipping."
                )
                continue
            colour_map[column] = dict()
            for i, value in enumerate(column_values):
                colour_map[column][value] = px.colors.qualitative.Set2[i]
        return colour_map

    def __plot_histogram__(self, values: list, title: str) -> px.histogram:
        fig = px.histogram(
            values,
            labels={"x": "Cluster"},
            title=title,
            color_discrete_sequence=["#8B0000"],
            marginal="box",
            nbins=20,
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(size=18, family="Arial"),
            showlegend=False,
        )
        fig.show()
        return fig

    def get_mean_and_std_of_emb_values(self, emb_name: str) -> tuple:
        """
        Returns the mean and standard deviation of the embedding values
        for a given embedding.
        """
        self.__check_for_embedding__(emb_name)
        embedding_values_mean = np.mean(self.emb[emb_name])
        embedding_values_std = np.std(self.emb[emb_name])
        return embedding_values_mean, embedding_values_std

    def fit_to_normal_distribution(self, emb_name: str) -> tuple:
        """
        Fits the embedding values to a normal distribution and returns
        the mean and standard deviation of the fitted distribution.
        """
        self.__check_for_embedding__(emb_name)
        embedding_values = self.emb[emb_name].flatten()
        mu, std = norm.fit(embedding_values)
        return mu, std

    def plot_histogram_of_emb_values(self, emb_name: str) -> px.histogram:
        self.__check_for_embedding__(emb_name)
        embedding_values = self.emb[emb_name].flatten()
        return self.__plot_histogram__(
            values=embedding_values,
            title=f"Histogram of embedding values for {emb_name}",
        )

    def plot_distribution_of_all_emb_values(self) -> px.histogram:
        df = pd.DataFrame()
        df["embedding"] = flatten_list(
            [self.emb[emb_name].flatten() for emb_name in self.emb_names]
        )
        df["embedding_name"] = [
            emb_name
            for emb_name in self.emb_names
            for _ in range(len(self.emb[emb_name].flatten()))
        ]
        fig = px.histogram(
            df,
            x="embedding",
            color="embedding_name",
            labels={"x": "Cluster"},
            title="Histogram of all embedding values",
            color_discrete_map={
                emb_name: px.colors.qualitative.Safe[i]
                for i, emb_name in enumerate(self.emb_names)
            },
            marginal="box",
            opacity=0.8,
        )
        fig.update_layout(
            template="plotly_white", font=dict(size=18, family="Arial")
        )
        fig.show()
        return fig

    def plot_distribution_of_emb_values(self, agg: str) -> px.scatter:
        """
        Plots the distribution of embedding values for each sample or feature.

        Parameters:
        agg (str): Aggregation type. Choose from 'sample', 'feature', 'all'.

        Returns:
        fig (plotly.graph_objs.Figure): Plotly figure object.
        """
        if agg == "sample":
            agg_axis = 1
        elif agg == "feature":
            agg_axis = 0
        elif agg == "all":
            return self.plot_distribution_of_all_emb_values()
        else:
            raise ValueError(
                "Aggregation type not understood. \
                    Choose from 'sample', 'feature', 'all'."
            )
        df = pd.DataFrame()
        # calculate mean and std of embedding values per sample
        for emb_name in self.emb_names:
            emb_values = self.emb[emb_name]
            embedding_values_mean = np.mean(emb_values, axis=agg_axis)
            embedding_values_std = np.std(emb_values, axis=agg_axis)
            if agg == "sample":
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "sample_name": self.sample_list,
                                "emb_mean": embedding_values_mean,
                                "emb_std": embedding_values_std,
                                "emb_space": [emb_name]
                                * len(self.sample_list),
                            }
                        ),
                    ]
                )
                hover_data_set = {"sample_name": True}
            elif agg == "feature":
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "emb_idx": range(emb_values.shape[1]),
                                "emb_mean": embedding_values_mean,
                                "emb_std": embedding_values_std,
                                "emb_space": [emb_name]
                                * len(embedding_values_mean),
                            }
                        ),
                    ]
                )
                hover_data_set = {"emb_idx": True}
        # plot scatter plot with error bars
        fig = px.scatter(
            df,
            x="emb_mean",
            y="emb_std",
            color="emb_space",
            color_discrete_map={
                emb_name: px.colors.qualitative.Safe[i]
                for i, emb_name in enumerate(self.emb_names)
            },
            labels={"x": "Mean embedding values", "y": "Std embedding values"},
            title=f"Mean and std of embedding values per sample for \
                {(', ').join([emb for emb in self.emb_names])}",
            hover_data=hover_data_set,
            opacity=0.8,
        )
        fig.update_layout(
            template="plotly_white", font=dict(size=18, family="Arial")
        )
        fig.show()
        return fig

    def add_embedding(self, emb_name: str, emb_fp: str) -> None:
        embeddings = np.load(emb_fp)
        assert embeddings.shape[0] == len(self.sample_list)
        self.emb[emb_name] = embeddings
        self.emb_names.append(emb_name)

    def get_embeddings(self, emb_name: str) -> np.ndarray:
        self.__check_for_embedding__(emb_name)
        return self.emb[emb_name]

    def get_embedding_names(self) -> list:
        return list(self.emb.keys())

    def get_embedding_for_sample(
        self, emb_name: str, sample: str
    ) -> np.ndarray:
        self.__check_for_embedding__(emb_name)
        self.__check_for_sample__(sample)
        return self.emb[emb_name][self.sample_list.index(sample)]

    def get_embedding_similarity(self, emb1: str, emb2: str) -> np.ndarray:
        return np.dot(self.emb[emb1], self.emb[emb2].T)

    def get_embedding_distance(
        self, emb1: str, emb2: str, normalisation=False
    ) -> np.ndarray:
        embedding_distance = np.linalg.norm(
            self.emb[emb1] - self.emb[emb2], axis=1, ord=2
        )
        if normalisation:
            embedding_distance = embedding_distance / np.linalg.norm(
                self.emb[emb1], axis=1, ord=2
            )
        return embedding_distance

    def get_tsne(self, data: np.array, n_components=2) -> np.ndarray:
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=10)
        # TODO: change perplexity to be variable
        return tsne.fit_transform(data)

    def visualise_emb_space_tsne(
        self, emb_name: str, colour: str
    ) -> px.scatter:
        """'
        Visualise the embedding space using t-SNE.

        Parameters:
        emb_name (str): Name of the embedding to visualise.
        colour (str): Name of the column in the meta data to use for colouring.

        Returns:
        fig (plotly.graph_objs.Figure): Plotly figure object.
        """

        self.__check_for_embedding__(emb_name)
        self.__check_col_in_meta_data__(colour)

        X_2d = self.get_tsne(self.emb[emb_name])

        fig = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            color=self.meta_data[colour],
            labels={"color": "Cluster"},
            title=f"t-SNE visualization of variant embeddings of {emb_name} embeddings",
            size=[8] * len(self.sample_list),
            hover_data={
                "Sample": self.sample_list,
            },
            color_discrete_map=self.colour_map[colour],
        )
        fig.update_layout(
            template="plotly_white", font=dict(size=18, family="Arial")
        )
        fig.show()
        return fig

    def get_umap(self, emb_name: str) -> np.ndarray:
        umap_ = umap.UMAP()
        return umap_.fit_transform(self.emb[emb_name])

    def get_pca(self, emb_name: str) -> np.ndarray:
        pca = PCA(n_components=2)
        return pca.fit_transform(self.emb[emb_name])

    def visualise_emb_space_umap(
        self, emb_name: str, colour: str
    ) -> px.scatter:
        """'
        Visualise the embedding space using UMAP.

        Parameters:
        emb_name (str): Name of the embedding to visualise.
        colour (str): Name of the column in the meta data to use for colouring.

        Returns:
        fig (plotly.graph_objs.Figure): Plotly figure object.
        """

        self.__check_for_embedding__(emb_name)
        self.__check_col_in_meta_data__(colour)

        X_2d = self.get_umap(emb_name)

        fig = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            color=self.meta_data[colour],
            labels={"color": "Cluster"},
            title=f"UMAP visualization of variant embeddings of {emb_name} embeddings",
            size=[8] * len(self.sample_list),
            hover_data={
                "Sample": self.sample_list,
            },
            color_discrete_map=self.colour_map[colour],
        )
        fig.update_layout(
            template="plotly_white", font=dict(size=18, family="Arial")
        )
        fig.show()
        return fig

    def visualise_emb_space_pca(
        self, emb_name: str, colour: str
    ) -> px.scatter:
        """'
        Visualise the embedding space using PCA.

        Parameters:
        emb_name (str): Name of the embedding to visualise.
        colour (str): Name of the column in the meta data to use for colouring.

        Returns:
        fig (plotly.graph_objs.Figure): Plotly figure object.
        """

        self.__check_for_embedding__(emb_name)
        self.__check_col_in_meta_data__(colour)

        X_2d = self.get_pca(emb_name)

        fig = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            color=self.meta_data[colour],
            labels={"color": "Cluster"},
            title=f"PCA visualization of variant embeddings of {emb_name} embeddings",
            size=[8] * len(self.sample_list),
            hover_data={
                "Sample": self.sample_list,
            },
            color_discrete_map=self.colour_map[colour],
        )
        fig.update_layout(
            template="plotly_white", font=dict(size=18, family="Arial")
        )
        fig.show()
        return fig

    def get_edm(self, emb_name: str, metric: str = "euclidean") -> np.ndarray:
        """Calculate the distance matrix for the embeddings."""
        distance_vector = pdist(self.emb[emb_name], metric=metric)
        distance_matrix = squareform(distance_vector)
        return distance_matrix

    def visualise_edm_distribution(
        self, metric: str = "euclidean"
    ) -> px.histogram:
        """Visualise the distribution of the embedding distances."""
        df = pd.DataFrame()
        for emb in self.emb_names:
            edm = self.get_edm(emb, metric=metric)
            distance_vector = squareform(edm)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "distance": distance_vector,
                            "embedding": [emb] * len(distance_vector),
                        }
                    ),
                ]
            )
        fig = px.histogram(
            df,
            x="distance",
            color="embedding",
            labels={"x": "Distance"},
            title=f"Distribution of embedding distances",
            color_discrete_map={
                emb: px.colors.qualitative.Safe[i]
                for i, emb in enumerate(self.emb_names)
            },
            marginal="box",
            opacity=0.8,
        )
        fig.update_layout(
            template="plotly_white", font=dict(size=18, family="Arial")
        )
        fig.show()
        return fig

    def get_similarity_scores_for_edm(self, edm):
        """Get similarity scores for the EDM."""
        return 1 / (1 + edm)

    def get_median_normalised_edm(self, edm):
        """Normalise the EDM by the median."""
        edm_norm = edm / np.median(edm)
        return edm_norm

    def get_similarity_scores_based_on_normal_distribution(self, edm):
        """Get similarity scores based on normal distribution."""
        # per sample
        edm_norm_dis = np.zeros(edm.shape)
        for i in range(edm.shape[0]):
            edm_norm_dis[i] = stats.norm.pdf(edm[i])
        return edm_norm_dis
