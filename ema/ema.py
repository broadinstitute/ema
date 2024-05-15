import pandas as pd
import plotly.express as px
import numpy as np
import itertools

from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform


class EmbeddingHandler:
    def __init__(self, sample_meta_data: pd.DataFrame):
        self.meta_data = sample_meta_data
        self.sample_names = self.meta_data.iloc[:, 0].tolist()
        self.colour_map = self.__get_colour_map_for_features__()
        self.emb = dict()
        print(f"{len(self.sample_names)} samples loaded.")
        print(f"Meta data columns: {self.meta_data.columns}")
        return

    def __get_colour_map_for_features__(self) -> dict:
        if len(self.meta_data.columns) == 1:
            print("No meta data provided. Cannot generate colour map.")
            return None
        colour_map = dict()
        for column in self.meta_data.columns[1:]:
            column_values = self.meta_data[column].unique()
            if len(column_values) > 20:
                print(
                    f"Column {column} has more than 20 unique values. \
                        Skipping."
                )
                continue
            colour_map[column] = dict()
            for i, value in enumerate(column_values):
                colour_map[column][value] = px.colors.qualitative.Set2[i]
        return colour_map

    def __check_for_emb_space__(self, emb_space_name: str) -> None:
        if emb_space_name not in self.emb.keys():
            raise ValueError  # Add error message
        else:
            return True

    def __check_col_in_meta_data__(self, col) -> None:
        if col not in self.meta_data.columns:
            raise ValueError(f"Column {col} not found in meta data.")

    def add_emb_space(self, embeddings: np.array, emb_space_name: str) -> None:
        if emb_space_name in self.emb.keys():
            raise ValueError  # Add error message
        if embeddings.shape[0] != len(self.sample_names):
            raise ValueError  # Add error message
        self.emb[emb_space_name] = dict()
        self.emb[emb_space_name]["emb"] = embeddings
        return

    def remove_emb_space(self, emb_space_name: str) -> None:
        self.__check_for_emb_space__(emb_space_name)
        del self.emb[emb_space_name]
        return

    def get_emb(self, emb_space_name: str):
        self.__check_for_emb_space__(emb_space_name)
        return self.emb[emb_space_name]["emb"]

    def get_sample_distance(
        self, emb_space_name: str, metric: str
    ) -> np.array:
        self.__check_for_emb_space__(emb_space_name)
        # TODO check metric is valid
        if "distance" not in self.emb[emb_space_name].keys():
            self.emb[emb_space_name]["distance"] = dict()
        if metric in self.emb[emb_space_name]["distance"].keys():
            return self.emb[emb_space_name]["distance"][metric]
        emb_pwd = squareform(
            pdist(self.emb[emb_space_name]["emb"], metric=metric)
        )
        self.emb[emb_space_name]["distance"][metric] = emb_pwd
        return emb_pwd

    def get_clusters(self, emb_space_name: str):
        self.__check_for_emb_space__(emb_space_name)
        # TODO check if clusters are already calculated
        # TODO calculate clusters
        # TODO add clusters to self.meta[emb-space-name-clusters]
        # TODO run __get_colour_map_for_features__ and add to self.colour_map
        # TODO return clusters as list(?)
        pass

    def get_sample_distance_difference(
        self,
        emb_space_name_1: str,
        emb_space_name_2: str,
        distance_metric: str,
    ):
        for emb_space_name in [emb_space_name_1, emb_space_name_2]:
            self.__check_for_emb_space__(emb_space_name)

        for emb_space_name in [emb_space_name_1, emb_space_name_2]:
            if "distance" not in self.emb[emb_space_name].keys():
                self.emb[emb_space_name]["distance"] = dict()
            if (
                distance_metric
                not in self.emb[emb_space_name]["distance"].keys()
            ):
                self.emb[emb_space_name]["distance"][distance_metric] = (
                    self.get_sample_distance(emb_space_name, distance_metric)
                )

        delta_emb_pwd = (
            self.emb[emb_space_name_1]["distance"][distance_metric]
            - self.emb[emb_space_name_2]["distance"][distance_metric]
        )

        return delta_emb_pwd

    def sample_indices_to_groups(self, group: str):
        self.__check_col_in_meta_data__(group)
        group_indices = dict()
        for group_name in self.meta_data[group].unique():
            group_indices[group_name] = self.meta_data[
                self.meta_data[group] == group_name
            ].index.tolist()
        return group_indices

    def visualise_emb_tsne(
        self, emb_space_name: str, colour: str, perplexity=10
    ):
        self.__check_for_emb_space__(emb_space_name)
        self.__check_col_in_meta_data__(colour)

        tsne = TSNE(n_components=2, random_state=8, perplexity=perplexity)
        X_2d = tsne.fit_transform(self.emb[emb_space_name]["emb"])
        fig = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            color=self.meta_data[colour],
            labels={"color": "Cluster"},
            title=f"t-SNE visualization of variant embeddings of {emb_space_name} embeddings",
            size=[2] * len(self.sample_names),
            hover_data={
                "Sample": self.sample_names,
            },
            color_discrete_map=self.colour_map[colour],
        )
        fig.update_layout(
            template="plotly_white", font=dict(size=12, family="Arial")
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
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
                emb: px.colors.qualitative.Safe[i]
                for i, emb in enumerate(self.emb.keys())
            },
            marginal="box",
            opacity=0.5,
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
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
                emb: px.colors.qualitative.Safe[i]
                for i, emb in enumerate(self.emb.keys())
            },
            labels={"emb_values": "Embedding values", "emb_group": ""},
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        return fig

    def plot_emb_dist_heatmap(
        self, emb_space_name: str, group: str, distance_metric: str
    ):
        self.__check_for_emb_space__(emb_space_name)
        self.__check_col_in_meta_data__(group)

        if "distance" not in self.emb[emb_space_name].keys():
            self.emb[emb_space_name]["distance"] = dict()
        if distance_metric not in self.emb[emb_space_name]["distance"].keys():
            emb_pwd = self.get_sample_distance(emb_space_name, distance_metric)
        else:
            emb_pwd = self.emb[emb_space_name]["distance"][distance_metric]

        fig = px.imshow(
            emb_pwd,
            labels=dict(
                x="Sample", y="Sample", color=f"{distance_metric} Distance"
            ),
            x=self.sample_names,
            y=self.sample_names,
            title=f"{distance_metric} Distance Matrix of {emb_space_name} Embedding Space",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12),
        )
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
            title=f"Distribution of {distance_metric} distance values per {group}",
            color_discrete_map={
                emb: px.colors.qualitative.Safe[i]
                for i, emb in enumerate(self.emb.keys())
            },
            labels={
                "emb_values": f"{distance_metric} distance",
                "emb_group": "",
            },
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        return fig

    def plot_emb_dis_dif_heatmap(
        self, emb_space_name_1, emb_space_name_2, distance_metric
    ):
        for emb_space_name in [emb_space_name_1, emb_space_name_2]:
            self.__check_for_emb_space__(emb_space_name)
        emb_pwd_diff = self.get_sample_distance_difference(
            emb_space_name_1=emb_space_name_1,
            emb_space_name_2=emb_space_name_2,
            distance_metric=distance_metric,
        )
        fig = px.imshow(
            emb_pwd_diff,
            labels=dict(
                x="Sample",
                y="Sample",
                color=f"Difference in {distance_metric} Distance",
            ),
            x=self.sample_names,
            y=self.sample_names,
            title=f"{distance_metric} Distance Matrix of {emb_space_name_1} and {emb_space_name_2} Embedding Space",
            color_continuous_scale="Greens",
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12),
        )
        return fig

    def plot_emb_dis_dif_box(
        self, emb_space_name_1, emb_space_name_2, group, distance_metric
    ):
        self.__check_col_in_meta_data__(group)
        for emb_space_name in [emb_space_name_1, emb_space_name_2]:
            self.__check_for_emb_space__(emb_space_name)
        for emb_space_name in [emb_space_name_1, emb_space_name_2]:
            if "distance" not in self.emb[emb_space_name].keys():
                self.emb[emb_space_name]["distance"] = dict()
            if (
                distance_metric
                not in self.emb[emb_space_name]["distance"].keys()
            ):
                self.emb[emb_space_name]["distance"][distance_metric] = (
                    self.get_sample_distance(emb_space_name, distance_metric)
                )

        group_ids = self.sample_indices_to_groups(group="family")
        delta_emb_pwd = (
            self.emb[emb_space_name_1]["distance"][distance_metric]
            - self.emb[emb_space_name_2]["distance"][distance_metric]
        )
        # get mean of distances for each group
        for group_name, indices in group_ids.items():
            # get set of unique combinations of combinations of indices
            set()
            # find average of delta_emb_pwd for each group not within the group_ids
        print()
        pass
