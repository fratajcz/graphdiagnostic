from functional import plot_disconnected_components, plot_paths, plot_degrees, plot_homophily, plot_metrics

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd


class GraphDiagnostic:
    def __init__(self, graph=None):
        self.graph = graph
        self.details = ["components", "paths", "degrees", "homophily", "metrics"]

    def get_diagnostics(self, details="", graph=None, save=True, kwargs: dict = {}):
        """details is empty string or list of strings"""

        # TODO: sort out and use kwargs!

        if type(details) == str:
            details = [details]

        if details[0] in ["", "all"]:
            details = self.details

        funcs = []
        kwarg_list = []
        title_list = []
        for detail in details:
            if detail.lower() in ["component", "components"]:
                funcs.append(self.check_components)
                kwarg_list.append({})
                title_list.append(detail)

            elif detail.lower() in ["path", "paths"]:
                funcs.append(self.check_paths)
                title_list.append(detail)
                kwarg_list.append({})
                if detail in kwargs.keys():
                    kwarg_list[-1].update(kwargs[detail])

                if self.check_is_directed(graph):
                    funcs.append(self.check_paths)
                    kwarg_list.append({"symmetrize": True})
                    title_list.append(detail + " (undirected)")

            elif detail.lower() in ["degree", "degrees"]:
                funcs.append(self.check_degrees)
                title_list.append(detail)
                kwarg_list.append({})
                if detail in kwargs.keys():
                    kwarg_list[-1].update(kwargs[detail])

                if self.check_is_directed(graph):
                    title_list[-1] += " (in)"
                    kwarg_list[-1].update({"direction": "in"})
                    funcs.append(self.check_degrees)
                    kwarg_list.append({"direction": "out"})
                    title_list.append(detail + " (out)")

            elif detail.lower() == "homophily":
                funcs.append(self.check_homophily)
                kwarg_list.append({})
                title_list.append(detail)
            elif detail.lower() == "metrics":
                funcs.append(self.check_metrics)
                kwarg_list.append({})
                title_list.append(detail)
            else:
                raise ValueError("No Detail '{}'. Only the following detail views are implemented: {}".format(detail, self.details))

            if detail in kwargs.keys():
                kwarg_list[-1].update(kwargs[detail])

        fig, axes = plt.subplots(1, len(funcs), figsize=(5 * len(funcs), 5.5), sharex=False, sharey=False)

        for i, detail_func in enumerate(funcs):
            try:
                fig, ax = detail_func(fig=fig, ax=axes[i], **kwarg_list[i])
            except TypeError:
                fig, ax = detail_func(fig=fig, ax=axes, **kwarg_list[i])

            ax.set_title(title_list[i][0].upper() + title_list[i][1:])
            plt.tight_layout()

        return fig, ax

    def check_metrics(self, graph=None, symmetrize=True, metrics=None, fig=None, ax=None):

        if metrics is None:
            metrics = self.get_metrics(graph, symmetrize)

        fig, ax = plot_metrics(metrics, fig=fig, ax=ax)
        return fig, ax

    def get_metrics(self, graph, symmetrize=True):
        import igraph as ig
        if graph is None:
            assert self.graph is not None
            graph = self.graph

        metrics = {}
        ppm, _ = self.positive_paths_matrix(graph, symmetrize)
        distance = ppm.flatten()
        nnz_distance = distance[distance != 0]
        avg_distance = nnz_distance.mean()
        sd_distance = nnz_distance.std()
        metrics.update({"Average Positive Shortest Path Length": (avg_distance, sd_distance)})
        G = graph.to_undirected()
        G0 = G.subgraph(max(nx.connected_components(G), key=len))
        G = ig.Graph.from_networkx(G0)
        diameter = G.diameter()
        shortest_path_lengths = G.shortest_paths()
        average_shortest_path_length = np.mean(shortest_path_lengths[shortest_path_lengths != 0])
        sd_shortest_path_length = np.std(shortest_path_lengths[shortest_path_lengths != 0])
        metrics.update({"Average Shortest Path Length": (average_shortest_path_length, sd_shortest_path_length)})
        metrics.update({"Diameter": (diameter,)})

        return metrics

    def check_components(self, graph=None, components=None, isolates=None, fig=None, ax=None):

        if components is None or isolates is None:
            components, isolates = self.get_connected_components(graph)
        fig, ax = plot_disconnected_components(components, isolates, fig=fig, ax=ax)
        return fig, ax

    def get_connected_components(self, graph=None) -> tuple:
        if graph is None:
            assert self.graph is not None
            graph = self.graph

        pos_idx, _ = self.find_pos_and_neg_idx(graph)

        components = [c for c in sorted(nx.connected_components(graph.to_undirected()), key=len, reverse=True) if len(c) > 1]
        pos_in_component = [[pos for pos in pos_idx if pos in component] for component in components]

        connected = {i: [component, pos] for i, (component, pos) in enumerate(zip(components, pos_in_component))}

        isolates = list(nx.isolates(graph))
        isolated_pos = [pos for pos in pos_idx if pos in isolates]

        disconnected = [isolates, isolated_pos]

        return connected, disconnected

    def check_homophily(self, graph=None, fig=None, ax=None, confusion=None):

        confusion = self.get_homophily(graph) if confusion is None else confusion
        fig, ax = plot_homophily(confusion, fig=fig, ax=ax)
        return fig, ax

    def get_homophily(self, graph) -> pd.DataFrame:
        if graph is None:
            assert self.graph is not None
            graph = self.graph

        pos_idx, neg_idx = self.find_pos_and_neg_idx(graph)

        adj_label_df = pd.DataFrame(data=np.empty((2, 2)), columns=["Neighborhood Unlabeled", "Neighborhood Positives"], index=["Self Unlabeled", "Self Positive"])

        for label, outer_name in zip([1, 0], ["Positives", "Unlabeled"]):
            for indices, name in zip([pos_idx, neg_idx], ["Positive", "Unlabeled"]):
                num_pos = [np.sum([graph.nodes[n]["y"] == label for n in graph.neighbors(
                    index)]) for index in indices]
                total_n = [len([graph.nodes[n]["y"] == label for n in graph.neighbors(
                    index)]) for index in indices]
                frac_pos = [n / total for n,
                            total in zip(num_pos, total_n) if total > 0]
                frac_pos = np.mean(frac_pos)
                num_pos = np.mean(num_pos)

                adj_label_df.loc["Self {}".format(name), "Neighborhood {}".format(outer_name)] = frac_pos

        return adj_label_df

    def find_pos_and_neg_idx(self, graph, label_key="y"):
        pos_idx = []
        neg_idx = []

        for node in graph.nodes(data=True):
            if node[1][label_key] == 1:
                pos_idx.append(node[0])
            else:
                neg_idx.append(node[0])

        return pos_idx, neg_idx

    def check_degrees(self, graph=None, fig=None, ax=None, density=True, degrees=None, direction="in"):
        if graph is None:
            assert self.graph is not None
            graph = self.graph

        if self.check_is_directed(graph):
            degrees = self.get_degrees(direction=direction) if degrees is None else degrees
        else:
            degrees = self.get_degrees() if degrees is None else degrees
        fig, ax = plot_degrees(degrees, fig=fig, ax=ax, density=density)
        return fig, ax

    def get_degrees(self, graph=None, direction="both") -> dict:
        """Calculates the degrees per node and returns degrees of positives and negatives seperately in a dict.
           Direction "both" counts either incoming or outgoing edges, but counts only one if both are present
           Direction "in" and "out" respectively only count incoming or outgoing edges.

           Returns:
            {"Positives": [5,4,3,...],
             "Unlabeled": [1,2,3,4]}
            """
        if graph is None:
            assert self.graph is not None
            graph = self.graph

        assert direction in ["both", "in", "out"], 'Direction has to be either "both", "in" or "out"'

        pos_idx, neg_idx = self.find_pos_and_neg_idx(graph)

        degrees = {}

        for indices, name in zip([pos_idx, neg_idx], ["Positive", "Unlabeled"]):
            if direction == "both":
                degrees.update({name: [node[1] / 2 for node in graph.degree(indices)]})
            elif direction == "in":
                degrees.update({name: [node[1] for node in graph.in_degree(indices)]})
            elif direction == "out":
                degrees.update({name: [node[1] for node in graph.out_degree(indices)]})

        return degrees

    def check_is_directed(self, graph, num_proofs=100):
        """ Checks if the graph is directed. Returns False if the graph is not of type DiGraph or MultiDiGraph.
            If it is of DiGraph or MultiDiGraph, draws num_proofs edges randomly and checks if the graph contains inverse edges.
            If all drawn edges have inverses, returns False, else returns true """

        import random
        if graph is None:
            assert self.graph is not None
            graph = self.graph

        if isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
            edges = list(graph.edges)

            for _ in range(num_proofs):
                edge = edges[random.randint(0, len(edges)-1)]
                if graph.has_edge(edge[1], edge[0]):
                    continue
                else:
                    return True

        return False

    def check_paths(self, graph=None, fig=None, ax=None, symmetrize=False, ppm=None):
        if ppm is None:
            ppm, _ = self.positive_paths_matrix(graph, symmetrize=symmetrize)
        fig, ax = plot_paths(ppm, fig=fig, ax=ax, symmetrize=symmetrize)
        return fig, ax

    def positive_paths_matrix(self, graph=None, symmetrize=False):
        if graph is None:
            assert self.graph is not None
            graph = self.graph

        pos_idx, neg_idx = self.find_pos_and_neg_idx(graph)

        ppm = np.zeros((len(pos_idx), len(pos_idx)), dtype=np.uint8)
        nodes_to_check = pos_idx[:]

        if symmetrize:
            graph = graph.to_undirected()

        for pos_node in pos_idx:
            if graph.degree(pos_node) == 0:
                # all nodes that are isolated don't have to be included
                nodes_to_check.remove(pos_node)

        for i, pos_node_A in enumerate(pos_idx):
            for j, pos_node_B in enumerate(pos_idx):
                # don't include self-loops
                if pos_node_A == pos_node_B:
                    continue
                if pos_node_A in nodes_to_check and pos_node_B in nodes_to_check:
                    try:
                        ppm[i, j] = nx.shortest_path_length(graph, source=pos_node_A, target=pos_node_B)
                    except nx.exception.NetworkXNoPath:
                        continue

        ppm_idx_2_real_idx = {i: pos_idx[i] for i in range(ppm.shape[0])}

        return ppm, ppm_idx_2_real_idx
