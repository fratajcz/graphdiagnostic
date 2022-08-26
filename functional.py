from collections import Counter

import matplotlib.pyplot as plt
import seaborn
import numpy as np
from scipy import sparse
import networkx as nx
from matplotlib import cm
from numpy import linspace


def plot_metrics(metrics, ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    expand_to = 45

    cell_text = []
    for key, value in metrics.items():
        if len(value) == 2:
            text = "{:.1f} +- {:.1f}".format(value[0], value[1])
        else:
            text = "{:.1f}".format(value[0])
        missing_len = expand_to - (len(key) + len(text))
        filler_string = " " * missing_len
        text = "{}:{}{}".format(key, filler_string, text)
        cell_text.append([text])

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis("off")

    the_table = plt.table(cellText=cell_text,
                          loc='center')

    #plt.subplots_adjust(bottom=0.9)
    fig.tight_layout()
    return fig, ax
    

def plot_homophily(confusion, ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    seaborn.heatmap(confusion, ax=ax, vmin=0, vmax=1, annot=True, xticklabels=True, yticklabels=True, linewidths=.5, cmap="Blues")

    return fig, ax


def plot_degrees(degrees, ax=None, fig=None, density=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_axisbelow(True)
    ax.grid()

    degreedict = {}

    for name in ["Unlabeled", "Positive"]:
        degree = degrees[name]
        degree_counter = Counter(degree)
        y = list(degree_counter.values())
        if density:
            y_sum = np.sum(y)
            y = [y_value / y_sum for y_value in y]
        x = degree_counter.keys()
        ax.scatter(x, y, alpha=0.5, label=name)
        mean_x = np.mean(np.asarray(degree)[np.asarray(degree) > 0])
        ax.axline((mean_x, 1), (mean_x, 0), label="Mean " + name[0] + ".", color="blue" if name == "Unlabeled" else "orange")
        degreedict.update({name: degree})

    if density:
        ax.set_ylabel("Density")
        ax.set_ylim(1e-5, 1)
        ypos1 = 0.5
        ypos2 = 0.2
    else:
        ax.set_ylabel("Count")
        ax.set_ylim(1, 1e4)
        ypos1 = 6000
        ypos2 = 4000

    for name, degree in degreedict.items():
        if name == "Unlabeled":
            ax.text(1, ypos1, name[0] + "n:" + str(len(degree)))
        elif name == "Positive":
            ax.text(1, ypos2, name[0] + "n:" + str(len(degree)))
        if name == "Unlabeled":
            ax.text(7, ypos1, name[0] + "0:" + str(np.sum([np.asarray(degree) == 0])))
        elif name == "Positive":
            ax.text(7, ypos2, name[0] + "0:" + str(np.sum([np.asarray(degree) == 0])))

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Degree")

    ax.legend()

    return fig, ax


def plot_paths(ppm, fig=None, ax=None, symmetrize=False):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    G = nx.from_scipy_sparse_matrix(sparse.coo_matrix(ppm), parallel_edges=False, create_using=nx.DiGraph, edge_attribute='weight')
    G.remove_nodes_from(list(nx.isolates(G)))
    """
    pos = nx.circular_layout(G)
    nx.draw(G, pos)
    print(G.nodes())
    node_labels = {id: preprocessor.id2hgnc[ppm_idx_2_real_idx[id]] for id in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.savefig('this.png')
    """

    neighborhood_dict = {}
    path_dict = {}
    for i in range(1, np.max(ppm) + 1):
        binary_ppm = ppm == i
        has_ith_order_neighbors = np.nonzero(np.sum(binary_ppm, axis=1))[0]
        i_th_order_neighborhood = {center: np.nonzero(binary_ppm[center, :])[0] for center in has_ith_order_neighbors}
        neighborhood_length_dict = dict(Counter([neighborhood.shape[0] for neighborhood in i_th_order_neighborhood.values()]))
        for key, value in neighborhood_length_dict.items():
            if key not in neighborhood_dict:
                neighborhood_dict.update({key: {i: value}})
            else:
                neighborhood_dict[key].update({i: value})
        path_dict.update({i: len(has_ith_order_neighbors)})
    zeros = np.sum(np.sum(ppm, axis=1) == 0)

    ax.set_axisbelow(True)
    ax.grid()
    ax.set_ylabel("Number of Nodes with Positives in Neighborhood")
    ax.set_xlabel("Path Length")
    try:
        max_x = np.max(list(path_dict.keys()))
    except ValueError:
        max_x = 0
    ax.set_xlim(-0.5, max_x + 0.5)
    ax.set_xticks(list(range(max_x + 1)))

    ax.set_xticklabels(["Unr."] + list(range(1, max_x + 1)))

    super_y_offset = 0.2
    if max_x == 0 or len(neighborhood_dict.keys()) == 0:
        ax.text(0, super_y_offset, str(zeros), ha='center',
                weight='bold', size=7)
        return ax

    bottom = np.zeros(np.max(ppm) + 1)

    start = 0.0
    stop = 1.0
    number_of_lines = np.max(list(neighborhood_dict.keys()))
    cm_subsection = linspace(start, stop, number_of_lines)

    colors = [cm.viridis(x) for x in cm_subsection]

    for neighborhood_size in range(1, len(neighborhood_dict.keys()) + 1):
        try:
            path_length_dict = neighborhood_dict[neighborhood_size]
        except KeyError:
            continue

        ax.bar(path_length_dict.keys(), height=path_length_dict.values(), bottom=bottom[np.asarray(list(path_length_dict.keys()))],
               label=neighborhood_size, color=colors[neighborhood_size - 1])

        for i, value in path_length_dict.items():
            bottom[i] += value

    # put labels on bar
    for i, total in enumerate(bottom):
        if i == 0:
            ax.text(0, super_y_offset, str(zeros), ha='center',
                    weight='bold', size=7)
        else:
            ax.text(i, total + super_y_offset, int(total), ha='center',
                    weight='bold', size=7)

    # put subbar labels
    """
    y_offset = -0.7
    for bar in ax.patches:
        try:
            ax.text(
                # Put the text in the middle of each bar. get_x returns the start
                # so we add half the width to get to the middle.
                bar.get_x() + bar.get_width() / 2,
                # Vertically, add the height of the bar to the start of the bar,
                # along with the offset.
                bar.get_height() + bar.get_y() + y_offset,
                # This is actual value we'll show.
                round(bar.get_height()),
                # Center the labels and style them a bit.
                ha='center',
                color='w',
                weight='bold',
                size=7
            )
        except AttributeError:
            continue
    """
    legend_title = '# Nodes\nReachable'
    ax.legend(title=legend_title)
    if len(neighborhood_dict.keys()) > 11:
        ax.get_legend().remove()
        max_colors = 20  # plot only so many distinct colors, above that use continuous value
        import matplotlib as mpl
        cmap = mpl.cm.viridis

        if len(neighborhood_dict.keys()) > max_colors:
            num_bounds = max_colors
            mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(1, len(neighborhood_dict.keys())), cmap=cmap)
            stride_width = int(len(neighborhood_dict.keys()) / max_colors)
            cbar = fig.colorbar(mappable,
                                ax=ax,
                                label=legend_title,
                                ticks=range(1, len(neighborhood_dict.keys()), stride_width),
                                pad=0.02)
        else:
            num_bounds = len(neighborhood_dict.keys())
            bounds = np.asarray(range(0, num_bounds)) + 0.5

            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                                ax=ax,
                                label=legend_title,
                                ticks=range(1, num_bounds),
                                pad=0.02)

    return fig, ax


def plot_disconnected_components(components, isolates, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # summarize tiny components into one
    cutoff = 0.01
    total_nodes = np.sum([len(component[0]) for component in components.values()])
    tiny_components = [[], []]
    components_to_delete = []
    for i, component in components.items():
        if len(component[0]) / total_nodes < cutoff:
            tiny_components[0] += component[0]
            tiny_components[1] += component[1]
            components_to_delete.append(i)

    for i in components_to_delete:
        del components[i]

    total_nodes_after = np.sum([len(component[0]) for component in components.values()] + [len(tiny_components[0])])
    assert total_nodes == total_nodes_after, "before: {}, after: {}".format(total_nodes, total_nodes_after)

    outer = [len(value[0]) for value in components.values()] + [len(tiny_components[0])] + [len(isolates[0])]
    labels = ["Component {}".format(key) for key in components.keys()] + ["Tiny Components"] + ["Disconnected"]

    start = 0.0
    stop = 1.0
    number_of_lines = len(outer) + 1
    cm_subsection = np.linspace(start, stop, number_of_lines)

    outer_colors = [cm.viridis(x) for x in cm_subsection[:-1]]
    inner_colors = ["tab:grey", "tab:orange"] * len(outer)

    inner = [[len(set(value[0]) - set(value[1])), len(value[1])] for value in components.values()] \
        + [[len(set(tiny_components[0]) - set(tiny_components[1])), len(tiny_components[1])]] \
        + [[len(set(isolates[0]) - set(isolates[1])), len(isolates[1])]]

    inner = [element for sublist in inner for element in sublist]

    bottom = np.zeros(2)

    for j in range(len(outer)):
        ax.bar(x=0, height=outer[j], bottom=bottom[0], label=labels[j], color=outer_colors[j], width=0.95)
        bottom[0] += outer[j]

    for i in range(len(inner)):
        ax.bar(x=1, height=inner[i], bottom=bottom[1], label="Unknown" if i % 2 == 0 else "Positive", color=inner_colors[i], width=0.95)
        bottom[1] += inner[i]

    current_handles, current_labels = ax.get_legend_handles_labels()

    # sort or reorder the labels and handles
    capped_handles = current_handles[:len(outer) + 2]
    capped_labels = current_labels[:len(outer) + 2]

    # Shrink current axis by 60%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height])

    # Put a legend to the right of the current axis
    ax.legend(capped_handles, capped_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_ylabel("Number of Nodes")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Components", "Pos/Unk"])

    for bar in ax.patches:
        y_offset = -500
        try:
            if bar.get_height() > np.abs(y_offset):
                ax.text(
                    # Put the text in the middle of each bar. get_x returns the start
                    # so we add half the width to get to the middle.
                    (bar.get_x() + bar.get_width() / 2),
                    # Vertically, add the height of the bar to the start of the bar,
                    # along with the offset.
                    bar.get_height() + bar.get_y() + y_offset,
                    # This is actual value we'll show.
                    round(bar.get_height()),
                    # Center the labels and style them a bit.
                    ha='center',
                    va="center",
                    color='w',
                    weight='bold',
                    size=7
                )
        except AttributeError:
            continue

    return fig, ax
