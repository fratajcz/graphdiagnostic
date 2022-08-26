# graphdiagnostic
Diagnostic plotting to help you figure out what is going on in your graph


# Getting Started

Graphdiagnostic has a short list of requirements which can be found in `requirements.txt`. 
To get started, just type the following lines (assuming you have conda installes):

```
git clone https://github.com/fratajcz/graphdiagnostic.git
cd graphdiagnostic
conda create -n gd python=3.7
conda activate gd
python3 -m pip install -r requirements.txt
```

And you are good to go :)

# Minimal Working Example

The diagnostic takes as input a NetworkX graph object.
Each node has to have a property "y" which has to be set to 0 or 1 for each node.

You can plot your diagnostics as follows: (see mwe.py)

```
from graphdiagnostic import GraphDiagnostic

import networkx as nx
import matplotlib.pyplot as plt
import random

# build the graph (replace this with your graph)
graph = nx.scale_free_graph(1000).to_undirected()

# add some extra isolated nodes for fun
graph.add_nodes_from(list(range(1000, 1400)))

# label nodes with binary labels
for node in graph.nodes(data=True):
    if random.randint(0, 9) == 0:
        node[1]["y"] = 1
    else:
        node[1]["y"] = 0

# initialize diagnostic with your graph
diagnostic = GraphDiagnostic(graph)
# plot all details (components, paths, degrees, homophily)
fig, ax = diagnostic.get_diagnostics()
# add title and save figure
plt.suptitle("Scale Free Graph")
plt.savefig("img/scalefree_diagnostics.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_diagnostics.png?raw=true)

# Individual Details 

To plot only one detail at a time, just specify it when you call `diagnostic.get_diagnostics(<detail>)`:

## Plot only Components
```
fig, ax = diagnostic.get_diagnostics("components")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_components.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_components.png?raw=true)

### Explanation

This plot shows you two things:
1. Is you graph split up into multiple disconnected components and are there isolated nodes (left bar).
2. How are your labels distributed among those components (right bar).


## Plot only Paths
```
fig, ax = diagnostic.get_diagnostics("paths")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_paths.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_paths.png?raw=true)

### Explanation

This plot shows you how far your positive labeled nodes are away from each other.

The black bar to the very right shows the Unreachable (Unr.) positive nodes. These nodes show up in the "Isolated" part in the previous plot.

The other bars are sorted by their path length, meaning in this case, the first non-black bar means that this number of positives has at least one other positive in their 1-hop neighborhood. The bar to the right means the same for the 2-hop neighborhood and so on. 

The height of the bars means the number of positive center nodes while the color codes the number of positive nodes in the respective n-hop neighborhood of the center node.

## Plot only Degrees
```
fig, ax = diagnostic.get_diagnostics("degrees")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_degrees.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_degrees.png?raw=true)

### Explanation

This plot simply looks at the degree distribution of your nodes, split up by their labels. It might show you that e.g. you Positive nodes are hub-nodes because they have a higher average degree than the others, or that one of the label groups is scale-free (roughly along a diagonal line) while the other isnt.

## Plot only Homophily
```
fig, ax = diagnostic.get_diagnostics("homophily")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_homophily.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_homophily.png?raw=true)

### Explanation

This plot is very important if you want to apply a Graph Neural Network (GNN) for node classification on the graph. If there are high numbers on the main diagonal of the matrix, then nodes that share the same label are connected to each other more than to nodes with a different label (Homophily, as found in social networks). If there are high values on the anti diagonal, then nodes are predominantly connected to nodes with the opposite label (Heterophily, as found on dating sites). 

This plot is inspired by Lim et al. and their paper [Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods](https://arxiv.org/abs/2110.14446). Since we introduced our labels randomly, they are independent of the graph structure, which likely is the worst case scenario for GNNs.


## plot only metrics
```
fig, ax = diagnostic.get_diagnostics("metrics")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_metrics.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_metrics.png?raw=true)

### Explanation

This plot is actually a Table and contains some additional metrics which are not really related to the other topics but still might be relevant. More metrics might be added in the future. If you have an Idea for an important metric, do not hesitate to open an Issue here on Github!
