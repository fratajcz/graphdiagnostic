# graphdiagnostic
Diagnostic plotting to help you figure out what is going on in your graph

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

## plot only components
```
fig, ax = diagnostic.get_diagnostics("components")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_components.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_components.png?raw=true)


## plot only paths
```
fig, ax = diagnostic.get_diagnostics("paths")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_paths.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_paths.png?raw=true)

## plot only degrees
```
fig, ax = diagnostic.get_diagnostics("degrees")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_degrees.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_degrees.png?raw=true)


## plot only homophily
```
fig, ax = diagnostic.get_diagnostics("homophily")

# add title and save figure
plt.suptitle("Scale Free Graph")
fig.tight_layout()
plt.savefig("img/scalefree_homophily.png")
```

This creates the following image in the subfolder `img`:

![](img/scalefree_homophily.png?raw=true)


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
