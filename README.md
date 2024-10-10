# infantnetwork

Python package for constructing networks from infant transfer records. Functions and example code is provided to construct networks and describe their shape and structure.


Code is released under the open-source [MIT License](LICENSE).



### Usage

``` 
from infantnetwork import computeNetwork, sample_transfers

#Compute network using default settings
output = computeNetwork(sample_transfers)

#Extract graphs
graph_networkx = output['graph_networkx']
graph_igraph = output['graph_igraph']

#Metrics
network_metrics = output['metrics']

#Extract a single network metric
network_metrics['centrality_mean_receiving']

```

The network_metrics atribute of computeNetwork returns the following quantatitve network metrics:

| Metric                            | Description                                                                                      |
|------------------------------------|--------------------------------------------------------------------------------------------------|
| `n_nodes`                         | Number of nodes in the network.                                                                  |
| `n_edges`                     | Total number of edges in the network.                                                |
| `n_transfers`                     | Total number of transfers in the network.                                                |
| `n_self_loops`                    | Number of self-loops.                                   |
| `centrality_median`               | Median node Katz centrality.                     |
| `centrality_mean_receiving`        | Mean node Katz centrality of nodes with incoming transfers.        |
| `density_unweighted`              | Network density (total proportion of possible edges observed)                                         |
| `density_weighted`                | Weighted network density: $$\frac{\sum \text{edge\_weights}}{\text{n}\_ \text{edges} \cdot \max(\text{edge\_weights})}$$ |
| `efficiency_global`               | Global network efficiency.                 |
| `efficiency_median_local`         | Median of local node efficiencies.          |
| `modularity_greedy`               | Network modularity using the greedy community modularity algorithm.      |
| `modularity_randomwalk`           | Network modularity using random walktrap algorithm             |
| `max_node_percentage_by_component`| Percentage of nodes in the largest connected component.                                   |
| `max_weight_percentage_by_component`| Percentage of weight (transfers) within the largest connected component.                  |

