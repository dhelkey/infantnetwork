import pandas as pd
import networkx as nx
import igraph as ig
import numpy as np

def compute_network(df_in: pd.DataFrame,
    from_var = 'prevhospid',
    to_var = 'hospid',
    id_var = None,
    edge_cutoff: int = 1,
    random_walk_steps: int = 5):

    df = df_in.copy()

    # Validate and preprocess input data
    variables_to_check = [from_var, to_var] + ([id_var] if id_var else [])
    for var in variables_to_check:
        if df[var].isnull().any():
            raise ValueError(f"Missing values detected in '{var}'.")
        df[var] = df[var].astype(str)

    # Prepare nodes and edges
    all_nodes = set(df[from_var]).union(df[to_var])
    n_nodes = len(all_nodes)
    n_individuals = df[id_var].nunique() if id_var else len(df)
    n_transfers = len(df)

    # Compute edge weights and apply cutoff
    edge_df = df.groupby([from_var, to_var]).size().reset_index(name='edge_weight')
    edge_df = edge_df[edge_df['edge_weight'] >= edge_cutoff]

    # Build NetworkX graphs
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_weighted_edges_from(edge_df[[from_var, to_var, 'edge_weight']].values)
    G_undirected = G.to_undirected()

    # Build iGraph graphs
    g = ig.Graph(directed=True)
    g.add_vertices(list(all_nodes))
    edges = list(zip(edge_df[from_var], edge_df[to_var]))
    g.add_edges(edges)
    g.es['weight'] = edge_df['edge_weight'].tolist()
    g_undirected = g.as_undirected()

    # Analyze graph components
    connected_components = list(nx.connected_components(G_undirected))
    num_connected_components = len(connected_components)
    total_graph_weight = np.max(G_undirected.size(weight='weight'), 1)

    def component_weight(component_nodes: Set[str]) -> float:
        subgraph = G_undirected.subgraph(component_nodes)
        return subgraph.size(weight='weight')

    component_weights = {frozenset(comp): component_weight(comp) for comp in connected_components}
    largest_component = max(component_weights, key=component_weights.get, default=set())
    largest_component_weight = component_weights.get(largest_component, 0)
    max_weight_percentage_by_component = (largest_component_weight / total_graph_weight) * 100

    # Create components DataFrame
    components_data = [
        {
            'Component_Num': idx + 1,
            'Component': comp,
            'Node_Percentage': (len(comp) / n_nodes) * 100,
            'Weight_Percentage': (component_weights[frozenset(comp)] / total_graph_weight) * 100
        }
        for idx, comp in enumerate(connected_components)
    ]
    df_components = pd.DataFrame(components_data).sort_values(by='Weight_Percentage', ascending=False)

    # Compute network metrics
    efficiency_global = nx.global_efficiency(G_undirected)
    local_efficiencies = [
        nx.global_efficiency(G_undirected.subgraph(G_undirected.neighbors(node)))
        if len(list(G_undirected.neighbors(node))) > 1 else 0
        for node in G_undirected.nodes()
    ]
    efficiency_median_local = np.median(local_efficiencies) if local_efficiencies else 0
    density_unweighted = nx.density(G)
    total_edge_weight = sum(weight for _, _, weight in G.edges(data='weight', default=1))
    max_possible_weight = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density_weighted = total_edge_weight / max_possible_weight

    katz_centralities = nx.katz_centrality_numpy(G_undirected, weight='weight')
    centrality_values = list(katz_centralities.values())
    centrality_median = np.median(centrality_values)
    in_degrees = G.in_degree()
    receiving_nodes = [node for node, degree in in_degrees if degree > 0]
    centrality_mean_receiving = np.mean([katz_centralities[node] for node in receiving_nodes]) if receiving_nodes else 0

    communities = list(nx.algorithms.community.greedy_modularity_communities(G_undirected, weight='weight'))
    modularity_greedy = nx.algorithms.community.modularity(G_undirected, communities, weight='weight')

    if g_undirected.vcount() > 0:
        largest_igraph_component = g_undirected.clusters().giant()
        walktrap = largest_igraph_component.community_walktrap(steps=random_walk_steps, weights='weight')
        clusters = walktrap.as_clustering()
        modularity_randomwalk = clusters.modularity
    else:
        modularity_randomwalk = 0

    network_metrics = {
        'n_individuals': n_individuals,
        'n_nodes': n_nodes,
        'n_transfers': n_transfers,
        'n_self_loops': nx.number_of_selfloops(G),
        'centrality_mean_receiving': centrality_mean_receiving,
        'centrality_median': centrality_median,
        'density_unweighted': density_unweighted,
        'density_weighted': density_weighted,
        'efficiency_global': efficiency_global,
        'efficiency_median_local': efficiency_median_local,
        'modularity_greedy': modularity_greedy,
        'modularity_randomwalk': modularity_randomwalk,
        'num_connected_components': num_connected_components,
        'max_node_percentage_by_component': (len(largest_component) / n_nodes) * 100 if n_nodes else 0,
        'max_weight_percentage_by_component': max_weight_percentage_by_component
    }

    return {
        'metrics': network_metrics,
        'df_components': df_components,
        'graph_networkx': G,
        'graph_igraph': g
    }