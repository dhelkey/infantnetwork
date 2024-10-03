import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
def computeNetwork(df_in, from_var='prevhospid', to_var='hospid', 
                        edge_cutoff=1, 
                        random_walk_steps = 5):
    """
    Construct network metrics and networks from infant transfer records.
    Designed to construct networks with nodes representing hospitals and
	    directed edges representing the number of acute transfers between hospitals,
         idenfified using infant transfer records. 
    Transfer records should be formated such that all rows corrospond to an infant transfer,
      with from_var and to_var identifying the transfer origin and destination respecfivly.  

    Parameters:
        df_in (DataFrame): Transfer DataFrame (One row per infant transfer).
        from_var (str): Hospital transfered from.
        to_var (str): Hospital transfered to.
        edge_cutoff (int): Minimum number of transfers to include edge.
        random_walk_steps (int): Randdom walk steps for random walk modularity    
        
    Returns:
        dict: A dictionary with the following items:
            'metrics' (DataFrame): A DataFrame containing network metrics.
            'df_components' (DataFrame): A DataFrame of network components.
            'graph_networkx' (networkx.Graph): A NetworkX graph object representing the network.
            'graph_igraph' (igraph.Graph): An iGraph graph object representing the network.

    Supports:
    Kunz, S.N., Helkey, D., Zitnik, M., Phibbs, C.S., Rigdon, J., Zupancic, J.A. and Profit, J., 2021. 
    Quantifying the variation in neonatal transport referral patterns 
    using network analysis. Journal of Perinatology, 41(12), pp.2795-2803.

    Code written by:
    Marinka Zitnik, Ph.D., Assistant Professor
    Email: marinka@hms.harvard.edu

    Compiled by:
    Daniel Helkey, MS
    Email: dhelkey@stanford.edu
    """
    df = df_in.copy()

    #Ensure all identifier variables are strings
    for var in [from_var, to_var] :
        if df[var].isnull().any():
            raise ValueError(f"Missing values in {var}.")
        df[var] = df[var].astype(str)  

    #Identify number of nodes and individuals
    all_nodes = set(df[from_var]).union(set(df[to_var]))
    n_nodes_all = len(all_nodes)

    # Observed edges
    edge_df = df.groupby([from_var, to_var]).size().reset_index(name='edge_weight')
    n_edges_all = len(edge_df)
    n_transfers_all = edge_df.edge_weight.sum()

    #If edge_cutoff > 1, remove edges smaller than edge_cutoff
    edge_df = edge_df[edge_df['edge_weight'] >= edge_cutoff]

    n_edges = len(edge_df)
    n_transfers = edge_df.edge_weight.sum()
    included_nodes = set(edge_df[from_var])\
            .union(edge_df[to_var])
    n_nodes = len(included_nodes)

    #  NetworkX representations 
    #G - directed networkX graph
    G = nx.DiGraph()
    G.add_nodes_from(included_nodes)
    G.add_weighted_edges_from(edge_df[[from_var, to_var, 'edge_weight']].values)
    #G_undirected - undirected networkX graph
    G_undirected = G.to_undirected()

    # iGraph representations 
    #g - Directed Igraph 
    g = ig.Graph(directed=True)
    g.add_vertices(sorted(set(edge_df[from_var]).union(edge_df[to_var])))
    edges = [(row[from_var], row[to_var]) for _, row in edge_df.iterrows()]
    weights = edge_df['edge_weight'].tolist()
    
    # Add edges with weights
    g.add_edges(edges)
    g.es['weight'] = weights
  
    #g_undirected - Undirected Igraph 
    g_undirected = g.as_undirected() 

    #Identify self loops
    n_self_loops = nx.number_of_selfloops(G)

    # Number of Connected Components
    num_connected_components = nx.number_connected_components(G_undirected)
    connected_components_list = list(nx.connected_components(G_undirected))

    # Nodes within Connected Components
    largest_connected_component_nodes = max(connected_components_list, key=len) if \
                                            num_connected_components > 0 else set()
    max_node_percentage_by_component = len(largest_connected_component_nodes) / len(G_undirected) * 100  

    # Edge Weights within Connected Components 
    edge_weights = {(min(u, v), max(u, v)): data['weight'] for u, v, data in G_undirected.edges(data=True)}
    total_graph_weight = sum(edge_weights.values())

    def component_weight(component):
        """Calculate sum of component edge weights
        """
        return sum(edge_weights.get((min(u, v), max(u, v)), 0) for \
                    u, v in nx.edges(G_undirected.subgraph(component)))

    # Largest Weighted Component
    if num_connected_components > 1:
        largest_weighted_component = max(connected_components_list,
                                          key=component_weight)
        largest_component_weight = component_weight(largest_weighted_component)
    else:
        # Calculate the weight for the single component
        largest_component_weight = component_weight(connected_components_list[0]) if \
                                    connected_components_list else 0

    # Calculate the percentage
    max_weight_percentage_by_component = largest_component_weight / \
                                             total_graph_weight * 100

    #Sumary dataframe of connected components
    components = list(nx.connected_components(G_undirected))
    df_components = pd.DataFrame([
        {
            'Component_Num': i + 1,  # Number components starting from 1
            'Component': component,
            'Node_Percentage': len(component) / len(G_undirected) * 100,
            'Weight_Percentage': component_weight(component) / total_graph_weight * 100
        }
        for i, component in enumerate(components)
    ])
    df_components.sort_values(by='Weight_Percentage', ascending=False, inplace=True)

    #Network metrics
    
    # Efficiency
    ## Global eficiency
    efficiency_global = nx.global_efficiency(G_undirected)
    
    ## Median local node efficiency
    local_efficiencies = []
    for node in G_undirected.nodes():
        # Subgraph induced by node's neighbors 
        neighbors = list(nx.neighbors(G_undirected, node))
        if len(neighbors) < 2:
            # Nodes with fewer than two neighbors have local efficiency = 0
            local_efficiencies.append(0)
            continue

        subgraph = G_undirected.subgraph(neighbors)
        local_efficiency = nx.global_efficiency(subgraph)
        local_efficiencies.append(local_efficiency)
    efficiency_median_local = np.median(local_efficiencies)
    
    # Density 
    ## Unweighted density
    density_unweighted = nx.density(G)
    ## Weighted density
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())
    density_weighted = np.sum(edge_weights) / (len(edge_weights) * np.max(edge_weights)) if \
                                                     edge_weights else 0

    ## Centrality
    katz_centralities = nx.katz_centrality_numpy(G_undirected)
    in_degrees = dict(G.in_degree())
    ## Median node centrality 
    centrality_median = np.median(list(katz_centralities.values()))
    ## Mean centrality of nodes with incoming transfers
    centrality_mean_receiving = np.mean([katz_centralities[node] for \
            node, count in in_degrees.items() if count > 0])
  
    ## Modularity
    #Greedy modularity
    modularity_greedy = nx.algorithms.community.modularity(G_undirected, 
        nx.algorithms.community.greedy_modularity_communities(G_undirected))
    
    #Random walk modularity
    giant_component = g.components(mode='weak').giant()
    random_walk = giant_component.community_walktrap(steps=random_walk_steps,
                     weights=giant_component.es['weight']).as_clustering()
    modularity_randomwalk = giant_component.modularity(random_walk.membership)

    network_metrics = {
        'n_nodes_all': n_nodes_all,
        'n_nodes': n_nodes,
        'n_edges_all':n_edges_all,
        'n_edges':n_edges,
        'n_transfer_all':n_transfers_all,
        'n_transfers':n_transfers,
        'n_self_loops':n_self_loops,
        'centrality_mean_receiving': centrality_mean_receiving,
        'centrality_median': centrality_median,       
        'density_unweighted': density_unweighted,
        'density_weighted': density_weighted, 
        'efficiency_global': efficiency_global,
        'efficiency_median_local':efficiency_median_local, 
        'modularity_randomwalk': modularity_randomwalk, 
        'modularity_greedy':modularity_greedy,
        'num_connected_components':num_connected_components,
        'max_node_percentage_by_component': max_node_percentage_by_component,
        'max_weight_percentage_by_component':max_weight_percentage_by_component 
    }
    return {
        'metrics':network_metrics,
        'df_components':df_components,
        'graph_networkx':G,
        'graph_network_igraph':g

    }