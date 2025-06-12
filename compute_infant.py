import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import networkx as nx
import igraph as ig


def process_edges(df_in: pd.DataFrame, 
                  from_var: str = 'prevhospid', 
                  to_var: str = 'hospid',
                  id_var: str =  None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes transfer records into edge lists with column names 'from', 'to', and 'weight'.

    Args:
        df_in (pd.DataFrame): DataFrame containing transfer records.
        from_var (str): Column for source hospital identifier.
        to_var (str): Column for destination hospital identifier.
        id_var (str): (Optional) Column for infant identifier

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - edge_df: Dataframe of edges and weights.

    Raises:
        ValueError: If input columns contain missing values.
        TypeError: If columns cannot be converted to strings.
    """
    df = df_in.copy()

    # Ensure no missing values
    for col in [from_var, to_var]:
        if df[col].isnull().any():
            raise ValueError(f"Column '{col}' contains NaNs. Remove missing values.")
        try:
            df[col] = df[col].astype(str)
        except Exception as e:
            raise TypeError(f"Failed to convert column '{col}' to string: {e}")

    # Group by original columns to get counts
    edge_df = (
        df.groupby([from_var, to_var])
        .size()
        .reset_index(name='weight')
        .rename(columns={from_var: 'from', to_var: 'to'})
    )
    return edge_df

def calculate_standard_metrics(edge_df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculates standard network metrics from a standardized edge DataFrame.

    Assumes input DataFrame has columns: 'from', 'to', and 'edge_weight'.

    Args:
        edge_df (pd.DataFrame): DataFrame representing network edges.

    Returns:
        Dict[str, int]: Dictionary containing:
            - 'n_edges': Number of edges.
            - 'n_transfers': Total transfers (sum of edge weights).
            - 'n_self_loops': Count of self-loop edges ('from' == 'to').
            - 'n_nodes': Total number of unique nodes involved.
    """
    if edge_df.empty:
        return {
            'n_edges': 0,
            'weight_total': 0,
            'n_self_loops': 0,
            'n_nodes': 0
        }

    # Compute metrics
    n_edges = len(edge_df)
    weight_total = edge_df['weight'].sum()
    n_self_loops = (edge_df['from'] == edge_df['to']).sum()

    # Compute unique nodes
    unique_nodes = set(edge_df['from']).union(edge_df['to'])
    n_nodes = len(unique_nodes)

    return {
        'n_edges': n_edges,
        'weight_total': weight_total,
        'n_self_loops': n_self_loops,
        'n_nodes': n_nodes
    }

def edge_df_check(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate an edge DataFrame for graph construction.

    Args:
        edge_df (pd.DataFrame): DataFrame expected to have columns ['from', 'to', 'weight'].

    Returns:
        pd.DataFrame: (self) The same DataFrame if checks pass.

    Raises:
        ValueError: If required columns are missing or if any 'from'/'to' values are NaN.
    """
    required = {'from', 'to', 'weight'}
    missing = required - set(edge_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in edge_df: {missing}")
    if edge_df[['from', 'to']].isnull().any().any():
        raise ValueError("Columns 'from' and 'to' must not contain NaN.")
    return edge_df

def create_networkx_graphs(edge_df: pd.DataFrame) -> Tuple[nx.Graph, nx.DiGraph]:
    """
    Creates directed and undirected NetworkX graphs from an edge DataFrame.

    Args:
        edge_df (pd.DataFrame): DataFrame with columns ['from', 'to', 'weight'].

    Returns:
        Tuple[nx.Graph, nx.DiGraph]:
            - nx_undirected: Undirected NetworkX graph.
            - nx_directed: Directed NetworkX graph.

    Raises:
        ValueError: If edge_df is invalid.
    """
    edge_df = edge_df_check(edge_df)

    # Identify all unique nodes
    all_nodes = set(edge_df['from']).union(edge_df['to'])

    # Create directed graph
    nx_directed = nx.DiGraph()
    nx_directed.add_nodes_from(all_nodes)
    nx_directed.add_weighted_edges_from(\
            edge_df[['from', 'to', 'weight']].values)

    # Create undirected graph
    nx_undirected = nx.Graph()
    nx_undirected.add_nodes_from(all_nodes)
    nx_undirected.add_weighted_edges_from(nx_directed.edges(data='weight'))

    return nx_undirected, nx_directed

def create_igraph_graphs(edge_df: pd.DataFrame) -> Tuple[ig.Graph, ig.Graph]:
    """
    Creates directed and undirected iGraph graphs from an edge DataFrame.

    Args:
        edge_df (pd.DataFrame): DataFrame with columns ['from', 'to', 'weight'].

    Returns:
        Tuple[ig.Graph, ig.Graph]:
            - ig_undirected: Undirected iGraph graph.
            - ig_directed: Directed iGraph graph.

    Raises:
        ValueError: If edge_df is invalid.
        RuntimeError: If iGraph encounters a problem building graphs.
    """
    edge_df = edge_df_check(edge_df)

    # Sorted unique nodes
    unique_nodes = sorted(set(edge_df['from']).union(edge_df['to']))

    # Directed iGraph
    ig_directed = ig.Graph(directed=True)
    ig_directed.add_vertices(unique_nodes)
    edge_tuples = list(zip(edge_df['from'], edge_df['to']))
    ig_directed.add_edges(edge_tuples)
    ig_directed.es['weight'] = edge_df['weight'].tolist()

    # Undirected iGraph
    ig_undirected = ig_directed.as_undirected(combine_edges=sum)
    return ig_undirected, ig_directed

def compute_node_metrics(
    edge_df: pd.DataFrame,
    nx_undirected: nx.Graph,
    nx_directed: nx.DiGraph,
    ig_undirected: ig.Graph,
    ig_directed: ig.Graph
) -> Dict[str, Dict[Any, float]]:
    """
    Calculates and validates node metrics using both NetworkX and iGraph.

    Args:
        edge_df (pd.DataFrame): DataFrame representing network 
            edges with columns 'to', 'from', and 'weight'
        nx_undirected (nx.Graph): Undirected NetworkX graph.
        nx_directed (nx.DiGraph): Directed NetworkX graph.
        ig_undirected (ig.Graph): Undirected iGraph graph.
        ig_directed (ig.Graph): Directed iGraph graph.

    Returns:
        Dict[str, Dict[Any, float]]:
            Dictionary where keys are metric names (str, prefixed
            with 'nx_' or 'ig_') and values are dictionaries mapping node
            identifiers (NetworkX nodes or iGraph 'name' attributes) to
            metric values (float).

    Raises:
        TypeError: If input arguments are not of the expected graph types
                   or directedness.
        AttributeError: If iGraph objects lack 'name' vertex attribute or
                        'weight' edge attribute when edges exist.
        ValueError: If graphs are empty or required attributes are missing.
        Exception: Can raise specific exceptions from NetworkX or iGraph
                   during metric calculation (e.g., convergence errors).
    """
    if not isinstance(edge_df, pd.DataFrame):
         raise TypeError("edge_df must be a pandas DataFrame.")
    if not isinstance(nx_undirected, nx.Graph) or isinstance(nx_undirected, nx.DiGraph):
        raise TypeError("nx_undirected must be a non-directed nx.Graph.")
    if not isinstance(nx_directed, nx.DiGraph):
        raise TypeError("nx_directed must be an nx.DiGraph.")
    if not isinstance(ig_undirected, ig.Graph) or ig_undirected.is_directed():
        raise TypeError("ig_undirected must be an undirected ig.Graph.")
    if not isinstance(ig_directed, ig.Graph) or not ig_directed.is_directed():
        raise TypeError("ig_directed must be a directed ig.Graph.")

    # non-empty check
    if not nx_undirected or not nx_directed or \
       ig_undirected.vcount() == 0 or ig_directed.vcount() == 0:
        raise ValueError("Input graphs cannot be empty.")
    
    edge_df = edge_df_check(edge_df)

    node_metrics = {}
    node_metrics['degree_in'] = dict(nx_directed.in_degree())
    node_metrics['degree_out'] = dict(nx_directed.out_degree())
    node_metrics['strength_in'] = \
        dict(nx_directed.in_degree(weight='weight'))
    node_metrics['strength_out'] = \
        dict(nx_directed.out_degree(weight='weight'))
    node_metrics['centrality_katz'] = \
        nx.katz_centrality_numpy(nx_undirected)
    node_metrics['centrality_betweenness'] = \
        nx.betweenness_centrality(nx_directed, weight='weight',normalized=True)
    node_metrics['centrality_closeness'] = \
        nx.closeness_centrality(nx_directed, distance='weight')
    node_metrics['clustering'] = nx.clustering(nx_directed, weight='weight')
    node_metrics['pagerank'] = nx.pagerank(nx_directed, weight='weight')
    node_metrics['burts_constraint'] = nx.constraint(nx_directed, weight='weight')

    node_metrics = pd.DataFrame(node_metrics)
    node_metrics = node_metrics[np.sort(node_metrics.columns)]
    return(node_metrics)

def compute_network_metrics(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    nx_undirected: nx.Graph,
    nx_directed: nx.DiGraph,
    ig_undirected: ig.Graph,
    ig_directed: ig.Graph) -> Dict[str, Any]:
    """
    Calculates global network metrics using NetworkX and iGraph objects,
    and computes average node metrics from the provided node_df.

    Args:
        node_df (pd.DataFrame): DataFrame where the index represents node names
                                and columns contain pre-calculated node metrics
                                (e.g., from compute_node_metrics).
        nx_undirected (nx.Graph): Undirected NetworkX graph.
        nx_directed (nx.DiGraph): Directed NetworkX graph.
        ig_undirected (ig.Graph): Undirected iGraph graph.
        ig_directed (ig.Graph): Directed iGraph graph.

    Returns:
        Dict[str, Any]: A dictionary containing various network-level metrics.
                        Keys indicate the metric name (often prefixed with 'nx_' or 'ig_').
                        Values are the calculated metric (int, float, bool, list, or None).

    Raises:
        TypeError: If input arguments are not of the expected graph types or directedness.
        ValueError: If graphs are empty, or if node_df is unsuitable.
        AttributeError: If iGraph objects lack expected attributes ('name', 'weight').
        KeyError: If node_df is missing expected columns for averaging, or during calculation if dependencies are missing.
        ZeroDivisionError: May be raised by density calculations if graph has 0 or 1 node.
        nx.NetworkXError: May be raised by various NetworkX functions (e.g., diameter on disconnected component).
        ig.InternalError: May be raised by various iGraph functions.
        Exception: Other specific exceptions from NetworkX or iGraph during metric calculation.
    """
    if not isinstance(edge_df, pd.DataFrame):
         raise TypeError("edge_df must be a pandas DataFrame.")
    if not isinstance(node_df, pd.DataFrame):
         raise TypeError("node_df must be a pandas DataFrame.")
    if not isinstance(nx_undirected, nx.Graph) or isinstance(nx_undirected, nx.DiGraph):
        raise TypeError("nx_undirected must be a non-directed nx.Graph.")
    if not isinstance(nx_directed, nx.DiGraph):
        raise TypeError("nx_directed must be an nx.DiGraph.")
    if not isinstance(ig_undirected, ig.Graph) or ig_undirected.is_directed():
        raise TypeError("ig_undirected must be an undirected ig.Graph.")
    if not isinstance(ig_directed, ig.Graph) or not ig_directed.is_directed():
        raise TypeError("ig_directed must be a directed ig.Graph.")

    edge_df = edge_df_check(edge_df)

    network_metrics = calculate_standard_metrics(edge_df)

    #Add network metrics
    network_metrics['weight_max'] = edge_df['weight'].max()

    network_metrics['density_undirected'] = nx.density(nx_undirected)
    network_metrics['density_directed'] = nx.density(nx_directed)

    network_metrics['density_weighted'] = network_metrics['weight_total']\
         / (network_metrics['n_edges'] * network_metrics['weight_max'])
    
    network_metrics['efficiency_global'] = nx.global_efficiency(nx_undirected)
    network_metrics['efficiency_local'] = nx.local_efficiency(nx_undirected)

    network_metrics['is_connected'] = nx.is_connected(nx_undirected)
    network_metrics['number_connected_components'] = \
            nx.number_connected_components(nx_undirected)
    network_metrics['is_weakly_connected'] = nx.is_weakly_connected(nx_directed)
    network_metrics['number_weakly_connected_components'] = \
        nx.number_weakly_connected_components(nx_directed)
    network_metrics['is_strongly_connected'] = \
        nx.is_strongly_connected(nx_directed)
    network_metrics['number_strongly_connected_components'] = \
        nx.number_strongly_connected_components(nx_directed)
    network_metrics['avg_clustering'] = \
        nx.average_clustering(nx_directed, weight='weight')
    network_metrics['reciprocity'] = nx.reciprocity(nx_directed)
    network_metrics['overall_reciprocity'] = nx.overall_reciprocity(nx_directed)
   
    network_metrics['global_clustering'] = ig_undirected.transitivity_undirected()

    # Greedy modularity
    network_metrics['modularity_greedy'] = nx.algorithms.community.modularity(nx_undirected, 
        nx.algorithms.community.greedy_modularity_communities(nx_undirected))
    
    # Random walk modularity
    random_walk_steps=5
    giant_component = ig_directed.components(mode='weak').giant()
    random_walk = giant_component.community_walktrap(steps=random_walk_steps,
                     weights=giant_component.es['weight']).as_clustering()
    network_metrics['modularity_randomwalk'] = giant_component.modularity(random_walk.membership)

    network_metrics['modularity_community'] = nx.community.modularity(nx_undirected, 
                                    nx.community.label_propagation_communities(nx_undirected))

    node_numeric_cols = node_df.select_dtypes(include='number').columns

    # Apply standard aggregations  across selected columns
    standard_aggregations = ['mean', 'median', 'std', 'min', 'max']
    standard_aggregations = ['mean', 'median']
    agg_results_df = node_df[node_numeric_cols].agg(standard_aggregations)

    for agg_name, row_series in agg_results_df.iterrows():
        prefix =  agg_name
        for col_name, value in row_series.items():
            metric_key = f"{col_name}_{prefix}"
            network_metrics[metric_key] = value
    return network_metrics

def compute_network(df_in: pd.DataFrame, 
                  from_var: str = 'prevhospid', 
                  to_var: str = 'hospid',
                  edge_cutoff=1,
                  return_initial=False):

    #Calculate included edges, all edges (all edges only used in computing initial_[vars])
    edge_df = process_edges(df_in, from_var, to_var)

    #Compute initial metrics
    network_metrics_dict = calculate_standard_metrics(edge_df)
    
    network_metrics_dict_initial = {f'{key}_initial':val for \
                        key,val in network_metrics_dict.items() }
    if return_initial:
        network_metrics_dict.update(network_metrics_dict_initial)

    #Restrict to edges matching cutoff
    edge_df = edge_df[edge_df['weight']>=edge_cutoff]

    #Construct networkx and Igraph directed and undirected   
    nx_undirected, nx_directed = create_networkx_graphs(edge_df)
    ig_undirected, ig_directed = create_igraph_graphs(edge_df)

    #Compute nodes and edges
    node_metrics = compute_node_metrics(edge_df,
                nx_undirected, nx_directed, ig_undirected, ig_directed)
    network_metrics = compute_network_metrics(edge_df,node_metrics,
                nx_undirected, nx_directed, ig_undirected, ig_directed)
    network_metrics_dict.update(network_metrics)

    #Connected component analysis
    # Number of Connected Components
    num_connected_components = nx.number_connected_components( nx_undirected)
    connected_components_nodes = list(nx.connected_components( nx_undirected))
    component_summary_list = list()

    if len(connected_components_nodes)==1:
        out = {'node_pct':1, 'edge_pct':1,'weight_pct':1}
        out.update(network_metrics)
        component_summary_list.append(out)
    if (len(connected_components_nodes)>1):
        for component_nodes in connected_components_nodes:
            edge_df_use = edge_df[edge_df['to'].isin(component_nodes) \
                                | edge_df['from'].isin(component_nodes)]
            nx_undirected_use, nx_directed_use = create_networkx_graphs(edge_df_use)
            ig_undirected_use, ig_directed_use = create_igraph_graphs(edge_df_use)
            node_metrics_use = compute_node_metrics(edge_df_use,
            nx_undirected_use, nx_directed_use,ig_undirected_use, ig_directed_use)

            network_metrics_component = compute_network_metrics(edge_df_use, node_metrics_use,
                nx_undirected_use, nx_directed_use, ig_undirected_use, ig_directed_use)
            
            out = dict()
            for summary_type, summary_level_var in zip(['node','edge','weight'],
                                    ['n_nodes','n_edges','weight_total']):
                out[f'{summary_type}_pct'] = network_metrics_component[summary_level_var]\
                                            /network_metrics_dict[summary_level_var]    
            out.update(network_metrics_component)
            component_summary_list.append(out)

    df_components = pd.DataFrame(component_summary_list)
    df_components = df_components[np.sort(df_components.columns)]

    return({'metrics':network_metrics_dict,
                    'node_metrics':node_metrics,
                    'edge_metrics':edge_df,
                    'df_components':df_components,
                    'nx_undirected':nx_undirected,
                    'nx_directed':nx_directed,
                    'ig_undirected':ig_undirected,
                    'ig_directed':ig_directed})