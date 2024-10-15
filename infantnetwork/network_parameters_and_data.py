
import pandas as pd

#Dictionary of metrics for each metric category (e.g. modularity)
metric_type_dict = {\
    'centrality':['centrality_median','centrality_mean_receiving'],
    'density':['density_weighted','density_unweighted'],
    'modularity':['modularity_greedy','modularity_randomwalk'],
    'efficiency':['efficiency_global', 'efficiency_median_local'],
    'shape':['n_nodes','n_edges']}

#Sample datasets
edge_dict = {'none':[],
        'one':[(1,2)],
        'three':[(1,2),(2,1)],
        'star':[(3,1),(2,1),(4,1)],
        'sample':[(2, 1),
                        (2, 1),
                        (2, 1),
                        (3, 1),
                        (3, 1),
                        (3, 1),
                        (4, 1),
                        (4, 1),
                        (4, 1),
                        (5, 1),
                        (5, 1),
                        (5, 1),
                        (6, 1),
                        (6, 1),
                        (6, 1),
                        (7, 1),
                        (7, 1),
                        (7, 1),
                        (8, 1),
                        (8, 1),
                        (8, 1),
                        (9, 1),
                        (9, 1),
                        (9, 1),
                        (10, 1),
                        (10, 1),
                        (10, 1),
                        (2, 3),
                        (2, 3),
                        (4, 5),
                        (4, 5),
                        (6, 7),
                        (6, 7),
                        (8, 9),
                        (8, 9)]
    }

transfer_dict = {key: pd.DataFrame(value, columns=['prevhospid', 'hospid'])\
                                 for key, value in edge_dict.items()}

sample_transfers = transfer_dict['sample']

# sample_edges = {'none':[],
#                 'one':[(1,2)]}
# edges_df = pd.DataFrame(edges, columns=['prevhospid', 'hospid'])


# # Convert all keys in the dictionary to DataFrames while keeping the same keys
# edges_df_dict = {key: pd.DataFrame(columns=['prevhospid', 'hospid']) for key, value in sample_edges.items()}



# transfers_high_centrality=1
