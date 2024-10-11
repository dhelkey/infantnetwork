from infantnetwork import computeNetwork, \
    metric_type_dict, sample_transfers



def test_computeNetwork_null():
    transfers = sample_transfers['none']
    expected_metrics = {'n_nodes':0,
                        'n_edges':0,
                            'n_transfers':0}
    output = computeNetwork(transfers)
    metrics = output['metrics']
    for metric, expected_value in expected_metrics.items():
        #asert metrics[metric] == value
        actual_value = metrics.get(metric, None)
        assert actual_value == expected_value,\
        f"Metric '{metric}' failed: expected {expected_value}, got {actual_value}"

