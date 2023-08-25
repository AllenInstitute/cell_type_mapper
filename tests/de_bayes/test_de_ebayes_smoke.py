"""
Run a smoketest to make sure we didn't break anything when we
copy-and-pasted the code from transcriptomic_clustering
"""
import numpy as np
import pandas as pd

from cell_type_mapper.de_bayes.de_ebayes import de_pairs_ebayes



def test_ebayes_smoke():
    n_genes = 45
    rng = np.random.default_rng(22312)
    pairs = [('clA', 'clB')]

    mean_data = []
    var_data = []
    for cluster in pairs[0]:
        this_mean = {'cluster_name': cluster}
        this_var = {'cluster_name': cluster}
        for ii in range(n_genes):
            this_mean[f'gene_{ii}'] = rng.random()
            this_var[f'gene_{ii}'] = rng.random()
        mean_data.append(this_mean)
        var_data.append(this_var)
    mean_df = pd.DataFrame(mean_data).set_index('cluster_name')
    var_df = pd.DataFrame(var_data).set_index('cluster_name')
    cluster_size = {cl: rng.integers(25, 66) for cl in pairs[0]}
    result = de_pairs_ebayes(
        pairs=pairs,
        cl_means=mean_df,
        cl_vars=var_df,
        cl_size=cluster_size,
        p_th=0.01)
    
    assert len(result) == 1
    assert pairs[0] in result
    assert isinstance(result[pairs[0]], np.ndarray)
    assert result[pairs[0]].dtype == np.float64
    assert len(result[pairs[0]]) == n_genes
