import time

import numpy as np

from cell_type_mapper.utils.stats_utils import (
    boring_t_from_p_value)

from cell_type_mapper.diff_exp.scores import (
    score_differential_genes)

def main():
    rng = np.random.default_rng(87123)
    n_genes = 32000

    n_iter = 1000
    dur = []
    t0 = time.time()
    for ii in range(n_iter):
        mean0 = rng.random(n_genes)*5.0
        mean1 = rng.random(n_genes)*5.0
        var0 = rng.random(n_genes)*2.0
        var1 = rng.random(n_genes)*2.0
        ge1_0 = rng.integers(0, 255, n_genes, dtype=int)
        ge1_1 = rng.integers(0, 255, n_genes, dtype=int)
        n0 = rng.integers(256, 512)
        n1 = rng.integers(256, 512)

        t0 = time.time()
        mask = return_differential_genes(
            mean0=mean0,
            var0=var0,
            ge1_0=ge1_0,
            n0=n0,
            mean1=mean1,
            var1=var1,
            ge1_1=ge1_1,
            n1=n1,
            p_th=0.01)
        dur.append(time.time()-t0)

    dur = np.array(dur)
    print(f'{n_iter} iterations in {np.mean(dur):.2e}  '
          f'+/- {np.std(dur, ddof=1):.2e} seconds per iteration')
    print(f'{n_genes:.2e} genes')
        


def return_differential_genes(
        mean0,
        var0,
        ge1_0,
        n0,
        mean1,
        var1,
        ge1_1,
        n1,
        p_th=0.01):
    """
    Read in summary statistics from two cell-by-gene populations.

    Return a boolean mask indicating which genes are valid markers
    for discriminating between the two populations.

    Parameters
    ----------
    mean0:
        An (n_genes,) numpy array containing the mean expression of
        the cells in population0 in each gene
    var0:
        An (n_genes,) numpy array containing the variance of the
        celles in population0 in each gene
    ge1_0:
        An (n_genes,) numpy array containing the counts of how many
        cells in population0 express each gene at greater than or
        equal to 1 CPM
    n0:
        An integer indicating how many cells are in population0

    mean1:
        An (n_genes,) numpy array containing the mean expression of
        the cells in population1 in each gene
    var1:
        An (n_genes,) numpy array containing the variance of the
        celles in population1 in each gene
    ge1_1:
        An (n_genes,) numpy array containing the counts of how many
        cells in population1 express each gene at greater than or
        equal to 1 CPM
    n1:
        An integer indicating how many cells are in population1

    p_th:
       Minimum p-value to accept for marker genes

    Returns
    -------
    valid_markers:
        An (n_genes,) numpy array of booleans that is True for all
        the genes that are valid markers for discriminating between
        population0 and population1.
    """

    # reshape the input data into a dict that the MapMyCells code
    # expected
    node_1 = 'pop1'
    node_2 = 'pop2'

    stats = {
        node_1: {
            'mean': mean0,
            'var': var0,
            'ge1': ge1_0,
            'n_cells': n0
        },
        node_2: {
            'mean': mean1,
            'var': var1,
            'ge1': ge1_1,
            'n_cells': n1
        }
    }

    (_,
     validity,
     _) = score_differential_genes(
             node_1=node_1,
             node_2=node_2,
             precomputed_stats=stats,
             p_th=p_th,
             boring_t=boring_t_from_p_value(p_th),
             exact_penetrance=True)

    return validity



if __name__ == "__main__":
    main()
