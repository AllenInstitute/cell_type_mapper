import numpy as np
import warnings

import cell_type_mapper.utils.anndata_utils as anndata_utils


class DuplicateGeneIDWarning(UserWarning):
    pass


def invalid_precompute_prefix():
    """
    Prefix for genes that are duplicated at precomputation
    stage
    """
    return 'INVALID_MARKER'


def invalid_query_prefix():
    """
    Prefix for genes that are duplicated at the mapping
    stage
    """
    return 'DUPLICATED_QUERY_GENE'



def get_gene_identifier_list(
        h5ad_path_list,
        gene_id_col,
        duplicate_prefix=invalid_precompute_prefix()):
    """
    Get list of gene identifiers from a list of h5ad files.

    Parameters
    ----------
    h5ad_path_list:
        list of h5ad files to extract gene identifiers from
    gene_id_col:
        column in var from which to get gene identifiers.
        If None, use the index of var
    duplicate_prefix:
        the prefix to add to the names of genes that are
        found to be duplicated in the original list of
        gene identifiers

    Returns
    -------
    list of gene identifiers

    Notes
    -----
    will raise an exception of the h5ad files give different
    results for gene name list
    """
    gene_names = None
    for pth in h5ad_path_list:
        var = anndata_utils.read_df_from_h5ad(pth, 'var')
        if gene_id_col is None:
            these_genes = list(var.index.values)
        else:
            these_genes = list(var[gene_id_col].values)

        if gene_names is None:
            gene_names = these_genes
        else:
            if gene_names != these_genes:
                raise RuntimeError(
                    "Inconsistent gene names list\n"
                    f"{pth}\nhas gene_names\n{these_genes}\n"
                    f"which does not match\n{h5ad_path_list[0]}\n"
                    f"genes\n{gene_names}")

    gene_names = mask_duplicate_gene_identifiers(
        gene_identifier_list=gene_names,
        mask_prefix=invalid_precompute_prefix(),
        log=None
    )

    return gene_names


def mask_duplicate_gene_identifiers(
        gene_identifier_list,
        mask_prefix='DUPLICATE_GENE',
        log=None):
    """
    Take a list of gene identifiers. Replace any duplicated
    identifiers with a nonsense (i.e. not a valid gene name)
    string so that they will not be used in the actual
    mapping process.

    Parameters
    ----------
    gene_identifier_list:
        input list of gene identifiers
    mask_prefix:
        string; the prefix for any invalid gene identifiers
    log:
        a CommandLog for recording which genes were duplicated

    Returns
    -------
    gene_identifier_list doctored so that duplicate gene
    identifiers have been replaced in such a way that gene identifiers
    are unique, but those that were duplicated in the input file now
    contain nonsense.
    """
    unq, ct = np.unique(gene_identifier_list, return_counts=True)
    duplicate_to_ct = dict()
    for unq_val, ct_val in zip(unq, ct):
        if ct_val == 1:
            continue
        duplicate_to_ct[unq_val] = 0

    if len(duplicate_to_ct) > 0:
        msg = (
            "The following gene identifiers occurred more than once "
            "in your data. They will be ignored: "
        )
        msg += str(sorted(duplicate_to_ct.keys()))
        if log is not None:
            log.warn(msg)
        else:
            warnings.warn(
                message=msg,
                category=DuplicateGeneIDWarning
            )

    output = []
    for gene in gene_identifier_list:
        if gene not in duplicate_to_ct:
            output.append(gene)
        else:
            ct = duplicate_to_ct[gene]
            new_name = f'{mask_prefix}_{gene}_{ct}'
            duplicate_to_ct[gene] += 1
            output.append(new_name)
    return output

