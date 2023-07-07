import argparse
import json
import pathlib

from hierarchical_mapping.utils.anndata_utils import (
    read_df_from_h5ad)


def create_mapping(
        input_dir_list,
        output_path,
        valid_columns=None):
    """
    Construct the JSON blob that holds the data for
    mapping between different gene identifier schemes.

    Parameters
    ----------
    input_dir_list:
        List of directories to scan for h5ad files. All relevant
        columns in the var dataframes will be used to construct
        the blob
    output_path:
        Path to the file where the JSON blob will be written. Blob will
        be a dict mapping gene_identifier to other identifications for
        a gene.
    valid_columns:
        List of columns to be assembled in the JSON blob. If None will
        use ['gene_identifier', 'gene_symbol', 'transcript_identifier']
    """

    output_path = pathlib.Path(output_path)

    if valid_columns is None:
        valid_columns = [
            'gene_symbol',
            'gene_identifier',
            'transcript_identifier']

    result_lookup = dict()
    for this_dir in input_dir_list:
        print(f"processing {this_dir}")
        this_dir = pathlib.Path(this_dir)
        h5ad_path_list = [n for n in this_dir.iterdir()
                          if n.is_file() and 'h5ad' in n.name]
        for h5ad_path in h5ad_path_list:
            result_lookup = _update_from_file(
                h5ad_path=h5ad_path,
                result_lookup=result_lookup,
                valid_columns=valid_columns)

    with open(output_path, 'w') as out_file:
        out_file.write(json.dumps(
            result_lookup,
            indent=2,
            sort_keys=True))


def _update_from_file(
        h5ad_path,
        result_lookup,
        valid_columns):
    """
    h5ad_path is the path to the h5ad file we are getting data from
    result_lookup is the JSON blob we are updated
    valid_columns are the columns in the var dataframe we are interested in
    """

    var = read_df_from_h5ad(h5ad_path, 'var')
    if var.index.name != 'gene_identifier':
        raise RuntimeError(
            f"Index of var in {h5ad_path} is {var.index.name}; "
            "not 'gene_identifier'")

    valid_columns = set(valid_columns)
    gene_identifier_list = list(var.index.values)
    var = var.to_dict(orient='records')
    valid_columns = valid_columns.intersection(set(var[0].keys()))
    for gene_id, gene in zip(gene_identifier_list, var):
        if 'gene_identifier' in gene:
            if gene['gene_identifier'] != gene_id:
                raise RuntimeError(
                    f"mis alignment in gene_id for {gene} in {h5ad_path}\n"
                    f"index says gene_id is {gene_id}")

        if gene_id not in result_lookup:
            result_lookup[gene_id] = dict()

        for col in valid_columns:
            if col in result_lookup[gene_id]:
                if result_lookup[gene_id][col] != gene[col]:
                    raise RuntimeError(
                        f"many values of {col} for gene {gene_id}:\n"
                        f"{gene[col]}\n{result_lookup[gene_id][col]}\n")
            else:
                result_lookup[gene_id][col] = gene[col]
    return result_lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, nargs='+', default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    assert args.output_path is not None
    if not isinstance(args.input_dir, list):
        input_dir_list = [args.input_dir]
    else:
        input_dir_list = args.input_dir

    create_mapping(
        input_dir_list=input_dir_list,
        output_path=args.output_path)

if __name__ == "__main__":
    main()
