import json
import pathlib


def blob_to_csv(
        results_blob,
        taxonomy_tree,
        output_path,
        metadata_path=None):
    """
    Write a set of results originally formatted for a JSON blob
    out to the CSV our users will expect.

    Parameters
    ----------
    results_blob:
        The blob-form results. A list of dicts. Each dict is a cell
        and looks like
            {
                'cell_id': my_cell,
                'level1': {'assignment':...
                           'confidence':...},
                'level2': {'assignment':...,
                           'confidence':...}}
    taxonomy_tree:
        TaxonomyTree that went into this run.
        Hierarchy will be listed as a comment at the top of this file.
    output_path:
        Path to the CSV file that will be written
    metadata_path:
        Path to the metadta path going with this output (if any;
        file name will be recorded as a comment at the top of
        the file)
    """
    str_hierarchy = json.dumps(taxonomy_tree.hierarchy)

    with open(output_path, 'w') as dst:
        if metadata_path is not None:
            metadata_path = pathlib.Path(metadata_path)
            dst.write(f'# metadata = {metadata_path.name}\n')
        dst.write(f'# taxonomy hierarchy = {str_hierarchy}\n')
        header = 'cell_id,'
        for level in taxonomy_tree.hierarchy:
            header += f'{level}_label,{level}_confidence,'
        header = header[:-1] + '\n'
        dst.write(header)
        for cell in results_blob:
            values = [cell['cell_id']]
            for level in taxonomy_tree.hierarchy:
                label = cell[level]['assignment']
                name = taxonomy_tree.label_to_name(
                            level=level,
                            label=label,
                            name_key='name')
                values.append(label)
                values.append(f"{cell[level]['confidence']:.4f}")
            dst.write(",".join(values)+"\n")
