"""
Define data assets needed to test the cell type mapper
with the mmc_gene_mapper infrastructure incorporated into it
"""
import pytest

import hashlib
import json
import pandas as pd
import pathlib
import tarfile
import tempfile
import unittest.mock

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up
)

import mmc_gene_mapper.mapper.mapper as mapper


@pytest.fixture(scope='session')
def tmp_dir_fixture(tmp_path_factory):
    result = tmp_path_factory.mktemp(
        'mmc_gene_mapper_integration_'
    )
    yield result
    _clean_up(result)


@pytest.fixture(scope='session')
def ncbi_file_package_fixture(
        tmp_dir_fixture):
    """
    Populate NCBI gene files for three species
    mouse = 10090
    human = 9606
    fish = 999

    mouse will have genes ranging from
    NCBIGene:{ii} for ii in 0 - 100
    Every gene will correspond to ENSM{ii}

    human will have NCBIGene{ii} for ii in 100-200
    but only ENSG genes for even values

    fish wil have NCBIGene:{ii} for ii in 200-300
    but only ENSF genes for odd values

    Return dict mapping
        'gene_info'
        'gene2ensembl'
        'gene_orthologs'

    to the corresponding gzipped CSVs

    There will also be bkbit files for each species under
    'human_bkbit'
    'mouse_bkbit'
    'fish_bkbit'

    """
    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    human_bkbit = [
        {"id": "NCBITaxon:9606",
         "category": ["biolink:OrganismTaxon"],
         "name": "human",
         "full_name": "Homo sapiens"},
        {"id": "humanAssembly",
         "category": ["bican:GenomeAssembly"],
         "name": "H001"},
        {"id": "H001-2025",
         "category": ["bican:GenomeAnnotation"],
         "version": 0,
         "authority": "ENSEMBL"}
    ]

    mouse_bkbit = [
        {"id": "NCBITaxon:10090",
         "category": ["biolink:OrganismTaxon"],
         "name": "mouse",
         "full_name": "Mus musculus"},
        {"id": "mouseAssembly",
         "category": ["bican:GenomeAssembly"],
         "name": "M001"},
        {"id": "M001-2025",
         "category": ["bican:GenomeAnnotation"],
         "version": 0,
         "authority": "ENSEMBL"}
    ]

    fish_bkbit = [
        {"id": "NCBITaxon:999",
         "category": ["biolink:OrganismTaxon"],
         "name": "fish",
         "full_name": "pisces"},
        {"id": "humanAssembly",
         "category": ["bican:GenomeAssembly"],
         "name": "F001"},
        {"id": "F001-2025",
         "category": ["bican:GenomeAnnotation"],
         "version": 0,
         "authority": "ENSEMBL"}
    ]

    n_genes = 100
    gene_info = []
    gene_2_ensembl = []
    gene_orthologs = []
    for ii in range(n_genes):
        mouse_ncbi = {
            '#tax_id': 10090,
            'GeneID': ii,
            'Symbol': f'mouse_symbol_{ii}'
        }
        human_ncbi = {
            '#tax_id': 9606,
            'GeneID': 100+ii,
            'Symbol': f'human_symbol_{ii}'
        }
        fish_ncbi = {
            '#tax_id': 999,
            'GeneID': 200+ii,
            'Symbol': f'fish_symbol_{ii}'
        }
        gene_info.append(mouse_ncbi)
        gene_info.append(human_ncbi)
        gene_info.append(fish_ncbi)

        mouse_2_ens = {
            '#tax_id': 10090,
            'GeneID': mouse_ncbi['GeneID'],
            'Ensembl_gene_identifier': f'ENSM{mouse_ncbi["GeneID"]}'
        }

        mouse_bkbit.append(
            {"category": ["bican:GeneAnnotation"],
             "source_id": mouse_2_ens["Ensembl_gene_identifier"],
             "symbol": f"e_mouse_symbol_{mouse_ncbi['GeneID']}",
             "name": f"e_mouse_name_{mouse_ncbi['GeneID']}",
             "in_taxon_label": "Mus musculus"}
        )

        gene_2_ensembl.append(mouse_2_ens)
        if ii % 2 == 0:
            human_2_ens = {
                '#tax_id': 9606,
                'GeneID': human_ncbi['GeneID'],
                'Ensembl_gene_identifier': f'ENSG{human_ncbi["GeneID"]}'
            }
            gene_2_ensembl.append(human_2_ens)
            human_bkbit.append(
                {"category": ["bican:GeneAnnotation"],
                 "source_id": human_2_ens["Ensembl_gene_identifier"],
                 "symbol": f"e_human_symbol_{human_ncbi['GeneID']}",
                 "name": f"e_human_name_{human_ncbi['GeneID']}",
                 "in_taxon_label": "Homo sapiens"}
            )

        else:
            fish_2_ens = {
                '#tax_id': 999,
                'GeneID': fish_ncbi['GeneID'],
                'Ensembl_gene_identifier': f'ENSF{fish_ncbi["GeneID"]}'
            }
            gene_2_ensembl.append(fish_2_ens)
            fish_bkbit.append(
                {"category": ["bican:GeneAnnotation"],
                 "source_id": fish_2_ens["Ensembl_gene_identifier"],
                 "symbol": f"e_fish_symbol_{fish_ncbi['GeneID']}",
                 "name": f"e_fish_name_{fish_ncbi['GeneID']}",
                 "in_taxon_label": "pisces"}
            )

        gene_orthologs.append(
            {'#tax_id': 999,
             'GeneID': fish_ncbi['GeneID'],
             'Other_tax_id': 10090,
             'Other_GeneID': mouse_ncbi['GeneID'],
             'relationship': 'Ortholog'}
        )
        gene_orthologs.append(
            {'#tax_id': 9606,
             'GeneID': human_ncbi['GeneID'],
             'Other_tax_id': 999,
             'Other_GeneID': fish_ncbi['GeneID'],
             'relationship': 'Ortholog'}
        )

    gene_info_path = f'{this_tmp}/gene_info.gz'
    pd.DataFrame(gene_info).to_csv(
        gene_info_path,
        index=False,
        compression='gzip',
        sep='\t')

    gene2ensembl_path = f'{this_tmp}/gene2enesmbl.gz'
    pd.DataFrame(gene_2_ensembl).to_csv(
        gene2ensembl_path,
        index=False,
        compression='gzip',
        sep='\t')

    gene_orthologs_path = f'{this_tmp}/gene_orthologs.gz'
    pd.DataFrame(gene_orthologs).to_csv(
        gene_orthologs_path,
        index=False,
        compression='gzip',
        sep='\t')

    human_bkbit_path = f'{this_tmp}/human_bkbit.jsonld'
    with open(human_bkbit_path, 'w') as dst:
        dst.write(
            json.dumps(
                {'@graph': human_bkbit},
                indent=2
            )
        )

    mouse_bkbit_path = f'{this_tmp}/mouse_bkbit.jsonld'
    with open(mouse_bkbit_path, 'w') as dst:
        dst.write(
            json.dumps(
                {'@graph': mouse_bkbit},
                indent=2
            )
        )

    fish_bkbit_path = f'{this_tmp}/fish_bkbit.jsonld'
    with open(fish_bkbit_path, 'w') as dst:
        dst.write(
            json.dumps(
                {'@graph': fish_bkbit},
                indent=2
            )
        )

    return {
        'gene_info': gene_info_path,
        'gene2ensembl': gene2ensembl_path,
        'gene_orthologs': gene_orthologs_path,
        'human_bkbit': human_bkbit_path,
        'mouse_bkbit': mouse_bkbit_path,
        'fish_bkbit': fish_bkbit_path
    }


@pytest.fixture(scope='session')
def species_file_fixture(
        tmp_dir_fixture):
    """
    Create a tsv simulating names.dmp. Put it into a tarfile.
    Create the text file with that tarfile's md5 hash.
    Return the path to the tarfile and the text file.
    This is meant to simulate downloading the taxons from NCBI.
    """
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir_fixture,
            prefix='species_simulation_'
        )
    )

    name_path = tmp_dir / 'names.dmp'
    with open(name_path, 'w') as dst:
        for species_name, species_id in [('human', 9606),
                                         ('mouse', 10090),
                                         ('fish', 999),
                                         ('Homo sapiens', 9606),
                                         ('Mus musculus', 10090),
                                         ('pisces', 999)]:
            dst.write(f'{species_id}\t|\t{species_name}\t|\t\n')

    tar_path = tmp_dir / 'new_taxdump.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as dst:
        with open(name_path, 'rb') as src:
            t_info = dst.gettarinfo(fileobj=src)
            t_info.name = name_path.name
            dst.addfile(
                tarinfo=t_info,
                fileobj=src
            )

    hasher = hashlib.md5()
    with open(tar_path, 'rb') as src:
        hasher.update(src.read())
    hash_path = tmp_dir / 'new_taxdump.tar.gz.md5'
    with open(hash_path, 'w') as dst:
        dst.write(hasher.hexdigest())

    return {"tar": tar_path, "hash": hash_path}


@pytest.fixture(scope='session')
def dummy_download_mgr_fixture(
        species_file_fixture,
        ncbi_file_package_fixture):
    """
    define dummy downloan manager class
    """
    class DummyDownloadManager(object):
        def __init__(self, *args, **kwargs):
            pass

        def get_file(
                self,
                host,
                src_path,
                force_download):
            if src_path.endswith('new_taxdump.tar.gz'):
                return {
                    'local_path': species_file_fixture['tar']
                }
            elif src_path.endswith('new_taxdump.tar.gz.md5'):
                return {
                    'local_path': species_file_fixture['hash']
                }
            elif src_path.endswith('gene_info.gz'):
                return {
                    'local_path': ncbi_file_package_fixture['gene_info']
                }
            elif src_path.endswith('gene2ensembl.gz'):
                return {
                    'local_path': ncbi_file_package_fixture['gene2ensembl']
                }
            elif src_path.endswith('gene_orthologs.gz'):
                return {
                    'local_path': ncbi_file_package_fixture['gene_orthologs']
                }
            else:
                raise RuntimeError(
                    "MockDownloadManager cannot handle src_path "
                    f"{src_path}"
                )

    return DummyDownloadManager


@pytest.fixture(scope='session')
def mapper_fixture(
        ncbi_file_package_fixture,
        dummy_download_mgr_fixture,
        tmp_dir_fixture):
    """
    Return an instantiation of the MMCGeneMapper class
    based on our simulated NCBI file package
    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)
    db_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='ncbi_gene_mapper_',
        suffix='.db',
        delete=True
    )

    file_spec = [
        {'type': 'bkbit',
         'path': ncbi_file_package_fixture[k]}
        for k in ('human_bkbit', 'mouse_bkbit', 'fish_bkbit')
    ]

    to_replace = "mmc_gene_mapper.download.download_manager.DownloadManager"
    with unittest.mock.patch(to_replace, dummy_download_mgr_fixture):
        gene_mapper = mapper.MMCGeneMapper.create_mapper(
            db_path=db_path,
            local_dir=tmp_dir_fixture,
            data_file_spec=file_spec,
            clobber=False,
            force_download=False,
            suppress_download_stdout=True
        )

    return gene_mapper


@pytest.fixture(scope='session')
def mapper_db_path_fixture(mapper_fixture):
    return str(mapper_fixture.db_path)
