import pytest

import hashlib
import json
import multiprocessing
import pathlib
import pandas as pd
import tarfile
import tempfile
import unittest.mock

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

import cell_type_mapper.test_utils.gene_mapping.mappers as legacy_mappers

import mmc_gene_mapper.mapper.mapper as mapper


multiprocessing.set_start_method('fork', force=True)


@pytest.fixture(scope='session')
def tmp_dir_fixture(
        tmp_path_factory):
    result = tmp_path_factory.mktemp('cell_type_mapper_')
    yield result
    _clean_up(result)


#########################################################################
# Below this we define fixtures in support of mmc_gene_mapper integration
#########################################################################

@pytest.fixture(scope='session')
def legacy_gene_mapper_tmp_dir_fixture(
        tmp_path_factory):
    result = tmp_path_factory.mktemp('legacy_gene_mapper_')
    yield result
    _clean_up(result)


@pytest.fixture(scope='session')
def legacy_gene_mapping_files(legacy_gene_mapper_tmp_dir_fixture):
    tmp_dir = tempfile.mkdtemp(
        dir=legacy_gene_mapper_tmp_dir_fixture,
        prefix='legacy_gene_mappings_'
    )

    mouse_taxon = 10090
    human_taxon = 9606

    human_bkbit = [
        {"id": f"NCBITaxon:{human_taxon}",
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
        {"id": f"NCBITaxon:{mouse_taxon}",
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

    legacy_mouse = legacy_mappers.get_mouse_gene_id_mapping()
    legacy_human = legacy_mappers.get_human_gene_id_mapping()

    ncbi_2_ens = []
    gene_info = []
    gene_orthologs = [
        {'#tax_id': 999,
         'GeneID': 999999,
         'Other_tax_id': 9999,
         'Other_GeneID': 99999999999,
         'relationship': 'Ortholog'
         }
    ]

    for mouse_k in legacy_mouse:
        mouse_v = legacy_mouse[mouse_k]
        if 'NCBI' in mouse_k:
            ncbi_2_ens.append(
                {'#tax_id': mouse_taxon,
                 'GeneID': int(mouse_k.replace('NCBIGene:', '')),
                 'Ensembl_gene_identifier': mouse_v
                 }
            )
            gene_info.append(
                {'#tax_id': mouse_taxon,
                 'GeneID': int(mouse_k.replace('NCBIGene:', '')),
                 'Symbol': None}
            )
        else:
            mouse_bkbit.append(
                {"category": ["bican:GeneAnnotation"],
                 "source_id": mouse_v,
                 "symbol": mouse_k,
                 "name": None,
                 "in_taxon_label": "Mus musculus"
                 }
            )

    for human_k in legacy_human:
        human_v = legacy_human[human_k]
        if 'NCBI' in human_k:
            ncbi_2_ens.append(
                {'#tax_id': human_taxon,
                 'GeneID': int(human_k.replace('NCBIGene:', '')),
                 'Ensembl_gene_identifier': human_v
                 }
            )
            gene_info.append(
                {'#tax_id': human_taxon,
                 'GeneID': int(human_k.replace('NCBIGene:', '')),
                 'Symbol': None}
            )
        else:
            human_bkbit.append(
                {"category": ["bican:GeneAnnotation"],
                 "source_id": human_v,
                 "symbol": human_k,
                 "name": None,
                 "in_taxon_label": "Homo sapiens"
                 }
            )

    gene2ensembl_path = f'{tmp_dir}/gene2ensembl.gz'
    pd.DataFrame(ncbi_2_ens).to_csv(
        gene2ensembl_path,
        index=False,
        compression='gzip',
        sep='\t'
    )

    gene_info_path = f'{tmp_dir}/gene_info.gz'
    pd.DataFrame(gene_info).to_csv(
        gene_info_path,
        index=False,
        compression='gzip',
        sep='\t'
    )

    gene_orthologs_path = f'{tmp_dir}/gene_orthologs.gz'
    pd.DataFrame(gene_orthologs).to_csv(
        gene_orthologs_path,
        index=False,
        compression='gzip',
        sep='\t'
    )

    human_bkbit_path = f'{tmp_dir}/human_bkbit.jsonld'
    with open(human_bkbit_path, 'w') as dst:
        dst.write(
            json.dumps(
                {'@graph': human_bkbit},
                indent=2
            )
        )

    mouse_bkbit_path = f'{tmp_dir}/mouse_bkbit.jsonld'
    with open(mouse_bkbit_path, 'w') as dst:
        dst.write(
            json.dumps(
                {'@graph': mouse_bkbit},
                indent=2
            )
        )

    return {
        'gene_info': gene_info_path,
        'gene2ensembl': gene2ensembl_path,
        'gene_orthologs': gene_orthologs_path,
        'human_bkbit': human_bkbit_path,
        'mouse_bkbit': mouse_bkbit_path
    }


@pytest.fixture(scope='session')
def legacy_species_file_fixture(
        legacy_gene_mapper_tmp_dir_fixture):
    """
    Create a tsv simulating names.dmp. Put it into a tarfile.
    Create the text file with that tarfile's md5 hash.
    Return the path to the tarfile and the text file.
    This is meant to simulate downloading the taxons from NCBI.
    """
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=legacy_gene_mapper_tmp_dir_fixture,
            prefix='legacy_species_simulation_'
        )
    )

    name_path = tmp_dir / 'names.dmp'
    with open(name_path, 'w') as dst:
        for species_name, species_id in [('human', 9606),
                                         ('mouse', 10090),
                                         ('Homo sapiens', 9606),
                                         ('Mus musculus', 10090)]:
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
def legacy_dummy_download_mgr_fixture(
        legacy_species_file_fixture,
        legacy_gene_mapping_files):
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
                    'local_path': legacy_species_file_fixture['tar']
                }
            elif src_path.endswith('new_taxdump.tar.gz.md5'):
                return {
                    'local_path': legacy_species_file_fixture['hash']
                }
            elif src_path.endswith('gene_info.gz'):
                return {
                    'local_path': legacy_gene_mapping_files['gene_info']
                }
            elif src_path.endswith('gene2ensembl.gz'):
                return {
                    'local_path': legacy_gene_mapping_files['gene2ensembl']
                }
            elif src_path.endswith('gene_orthologs.gz'):
                return {
                    'local_path': legacy_gene_mapping_files['gene_orthologs']
                }
            else:
                raise RuntimeError(
                    "MockDownloadManager cannot handle src_path "
                    f"{src_path}"
                )

    return DummyDownloadManager


@pytest.fixture(scope='session')
def legacy_gene_mapper_fixture(
        legacy_gene_mapping_files,
        legacy_dummy_download_mgr_fixture,
        legacy_gene_mapper_tmp_dir_fixture):
    """
    Return an instantiation of the MMCGeneMapper class
    based on our simulated NCBI file package
    """
    tmp_dir = tempfile.mkdtemp(dir=legacy_gene_mapper_tmp_dir_fixture)
    db_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='ncbi_gene_mapper_',
        suffix='.db',
        delete=True
    )

    file_spec = [
        {'type': 'bkbit',
         'path': legacy_gene_mapping_files[k]}
        for k in ('human_bkbit', 'mouse_bkbit')
    ]

    to_replace = "mmc_gene_mapper.download.download_manager.DownloadManager"
    with unittest.mock.patch(to_replace, legacy_dummy_download_mgr_fixture):
        gene_mapper = mapper.MMCGeneMapper.create_mapper(
            db_path=db_path,
            local_dir=legacy_gene_mapper_tmp_dir_fixture,
            data_file_spec=file_spec,
            clobber=False,
            force_download=False,
            suppress_download_stdout=True
        )

    return gene_mapper


@pytest.fixture(scope='session')
def legacy_gene_mapper_db_path_fixture(
        legacy_gene_mapper_fixture):
    return str(legacy_gene_mapper_fixture.db_path)
