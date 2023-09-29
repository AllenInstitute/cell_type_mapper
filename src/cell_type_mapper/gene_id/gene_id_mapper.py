import json
import warnings

import cell_type_mapper.utils.utils as utils

from cell_type_mapper.gene_id.utils import (
    is_ensembl)

from cell_type_mapper.data.gene_id_lookup import (
    gene_id_lookup)


class GeneIdMapper(object):

    def __init__(
            self,
            data,
            log=None):
        """
        data is a dict. The keys are non-canonical gene identifiers
        (i.e. gene symbols or NCBI identifiers). The values are the
        corresponding EnsemblIDs.
        """
        self.random_name_generator = RandomNameGenerator()
        self._lookup = data
        self.log = log
        self._preferred_type = "EnsemblID"

    @classmethod
    def from_default(cls, log=None):
        return cls(data=gene_id_lookup, log=log)

    @property
    def preferred_type(self):
        return self._preferred_type

    def map_gene_identifiers(
            self,
            gene_id_list,
            strict=False):
        """
        Take a list of gene identifiers. Find the set of IDs in this mapper
        that best match the list. If that set is the preferred set of gene
        identifiers, just return this list. If not, map from the current
        gene identification scheme to the preferred gene_identifiers, adding
        nonsense names for the genes that do not map.

        If strict == True, raise an error if there are genes that
        cannot be mapped.
        """
        if len(gene_id_list) == 0:
            return []

        if strict:
            bad_genes = []

        mapped_genes = 0
        unmappable_genes = 0
        already_fine = 0
        output = []
        for input_gene in gene_id_list:
            if is_ensembl(input_gene):
                output.append(input_gene)
                already_fine += 1
            else:
                if input_gene in self._lookup:
                    output.append(self._lookup[input_gene])
                    mapped_genes += 1
                else:
                    output.append(self.random_name_generator.name())
                    unmappable_genes += 1
                    if strict:
                        bad_genes.append(input_gene)

        if mapped_genes + unmappable_genes > 0:
            msg = "Not all of your gene identifiers were "
            msg += f"{self.preferred_type}; "
            msg += f"{mapped_genes} were mapped to {self.preferred_type}"
            if already_fine > 0:
                msg += f"; {already_fine} "
                msg += f"were already {self.preferred_type}"
            if unmappable_genes > 0:
                msg += f"; {unmappable_genes} "
                msg += f"could not be mapped to {self.preferred_type}"
            if unmappable_genes > 0 and strict:
                msg += "\nunmappable genes were\n"
                msg += f"{json.dumps(bad_genes,indent=2)}"
                if self.log is not None:
                    self.log.error(msg)
                else:
                    raise RuntimeError(msg)
            else:
                if self.log is not None:
                    self.log.warn(msg)
                else:
                    warnings.warn(msg)
        return output


class RandomNameGenerator(object):
    """
    Class to consistently assign a nonsense name to genes that do not map
    """
    def __init__(self):
        self.ct = 0

    def name(self):
        """
        Get a locally unique nonsense name
        """
        name = f"unmapped_{self.ct}_{utils.get_timestamp()}"
        self.ct += 1
        return name
