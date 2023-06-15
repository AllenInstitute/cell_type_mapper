import warnings

import hierarchical_mapping.utils.utils as utils

from hierarchical_mapping.data.gene_id_lookup import (
    gene_id_lookup)


class GeneIdMapper(object):

    def __init__(
            self,
            data,
            log=None):
        """
        data is a dict. The keys are gene_identifier (the preferred
        way to refer to genes). Each gene_identifiers maps to a dict
        that lists the other ways to refer to that gene).
        """
        self.random_name_generator = RandomNameGenerator()

        self._preferred_type = "EnsemblID"
        self.preferred_gene_id = set(
            data.keys())

        self.log = log

        # create a reverse lookup between alternative gene IDs and
        # the preferred gene ID
        self.other_gene_id = dict()
        for preferred in self.preferred_gene_id:
            gene = data[preferred]
            for other in gene:
                if other not in self.other_gene_id:
                    self.other_gene_id[other] = dict()
                if gene[other] in self.other_gene_id:
                    raise RuntimeError(
                        f"{other}.{gene[other]} maps to more than one "
                        "gene_identifier: "
                        f"{self.other_gene_id[other][gene[other]]} and "
                        f"{preferred}")
                self.other_gene_id[other][gene[other]] = preferred

    @classmethod
    def from_default(cls, log=None):
        return cls(data=gene_id_lookup, log=log)

    @property
    def preferred_type(self):
        return self._preferred_type

    def map_gene_identifiers(
            self,
            gene_id_list):
        """
        Take a list of gene identifiers. Find the set of IDs in this mapper
        that best match the list. If that set is the preferred set of gene
        identifiers, just return this list. If not, map from the current
        gene identification scheme to the preferred gene_identifiers, adding
        nonsense names for the genes that do not map.
        """
        gene_id_set = set(gene_id_list)
        n_max = len(gene_id_set.intersection(self.preferred_gene_id))
        map_from = None
        for other_id_type in self.other_gene_id:
            this_set = set(self.other_gene_id[other_id_type].keys())
            n_intersection = len(gene_id_set.intersection(this_set))
            if n_intersection > n_max:
                map_from = other_id_type
                n_max = n_intersection

        if n_max == 0:
            raise RuntimeError(
                "You gene identifiers did not match any known schema")

        if map_from is None:
            return gene_id_list

        msg = f"Your gene IDs appear to be of type '{map_from}'\n"
        msg += f"Mapping them to {self.preferred_type}."

        new_id = []
        ct_bad = 0
        for g in gene_id_list:
            if g in self.other_gene_id[map_from]:
                new_id.append(self.other_gene_id[map_from][g])
            else:
                ct_bad += 1
                new_id.append(self.random_name_generator.name())
        if ct_bad > 0:
            msg += f"\n{ct_bad} genes had no mapping."

        if self.log is not None:
            self.log.warn(msg)
        else:
            warnings.warn(msg)

        return new_id


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
        name = f"nonsense_{self.ct}_{utils.get_timestamp()}"
        self.ct += 1
        return name
