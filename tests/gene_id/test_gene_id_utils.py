import pytest

from cell_type_mapper.gene_id.utils import (
    detect_species)


def test_detect_species():

    assert detect_species([f'garbage_{ii}' for ii in range(4)]) is None

    # ens from both mouse and human present
    bad = ["ENSMUSG00000051951", "ENSMUSG00000033774", "Pcmtd1",
           "Gm16041", "ENSG00000268895"]
    with pytest.raises(RuntimeError,
                       match="There are EnsemblIDs from"):
        detect_species(bad)

    # mouse Ensembl ID but human symbols
    mouse = ["ENSMUSG00000025907", "AADACL2-AS1", "A3GALT2"]
    assert detect_species(mouse) == 'mouse'

    # human Ensembl ID but mouse symbols
    human = ["Pcmtd1", "Gm16041", "ENSG00000081760"]
    assert detect_species(human) == 'human'

    # more mouse symbols than human
    mouse = ["A1BG-AS1", "Atp6v1h", "Tcea1"]
    assert detect_species(mouse) == 'mouse'

    # more human symbols than mouse
    human = ["AACS", "A1CF", "Sntg1"]
    assert detect_species(human) == 'human'

    # 2 mouse and 2 human symbols
    bad = ["AACS", "A1CF", "Pcmtd1", "Gm16041"]
    with pytest.raises(RuntimeError,
                       match="These species"):
        detect_species(bad)

    # mouse ensembl with dots
    mouse_dot = ['ENSMUSG00000051951.5', 'ENSMUSG00000033740.2']
    assert detect_species(mouse_dot) == 'mouse'

    #human ensembl with dots
    human_dot = ['ENSG00000256904.6', 'ENSG00000245105.8']
    assert detect_species(human_dot) == 'human'
