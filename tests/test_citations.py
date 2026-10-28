from docresearch.citations import link_citations

PASSAGES = [
    {"id": "S1", "source": "paper1.pdf", "page": 4},
    {"id": "S2", "source": "report.pdf", "page": 12},
]


def test_tags_become_file_and_page():
    text, used = link_citations("RAG keeps the model frozen [S1].", PASSAGES)
    assert text == "RAG keeps the model frozen [paper1.pdf, p.4]."
    assert used == ["S1"]


def test_multiple_tags_in_one_bracket():
    text, used = link_citations("Both agree [S1, S2].", PASSAGES)
    assert "paper1.pdf, p.4; report.pdf, p.12" in text
    assert used == ["S1", "S2"]


def test_invented_tags_are_dropped():
    text, used = link_citations("Made up claim [S9].", PASSAGES)
    assert "[S9]" not in text
    assert used == []


def test_order_of_first_use():
    _, used = link_citations("B [S2] then A [S1] then B again [S2].", PASSAGES)
    assert used == ["S2", "S1"]
