import pytest

from src.corpus import Document, build_chunks, chunk_document, load_corpus
from src.retriever import EmbeddingRetriever, TfidfRetriever


def write_corpus(tmp_path, files: dict[str, str]):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_chunk_ids_are_stable_and_ordered():
    doc = Document("handbook", "\n\n".join(f"Paragraph {i}. " * 20 for i in range(6)))
    first = [c.chunk_id for c in chunk_document(doc)]
    second = [c.chunk_id for c in chunk_document(doc)]

    assert first == second
    assert first[0] == "handbook#0"
    assert first == sorted(first, key=lambda c: int(c.split("#")[1]))


def test_chunks_overlap_so_facts_are_not_cut_in_half():
    doc = Document("d", "\n\n".join(f"Sentence number {i} about fares." * 8
                                    for i in range(5)))
    chunks = chunk_document(doc, target_chars=300, overlap_chars=80)

    assert len(chunks) > 1
    tail = chunks[0].text[-80:]
    assert tail in chunks[1].text


def test_loading_skips_non_text_and_empty_files(tmp_path):
    write_corpus(tmp_path, {
        "a.md": "Real content here.",
        "b.txt": "More content.",
        "c.json": '{"ignored": true}',
        "d.md": "   ",
    })
    names = [d.name for d in load_corpus(tmp_path)]
    assert names == ["a", "b"]


def test_empty_corpus_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="cannot evaluate grounding"):
        load_corpus(tmp_path)


def test_missing_corpus_directory_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


@pytest.mark.parametrize("target,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_parameters_rejected(target, overlap):
    with pytest.raises(ValueError):
        chunk_document(Document("d", "text"), target, overlap)


def test_relevance_floor_returns_nothing_rather_than_noise(chunks):
    """The floor is the point of the retriever, so it gets its own test.

    Without it an unanswerable question still arrives at the model wrapped in
    plausible context, which is the condition that produces a fabrication.
    """
    retriever = TfidfRetriever(chunks, min_score=0.05)
    assert retriever.search("quantum chromodynamics in medieval poetry", k=3) == []


def test_floor_of_zero_returns_everything(chunks):
    retriever = TfidfRetriever(chunks, min_score=0.0)
    assert len(retriever.search("zone", k=3)) == 3


def test_relevant_query_ranks_the_right_chunk_first(chunks):
    retriever = TfidfRetriever(chunks, min_score=0.0)
    assert retriever.search("penalty fare", k=1)[0].chunk.chunk_id == "faq#1"


def test_blank_query_returns_nothing(retriever):
    assert retriever.search("   ", k=3) == []


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_rejected(retriever, k):
    with pytest.raises(ValueError):
        retriever.search("fare", k=k)


def test_empty_chunk_list_rejected():
    with pytest.raises(ValueError, match="at least one chunk"):
        TfidfRetriever([])


@pytest.mark.parametrize("floor", [-0.1, 1.0, 1.5])
def test_invalid_floor_rejected(chunks, floor):
    with pytest.raises(ValueError):
        TfidfRetriever(chunks, min_score=floor)


def test_embedding_retriever_refuses_to_pretend(chunks):
    """Better an explicit NotImplementedError than a silently poor retriever."""
    with pytest.raises(NotImplementedError, match="extension point"):
        EmbeddingRetriever(chunks)


def test_real_corpus_builds(tmp_path):
    from pathlib import Path
    chunks = build_chunks(Path(__file__).resolve().parent.parent / "corpus")
    assert len(chunks) >= 10
    assert {c.document for c in chunks} == {"ticketing", "operations", "staff_handbook"}
