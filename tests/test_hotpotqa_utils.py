from pathlib import Path

from evaluation.hotpotqa.utils import (
    HotpotExample,
    answer_exact_match,
    answer_token_f1,
    clean_prediction_for_scoring,
    context_text,
    lexical_retrieve,
    select_pilot_examples,
    write_context_corpus,
)


def test_hotpotqa_answer_metrics_normalize_articles_and_punctuation():
    assert answer_exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert answer_exact_match("Eiffel tower is in Paris", "Eiffel Tower") == 0.0
    assert round(answer_token_f1("Eiffel tower Paris", "The Eiffel Tower"), 3) == 0.8


def test_clean_prediction_for_scoring_handles_graphrag_citations_and_yes_no():
    prediction = "Yes, Scott Derrickson and Ed Wood were both American [Data: Sources (171, 70)]."

    assert clean_prediction_for_scoring(prediction, "yes") == "yes"


def test_clean_prediction_for_scoring_strips_answer_prefixes():
    prediction = "The answer is Chief of Protocol."

    assert clean_prediction_for_scoring(prediction, "Chief of Protocol") == "Chief of Protocol"


def test_select_pilot_examples_prefers_non_empty_short_answer_examples():
    examples = [
        HotpotExample(
            question_id="bad",
            question="Bad?",
            answer="",
            contexts=[{"title": "A", "sentences": ["text"]}],
            supporting_facts=[],
        ),
        HotpotExample(
            question_id="ok",
            question="Where?",
            answer="Paris",
            contexts=[{"title": "B", "sentences": ["Paris is in France."]}],
            supporting_facts=[],
        ),
    ]

    selected = select_pilot_examples(examples, n=1)

    assert [example.question_id for example in selected] == ["ok"]


def test_write_context_corpus_deduplicates_by_title(tmp_path: Path):
    examples = [
        HotpotExample(
            question_id="q1",
            question="Q1?",
            answer="A",
            contexts=[
                {"title": "Doc One", "sentences": ["First sentence.", "Second sentence."]},
                {"title": "Doc One", "sentences": ["First sentence.", "Second sentence."]},
            ],
            supporting_facts=[],
        )
    ]

    manifest = write_context_corpus(examples, tmp_path)

    assert len(manifest) == 1
    assert (tmp_path / "doc_one.txt").read_text(encoding="utf-8") == "First sentence. Second sentence."


def test_lexical_retrieve_ranks_matching_contexts_first():
    contexts = [
        {"title": "A", "sentences": ["Paris is the capital of France."]},
        {"title": "B", "sentences": ["Saturn has rings."]},
    ]

    retrieved = lexical_retrieve("What is the capital of France?", contexts, k=1)

    assert retrieved[0]["title"] == "A"
    assert context_text(retrieved[0]) == "Paris is the capital of France."
