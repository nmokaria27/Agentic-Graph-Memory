import json

from evaluation.musique.utils import load_musique_json, select_pilot_examples, write_context_corpus


def test_load_musique_json_handles_paragraph_schema(tmp_path) -> None:
    source = tmp_path / "musique.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "m1",
                "question": "Who directed the film starring Example Actor?",
                "answer": "Example Director",
                "paragraphs": [
                    {
                        "title": "Example Film",
                        "paragraph_text": "Example Actor starred in Example Film.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Example Director",
                        "paragraph_text": "Example Film was directed by Example Director.",
                        "is_supporting": True,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_musique_json(source)

    assert examples[0].question_id == "m1"
    assert examples[0].answer == "Example Director"
    assert len(examples[0].contexts) == 2
    assert examples[0].supporting_facts == [["Example Film", 0], ["Example Director", 0]]


def test_select_and_write_musique_pilot(tmp_path) -> None:
    source = tmp_path / "musique.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "m1",
                    "question": "Q?",
                    "answer": "A",
                    "paragraphs": [
                        {"title": "One", "text": "One text."},
                        {"title": "Two", "text": "Two text."},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    examples = select_pilot_examples(load_musique_json(source), 1)
    manifest = write_context_corpus(examples, tmp_path / "docs")

    assert len(examples) == 1
    assert len(manifest) == 2
    assert (tmp_path / "docs" / "one.txt").exists()
