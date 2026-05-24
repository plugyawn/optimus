import json
import tempfile
import unittest
from pathlib import Path

from optimus.tasks.countdown import (
    CountdownExample,
    extract_numeric_vote,
    load_examples,
    score_completion,
    unique_semantic_example_count,
    voted_answer_exact,
)
from optimus.serving.transformers import TransformersLoraBackend


class CountdownDataTests(unittest.TestCase):
    def test_load_examples_refuses_silent_repetition(self):
        with self.assertRaisesRegex(ValueError, "only 32 unique examples"):
            load_examples(None, 33, seed=1)

    def test_load_examples_allows_repetition_for_smoke_tests_only(self):
        examples = load_examples(None, 33, seed=1, allow_repeat=True)
        self.assertEqual(len(examples), 33)
        self.assertLess(len({ex.id for ex in examples}), 33)

    def test_load_examples_excludes_ids_before_sampling(self):
        examples = load_examples(None, 8, seed=1, exclude_ids=set(range(24)))
        self.assertEqual(len(examples), 8)
        self.assertTrue(all(ex.id not in set(range(24)) for ex in examples))

    def test_load_examples_refuses_semantic_duplicates(self):
        rows = [
            {"id": 1, "numbers": [1, 2, 3], "target": 6},
            {"id": 2, "numbers": [3, 2, 1], "target": 6},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "duplicate semantic"):
                load_examples(str(path), 1, seed=1)

    def test_unique_semantic_count(self):
        examples = [
            CountdownExample(1, (1, 2, 3), 6),
            CountdownExample(2, (3, 2, 1), 6),
            CountdownExample(3, (3, 2, 1), 7),
        ]
        self.assertEqual(unique_semantic_example_count(examples), 2)


class CountdownParserTests(unittest.TestCase):
    def setUp(self):
        self.example = CountdownExample(1, (3, 7, 8, 8), 24)

    def test_strict_valid_answer(self):
        score = score_completion("<answer>8*(7-3)-8</answer>", self.example)
        self.assertEqual(score["exact"], 1.0)
        self.assertFalse(score["malformed"])
        self.assertEqual(score["answer_count"], 1)

    def test_strict_missing_answer_is_malformed(self):
        score = score_completion("8*(7-3)-8", self.example)
        self.assertEqual(score["exact"], 0.0)
        self.assertTrue(score["malformed"])
        self.assertTrue(score["missing_answer"])

    def test_strict_trailing_text_is_malformed(self):
        score = score_completion("<answer>8*(7-3)-8</answer> done", self.example)
        self.assertEqual(score["exact"], 0.0)
        self.assertTrue(score["malformed"])
        self.assertTrue(score["trailing_text"])

    def test_strict_multiple_answers_is_malformed(self):
        score = score_completion("<answer>1+2</answer><answer>8*(7-3)-8</answer>", self.example)
        self.assertEqual(score["exact"], 0.0)
        self.assertTrue(score["malformed"])
        self.assertTrue(score["multiple_answers"])

    def test_numeric_vote_uses_evaluated_result(self):
        vote = extract_numeric_vote("<answer>8*(7-3)-8</answer>", self.example)
        self.assertTrue(vote["valid_vote"])
        self.assertEqual(vote["vote"], "24")
        self.assertEqual(voted_answer_exact(vote["vote"], self.example), 1.0)

    def test_numeric_vote_rejects_wrong_numbers_even_if_target_matches(self):
        vote = extract_numeric_vote("<answer>24</answer>", self.example)
        self.assertFalse(vote["valid_vote"])
        self.assertEqual(vote["vote_reject"], "wrong_numbers")


class BackendTextNormalizationTests(unittest.TestCase):
    def test_truncate_at_answer_stop(self):
        backend = object.__new__(TransformersLoraBackend)
        backend.answer_stop_text = "</answer>"
        text = backend._truncate_at_answer_stop(" <answer>8*(7-3)-8</answer>Human:")
        self.assertEqual(text, " <answer>8*(7-3)-8</answer>")


if __name__ == "__main__":
    unittest.main()
