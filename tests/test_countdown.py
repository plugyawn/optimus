import unittest

from randopt_lora_lab.countdown import CountdownExample, load_examples, score_completion


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


if __name__ == "__main__":
    unittest.main()
