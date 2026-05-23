import unittest

from optimus.tasks.countdown import safe_eval_expr
from optimus.tasks.generation import generate_examples


class MakeCountdownDataTests(unittest.TestCase):
    def test_generated_examples_are_unique_and_solved_by_solution(self):
        examples = generate_examples(count=50, seed=7, max_number=20, max_target=250)
        keys = {(tuple(sorted(ex.numbers)), ex.target) for ex in examples}
        self.assertEqual(len(keys), 50)
        for ex in examples:
            self.assertEqual(int(safe_eval_expr(ex.solution)), ex.target)


if __name__ == "__main__":
    unittest.main()
