import unittest

from HMM import HMM


class TestHMM(unittest.TestCase):
    def test_load(self):
        hmm = HMM()
        hmm.load("cat")
        self.assertIn("happy", hmm.transitions)
        self.assertIn("meow", hmm.emissions["happy"])
        self.assertAlmostEqual(hmm.transitions["#"]["happy"], 0.5)
        self.assertAlmostEqual(hmm.emissions["happy"]["meow"], 0.3)

    def test_forward(self):
        hmm = HMM()
        hmm.load("cat")
        most_probable_state, prob = hmm.forward(["meow", "purr", "silent"])
        self.assertEqual(most_probable_state, "happy")

    def test_viterbi(self):
        hmm = HMM()
        hmm.load("cat")
        states, prob = hmm.viterbi(["meow", "purr", "silent"])
        self.assertEqual(states, ["grumpy", "happy", "happy"])


if __name__ == '__main__':
    unittest.main()
