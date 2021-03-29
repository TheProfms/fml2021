import os
import unittest
from time import time

from main import main


class MainTestCase(unittest.TestCase):
    def test_play(self):
        start_time = time()
        if input("Train?:") == "y":
            main(["play", "--n-rounds", "5000", "--no-gui", "--my-agent", "HAARP", "--train", "1"])
            #main(["play", "--n-rounds", "1000", "--no-gui", "--agents", "HAARP", "--train", "1"])
            #main(["play", "--n-rounds", "1000", "--no-gui", "--agents", "HAARP", "peaceful_agent", "peaceful_agent", "peaceful_agent", "--train", "1"])
        else:
            #main(["play", "--n-rounds", "20", "--agents", "BEST_HAARP", "rule_based_agent", "rule_based_agent", "rule_based_agent"])
            main(["play", "--n-rounds", "1", "--my-agent", "HAARP"])
        # Assert that log exists
        self.assertTrue(os.path.isfile("logs/game.log"))
        # Assert that game log way actually written
        self.assertGreater(os.path.getmtime("logs/game.log"), start_time)

if __name__ == '__main__':
    unittest.main()
