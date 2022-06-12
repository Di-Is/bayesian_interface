import unittest


class TestAutoCorrBase(unittest.TestCase):
    def __init__(self, test_class):
        super().__init__()
        self._test_class = test_class
