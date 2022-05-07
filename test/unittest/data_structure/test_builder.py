from dataclasses import dataclass

import unittest
from bayesian_interface.data_structure.builder import Director, SaveType, Param
import bayesian_interface.data_structure.structure_parts as parts


class TestDirector(unittest.TestCase):
    def test_1(self):
        @dataclass
        class AA:
            aa: parts.AbsAttr
            bb: parts.AbsArray

        inst = Director.create(
            AA,
            dict(
                aa=Param(kwargs={}, save_type=SaveType.memory),
                bb=Param(kwargs={}, save_type=SaveType.memory),
            ),
        )

    def test_e_1(self):
        @dataclass
        class AA:
            aa: parts.AbsAttr
            bb: parts.AbsArray
            cc: parts.AbsAttr

        with self.assertRaises(TypeError):
            inst = Director.create(
                AA,
                dict(
                    aa=Param(kwargs={}, save_type=SaveType.memory),
                    bb=Param(kwargs={}, save_type=SaveType.memory),
                ),
            )

    def test_e_2(self):
        @dataclass
        class AA:
            aa: parts.AbsAttr
            bb: parts.AbsArray

        with self.assertRaises(ValueError):
            inst = Director.create(
                AA,
                dict(
                    aa=Param(kwargs={}, save_type=SaveType.memory),
                    bb=Param(kwargs={}, save_type=SaveType.memory),
                    cc=Param(kwargs={}, save_type=SaveType.memory),
                ),
            )

    def test_e_3(self):
        @dataclass
        class AA:
            aa: parts.AbsAttr
            bb: int

        with self.assertRaises(TypeError):
            inst = Director.create(
                AA,
                dict(
                    aa=Param(kwargs={}, save_type=SaveType.memory),
                    bb=Param(kwargs={}, save_type=SaveType.memory),
                ),
            )
