"""Tests for the exact product-union initializer arithmetic."""

from __future__ import annotations

import unittest

import numpy as np

from .initializer import (
    EXPECTED_CROSSED_T0_FINGERPRINT,
    EXPECTED_ISOLATED_T0_FINGERPRINT,
    array_fingerprint,
    construct_product_union,
)


class SentinelInitializerTests(unittest.TestCase):
    def test_product_union_is_exact_and_chunk_invariant(self) -> None:
        first = np.array(
            [[0.0, 0.2, 1.0], [0.1, 0.6, 0.9]],
            dtype=np.float64,
        )
        second = np.array(
            [[0.0, 0.4, 1.0, 0.3], [0.2, 0.8, 0.5, 0.1]],
            dtype=np.float64,
        )
        expected_q = (
            first[:, :, None]
            + second[:, None, :]
            - first[:, :, None] * second[:, None, :]
        )
        vapor = 0.02
        solid = 1.01
        expected = vapor + (solid - vapor) * expected_q
        one = construct_product_union(
            first, second, vapor=vapor, solid=solid, chunk_depth=1
        )
        four = construct_product_union(
            first, second, vapor=vapor, solid=solid, chunk_depth=4
        )
        np.testing.assert_array_equal(one, expected)
        np.testing.assert_array_equal(four, expected)
        self.assertEqual(array_fingerprint(one), array_fingerprint(four))

    def test_crossed_fingerprint_is_frozen(self) -> None:
        self.assertEqual(
            EXPECTED_CROSSED_T0_FINGERPRINT,
            "ee793415bb02121d268fa3ba6f40c13ec51007f120672a171cc1fa590d300a6a",
        )

    def test_isolated_fingerprint_is_frozen(self) -> None:
        self.assertEqual(
            EXPECTED_ISOLATED_T0_FINGERPRINT,
            "b09bd99f8d15f35b7b355b4aa134a77410b5a1bed3ddcefbda7a73fc6c556b52",
        )


if __name__ == "__main__":
    unittest.main()
