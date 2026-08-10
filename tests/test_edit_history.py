"""Unit tests for undo/redo history."""

import unittest

import numpy as np

from src.persistence.edit_history import EditHistory, EditSnapshot


class EditHistoryTests(unittest.TestCase):
    def test_undo_redo_round_trip(self) -> None:
        history = EditHistory(limit=10)
        a = EditSnapshot(label="a", mesh_rows=2, mesh_cols=2)
        b = EditSnapshot(label="b", mesh_rows=3, mesh_cols=3)
        c = EditSnapshot(label="c", mesh_rows=4, mesh_cols=4)

        history.push(a)
        history.push(b)
        self.assertTrue(history.can_undo())
        self.assertFalse(history.can_redo())

        restored = history.undo(c)
        self.assertEqual(restored.label, "b")
        self.assertTrue(history.can_redo())

        restored = history.redo(restored)
        self.assertEqual(restored.label, "c")

    def test_slider_coalesce_keeps_baseline(self) -> None:
        history = EditHistory()
        history.push(EditSnapshot(label="scale", coalesce_key="slider:design_scale"))
        history.push(EditSnapshot(label="scale2", coalesce_key="slider:design_scale"))
        self.assertEqual(len(history._undo), 1)
        self.assertEqual(history._undo[0].label, "scale")

    def test_new_push_clears_redo(self) -> None:
        history = EditHistory()
        history.push(EditSnapshot(label="one"))
        history.undo(EditSnapshot(label="two"))
        self.assertTrue(history.can_redo())
        history.push(EditSnapshot(label="three"))
        self.assertFalse(history.can_redo())


if __name__ == "__main__":
    unittest.main()
