"""Scientific, layout, and artifact checks for the released figure set."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
from PIL import Image
from pypdf import PdfReader

from . import figure1, figure2, figure3, figure4
from . import figure_common as common


ARTWORK = common.FIGURE_DIR / "artwork"


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = common.load_core_sources()
        cls.position = cls.data["position"]
        cls.paired = cls.data["paired"]
        cls.transport = cls.data["transport"]

    def test_source_validation(self) -> None:
        validation = common.validate_sources(self.data)
        self.assertTrue(validation["passed"], validation["failures"])

    def test_figure1_layout_and_semantics(self) -> None:
        with np.load(common.MORPHOLOGY_SOURCE, allow_pickle=False) as arrays:
            figure = figure1.build_figure(
                self.position, self.data["morphology_manifest"], arrays
            )
        try:
            headings = {
                text.get_gid(): text
                for text in figure.texts
                if (text.get_gid() or "").startswith("descriptor-")
            }
            self.assertEqual(
                {text.get_text() for text in headings.values()},
                {
                    r"Near, $s_c=18.5$: remote",
                    r"Outboard, $s_c=34.5$: local",
                },
            )
            labels = {
                text.get_gid(): text
                for text in figure.texts
                if text.get_gid() in {f"panel-label-{letter}" for letter in "abcd"}
            }
            self.assertEqual(set(labels), {f"panel-label-{letter}" for letter in "abcd"})
            self.assertEqual(
                labels["panel-label-a"].get_position()[0],
                labels["panel-label-c"].get_position()[0],
            )
            self.assertEqual(
                labels["panel-label-b"].get_position()[0],
                labels["panel-label-d"].get_position()[0],
            )
            for axis in figure.axes[2:]:
                self.assertFalse(
                    any(
                        token in (line.get_gid() or "")
                        for line in axis.lines
                        for token in ("collar", "bracket")
                    )
                )
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for label_gid, heading_gid in (
                ("panel-label-c", "descriptor-near"),
                ("panel-label-d", "descriptor-outboard"),
            ):
                label_box = labels[label_gid].get_window_extent(renderer=renderer)
                heading_box = headings[heading_gid].get_window_extent(renderer=renderer)
                self.assertGreater(heading_box.x0 - label_box.x1, 4.0)
        finally:
            plt.close(figure)

    def test_figure2_roles_spacing_and_band(self) -> None:
        figure = figure2.build_figure(self.paired)
        try:
            labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
            self.assertEqual(
                labels,
                ["A (exploratory)", "B (confirmation)", "C (confirmation)"],
            )
            self.assertEqual(figure.axes[0].get_ylim(), (-300.0, 900.0))
            band_label = next(
                text for text in figure.axes[1].texts if text.get_gid() == "junction-adjacent-label"
            )
            self.assertEqual(band_label.get_text(), "junction-adjacent")
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            top = figure.axes[0].get_window_extent(renderer=renderer)
            bottom = figure.axes[1].get_window_extent(renderer=renderer)
            self.assertGreater(top.y0 - bottom.y1, 20.0)
        finally:
            plt.close(figure)

    def test_figure3_implemented_geometry_and_values(self) -> None:
        figure = figure3.build_figure(self.transport)
        try:
            labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
            self.assertIn("farther-out 38.5", labels)
            self.assertNotIn("outboard 38.5", labels)
            diagram = figure.axes[1]
            controls = [patch for patch in diagram.patches if "control-volume" in (patch.get_gid() or "")]
            normals = [patch for patch in diagram.patches if (patch.get_gid() or "").startswith("outward-")]
            self.assertEqual(len(controls), 4)
            self.assertEqual(len(normals), 4)
            key_text = " ".join(text.get_text() for text in figure.axes[2].texts)
            self.assertIn("axial faces (2)", key_text)
            self.assertIn("transverse crop\nfaces (2)", key_text)
            self.assertIn("full periodic $x$ span: net 0", key_text)
        finally:
            plt.close(figure)

    def test_figure4_layout_and_values(self) -> None:
        data = figure4.load_and_validate()
        figure = figure4.build_figure(data, common.COMPACT_WIDTH_MM)
        try:
            self.assertEqual(len(data["contrast"]), 8)
            self.assertEqual(
                sorted(
                    float(row["paired_shift"])
                    for row in data["contrast"]
                    if row["m_min"] == "0.3"
                ),
                [-60.0, -60.0, 200.0, 200.0],
            )
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            top = figure.axes[0].get_window_extent(renderer=renderer)
            bottom = figure.axes[1].get_window_extent(renderer=renderer)
            self.assertGreater(top.y0 - bottom.y1, 20.0)
            legends = [
                child
                for axis in figure.axes
                for child in axis.get_children()
                if isinstance(child, Legend)
            ]
            self.assertEqual(len(legends), 3)
        finally:
            plt.close(figure)

    def test_artifact_manifests_and_physical_widths(self) -> None:
        records = [
            (ARTWORK / "main" / f"figure{index}_manifest.json", "compact_outputs", "full_width_outputs")
            for index in range(1, 5)
        ]
        for manifest_path, compact_key, full_key in records:
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest["schema_version"], 1)
            for key, expected_width in ((compact_key, 137.0), (full_key, 190.0)):
                outputs = manifest[key]
                self.assertEqual(len(outputs), 3)
                for output in outputs:
                    candidate = common.REPOSITORY_ROOT / output["path"]
                    self.assertTrue(candidate.is_file(), candidate)
                    self.assertEqual(output["sha256"], common.sha256(candidate))
                    if candidate.suffix == ".pdf":
                        page = PdfReader(str(candidate)).pages[0]
                        width_mm = float(page.mediabox.width) / 72.0 * 25.4
                        self.assertAlmostEqual(width_mm, expected_width, delta=0.03)

    def test_graphical_abstract(self) -> None:
        generated = ARTWORK / "graphical_abstract"
        manifest = load_manifest(generated / "graphical_abstract_manifest.json")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertFalse(manifest["science_values_changed"])
        with Image.open(generated / "graphical_abstract.png") as image:
            self.assertEqual(image.size, (3000, 1200))
        with Image.open(generated / "graphical_abstract_500x200.png") as image:
            self.assertEqual(image.size, (500, 200))
        svg = (generated / "graphical_abstract.svg").read_text(encoding="utf-8")
        visible = " ".join(re.findall(r"<text[^>]*>(.*?)</text>", svg)).upper()
        for phrase in ("INTERMEDIATE", "OUTBOARD", "RESPONSE REVERSES", "JUNCTION-ADJACENT"):
            self.assertIn(phrase, visible)


if __name__ == "__main__":
    unittest.main()
