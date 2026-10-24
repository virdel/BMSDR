"""Tests for :mod:`src.uv_unwrap`.

Run from the repository root with::

    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import uv_unwrap


CUBE_VERTICES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]

CUBE_FACES = [
    (0, 2, 1), (0, 3, 2),  # bottom
    (4, 5, 6), (4, 6, 7),  # top
    (0, 1, 5), (0, 5, 4),  # front
    (2, 3, 7), (2, 7, 6),  # back
    (1, 2, 6), (1, 6, 5),  # right
    (3, 0, 4), (3, 4, 7),  # left
]


def _signed_area_2d(a, b, c):
    return 0.5 * (
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    )


def _write_input_obj(path, vertices, faces):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("o test\n")
        for x, y, z in vertices:
            handle.write("v %.6f %.6f %.6f\n" % (x, y, z))
        for face in faces:
            handle.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def _parse_uv_obj(path):
    vertices, uvs, faces = [], [], []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append(tuple(float(v) for v in parts[1:4]))
            elif parts[0] == "vt":
                uvs.append(tuple(float(v) for v in parts[1:3]))
            elif parts[0] == "f":
                corners = []
                for token in parts[1:]:
                    position, uv = token.split("/")
                    corners.append((int(position) - 1, int(uv) - 1))
                faces.append(tuple(corners))
    return vertices, uvs, faces


class UVUnwrapTests(unittest.TestCase):
    def test_cube_splits_into_six_charts(self):
        result = uv_unwrap.unwrap(CUBE_VERTICES, CUBE_FACES, angle_limit=66.0)
        self.assertEqual(result.chart_count, 6)
        self.assertEqual(len(result.faces), 12)
        self.assertEqual(len(result.vertices), 24)  # 6 charts x 4 corners

        for u, v in result.uvs:
            self.assertGreaterEqual(u, 0.0)
            self.assertLessEqual(u, 1.0)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # A planar projection inside the seed cone must not flip triangles.
        total_area = 0.0
        for a, b, c in result.faces:
            area = _signed_area_2d(
                result.uvs[a], result.uvs[b], result.uvs[c]
            )
            self.assertGreater(area, 0.0)
            total_area += area
        self.assertLessEqual(total_area, 1.0 + 1e-9)
        self.assertGreater(total_area, 0.1)

    def test_coplanar_faces_share_one_chart(self):
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ]
        faces = [
            (0, 1, 3),
            (1, 4, 3),
            (1, 2, 4),
            (2, 5, 4),
        ]
        result = uv_unwrap.unwrap(vertices, faces, angle_limit=66.0)
        self.assertEqual(result.chart_count, 1)
        self.assertEqual(len(result.vertices), 6)
        self.assertEqual(len(result.faces), 4)

    def test_angle_limit_controls_chart_split(self):
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.8660254, 0.5),  # 30 degrees around the shared x edge
        ]
        faces = [(0, 1, 2), (0, 1, 3)]

        merged = uv_unwrap.unwrap(vertices, faces, angle_limit=66.0)
        self.assertEqual(merged.chart_count, 1)

        split = uv_unwrap.unwrap(vertices, faces, angle_limit=10.0)
        self.assertEqual(split.chart_count, 2)
        self.assertEqual(len(split.vertices), 6)  # shared edge is duplicated

    def test_load_obj_supports_polygons_and_negative_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "quad.obj")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n")
                handle.write("vt 0 0\nvn 0 0 1\n")
                handle.write("f -4/-1/-1 -3/-1/-1 -2/-1/-1 -1/-1/-1\n")
            vertices, faces, name = uv_unwrap.load_obj(path)
            self.assertEqual(len(vertices), 4)
            self.assertEqual(faces, [(0, 1, 2), (0, 2, 3)])
            self.assertIsNone(name)

    def test_generate_uv_writes_obj_and_mtl(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "cube.obj")
            _write_input_obj(source, CUBE_VERTICES, CUBE_FACES)
            obj_path, mtl_path = uv_unwrap.generate_uv(
                source, directory, name="mesh_alpha_uv"
            )

            self.assertTrue(os.path.isfile(obj_path))
            self.assertTrue(os.path.isfile(mtl_path))

            vertices, uvs, faces = _parse_uv_obj(obj_path)
            self.assertEqual(len(faces), 12)
            self.assertEqual(len(vertices), len(uvs))
            for face in faces:
                for position_index, uv_index in face:
                    self.assertLess(position_index, len(vertices))
                    self.assertLess(uv_index, len(uvs))

            with open(mtl_path, "r", encoding="utf-8") as handle:
                mtl = handle.read()
            self.assertIn("newmtl Default_OBJ", mtl)
            self.assertIn("Kd ", mtl)

    def test_cli_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "cube.obj")
            _write_input_obj(source, CUBE_VERTICES, CUBE_FACES)
            output_dir = os.path.join(directory, "out")
            exit_code = uv_unwrap.main([source, output_dir, "--name", "uv"])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "uv.obj")))


if __name__ == "__main__":
    unittest.main()
