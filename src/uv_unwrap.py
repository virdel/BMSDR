"""Small, dependency-free UV unwrapper for the simplification pipeline.

Why not Blender?
----------------
The original pipeline launched Blender in background mode only to run
``bpy.ops.uv.smart_project``.  That added a ~300 MB dependency and a
hard-coded Windows path for a single operation.  This module implements a
simplified version of the same idea with the Python standard library only:

1. read the geometry of one OBJ file (``v`` / ``f``, polygons are fan
   triangulated);
2. grow charts across edges whose face normals differ by less than
   ``angle_limit`` degrees (all faces of a chart also stay inside that
   cone around the seed face, which guarantees that the planar projection
   does not fold triangles over);
3. project every chart on its area-weighted average plane;
4. shelf-pack the chart bounding boxes into the unit square;
5. write ``<name>.obj`` / ``<name>.mtl`` with one vertex per chart corner.

Compared with ``uv.smart_project`` the result is simpler: strongly curved
charts can still overlap themselves, there is no island rotation search
and no hole filling.  For the differentiable texture fitting done in this
repository that is more than enough, and the unwrap is deterministic and
needs no external executable.

Command line
------------
    python -m src.uv_unwrap input.obj output_dir [--name mesh_alpha_uv]
"""

from __future__ import print_function

import argparse
import math
import os
import sys
from collections import deque

MIN_ANGLE = 1.0
MAX_ANGLE = 89.0
DEFAULT_ANGLE_LIMIT = 66.0
DEFAULT_MARGIN = 0.002
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "latin-1")
EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Small vector helpers (kept local so the module has no dependencies)
# ---------------------------------------------------------------------------

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalized(a):
    length = _length(a)
    if length <= EPSILON:
        return None
    return (a[0] / length, a[1] / length, a[2] / length)


def _triangle_normal(a, b, c):
    return _cross(
        (b[0] - a[0], b[1] - a[1], b[2] - a[2]),
        (c[0] - a[0], c[1] - a[1], c[2] - a[2]),
    )


# ---------------------------------------------------------------------------
# OBJ input
# ---------------------------------------------------------------------------

def _read_lines(path):
    last_error = None
    for encoding in TEXT_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.readlines()
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError("cannot decode %s: %s" % (path, last_error))


def _obj_index(token, count):
    """Convert a (possibly negative) OBJ index to a zero-based index."""
    value = int(token)
    return value - 1 if value > 0 else count + value


def load_obj(path):
    """Read vertices and fan-triangulated faces from an OBJ file.

    Returns ``(vertices, faces, object_name)`` where vertices are
    ``(x, y, z)`` tuples and faces are zero-based index triples.  Texture
    coordinates, normals and materials of the input are intentionally
    ignored: the unwrapper replaces them.
    """
    lines = _read_lines(path)
    vertices = []
    faces = []
    object_name = None

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tag in ("o", "g") and object_name is None and len(parts) > 1:
            object_name = " ".join(parts[1:])
        elif tag == "f" and len(parts) >= 4:
            indices = []
            for token in parts[1:]:
                vertex_token = token.split("/")[0]
                if vertex_token:
                    indices.append(_obj_index(vertex_token, len(vertices)))
            for i in range(1, len(indices) - 1):
                triangle = (indices[0], indices[i], indices[i + 1])
                if len(set(triangle)) == 3:
                    faces.append(triangle)

    if not vertices:
        raise ValueError("%s does not contain any vertices" % path)
    if not faces:
        raise ValueError("%s does not contain any faces" % path)

    vertex_count = len(vertices)
    for triangle in faces:
        for index in triangle:
            if index < 0 or index >= vertex_count:
                raise ValueError(
                    "%s references vertex %d outside [0, %d)"
                    % (path, index, vertex_count)
                )
    return vertices, faces, object_name


# ---------------------------------------------------------------------------
# Chart construction
# ---------------------------------------------------------------------------

class Chart(object):
    """One UV island: a set of triangles and their planar projection."""

    __slots__ = ("faces", "normal", "uvs", "width", "height", "offset", "scale")

    def __init__(self, faces, normal, uvs, width, height):
        self.faces = faces
        self.normal = normal
        self.uvs = uvs
        self.width = width
        self.height = height
        self.offset = (0.0, 0.0)
        self.scale = 1.0


def _build_neighbors(faces):
    """Return face adjacency for edges shared by exactly two triangles."""
    edge_faces = {}
    for face_index, (a, b, c) in enumerate(faces):
        for start, end in ((a, b), (b, c), (c, a)):
            key = (start, end) if start < end else (end, start)
            edge_faces.setdefault(key, []).append(face_index)

    neighbors = [set() for _ in faces]
    for shared in edge_faces.values():
        if len(shared) == 2:
            first, second = shared
            neighbors[first].add(second)
            neighbors[second].add(first)
    return neighbors


def _face_normals_and_areas(vertices, faces):
    normals = []
    areas = []
    for a, b, c in faces:
        normal = _triangle_normal(vertices[a], vertices[b], vertices[c])
        length = _length(normal)
        if length <= EPSILON:
            normals.append((0.0, 0.0, 1.0))
            areas.append(0.0)
        else:
            normals.append(
                (normal[0] / length, normal[1] / length, normal[2] / length)
            )
            areas.append(0.5 * length)
    return normals, areas


def _grow_charts(faces, normals, areas, angle_limit):
    """Greedy region growing over edges with a bounded normal cone.

    A face joins a chart when it touches the chart and both its normal and
    the seed normal are within ``angle_limit``.  Keeping every chart inside
    the seed cone (angle < 90) means the planar projection preserves the
    orientation of every triangle, which avoids folded UV triangles.
    """
    neighbors = _build_neighbors(faces)
    cosine_limit = math.cos(math.radians(angle_limit))
    order = sorted(range(len(faces)), key=lambda index: (-areas[index], index))
    visited = [False] * len(faces)
    charts = []

    for seed in order:
        if visited[seed]:
            continue
        visited[seed] = True
        seed_normal = normals[seed]
        chart_faces = [seed]
        queue = deque([seed])

        while queue:
            current = queue.popleft()
            current_normal = normals[current]
            for neighbor in neighbors[current]:
                if visited[neighbor]:
                    continue
                neighbor_normal = normals[neighbor]
                if _dot(current_normal, neighbor_normal) < cosine_limit:
                    continue
                if _dot(seed_normal, neighbor_normal) < cosine_limit:
                    continue
                visited[neighbor] = True
                chart_faces.append(neighbor)
                queue.append(neighbor)

        charts.append(chart_faces)
    return charts


def _chart_basis(normal):
    """Return an orthonormal (u, v) basis such that u x v = normal."""
    helper = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = _normalized(_cross(helper, normal))
    if u is None:  # pragmatic fallback for a degenerate chart normal
        u = (1.0, 0.0, 0.0)
    v = _cross(normal, u)
    return u, v


def _project_chart(vertices, faces, chart_faces, normal):
    """Orthographically project a chart on its plane and normalise bounds."""
    u_axis, v_axis = _chart_basis(normal)
    uvs = {}
    min_u = min_v = float("inf")
    max_u = max_v = float("-inf")

    for face_index in chart_faces:
        for vertex_index in faces[face_index]:
            if vertex_index in uvs:
                continue
            position = vertices[vertex_index]
            u = _dot(position, u_axis)
            v = _dot(position, v_axis)
            uvs[vertex_index] = (u, v)
            min_u = min(min_u, u)
            max_u = max(max_u, u)
            min_v = min(min_v, v)
            max_v = max(max_v, v)

    if max_u - min_u <= EPSILON or max_v - min_v <= EPSILON:
        # Degenerate chart (zero area projection): give it a tiny box so it
        # can still be packed without special cases later.
        width = max(max_u - min_u, 1e-6)
        height = max(max_v - min_v, 1e-6)
    else:
        width = max_u - min_u
        height = max_v - min_v

    shifted = {}
    for vertex_index, (u, v) in uvs.items():
        shifted[vertex_index] = (u - min_u, v - min_v)
    return shifted, width, height


def _chart_normal(normals, areas, chart_faces):
    nx = ny = nz = 0.0
    for face_index in chart_faces:
        weight = areas[face_index]
        normal = normals[face_index]
        nx += normal[0] * weight
        ny += normal[1] * weight
        nz += normal[2] * weight
    averaged = _normalized((nx, ny, nz))
    if averaged is None:
        return normals[chart_faces[0]]
    return averaged


# ---------------------------------------------------------------------------
# Shelf packing
# ---------------------------------------------------------------------------

def _shelf_pack(boxes, width, gap):
    """Place boxes (w, h) on shelves; returns positions and used extent."""
    order = sorted(
        range(len(boxes)), key=lambda index: (-boxes[index][1], -boxes[index][0], index)
    )
    positions = [None] * len(boxes)
    x = y = row_height = 0.0
    used_width = max((box[0] for box in boxes), default=0.0)

    for index in order:
        box_width, box_height = boxes[index]
        if x > 0.0 and x + box_width > width:
            x = 0.0
            y += row_height + gap
            row_height = 0.0
        positions[index] = (x, y)
        x += box_width + gap
        used_width = max(used_width, x - gap)
        row_height = max(row_height, box_height)

    used_height = y + row_height
    return positions, used_width, used_height


def _pack_charts(charts, margin):
    """Scale and translate chart projections into [0, 1]^2."""
    if not charts:
        return

    padding = max(margin, 1e-4)
    boxes = [(chart.width, chart.height) for chart in charts]
    total_area = sum(width * height for width, height in boxes)
    max_width = max(box[0] for box in boxes)

    scale = 1.0
    positions = None

    # Two outer passes: the first one gives a scale estimate so that the
    # requested margin (expressed in final UV units) can be converted to
    # chart units.  The inner loop searches for the shelf width that makes
    # the packed atlas as square as possible.
    gap = 0.0
    for _ in range(2):
        width = max(max_width, math.sqrt(total_area), 1e-6)
        best = None
        for _ in range(48):
            candidate, used_width, used_height = _shelf_pack(boxes, width, gap)
            extent = max(used_width, used_height)
            if best is None or extent < best[0]:
                best = (extent, candidate, used_width, used_height)
            if used_height <= used_width * 1.02 and used_width <= used_height * 1.02:
                break
            if used_height > used_width:
                if width <= max_width:
                    break
                width *= 1.12
            else:
                width *= 0.88

        extent, positions, used_width, used_height = best
        if extent <= EPSILON:
            break
        scale = (1.0 - 2.0 * padding) / extent
        gap = margin / scale if scale > EPSILON else 0.0

    for chart, position in zip(charts, positions):
        chart.offset = (
            padding + position[0] * scale,
            padding + position[1] * scale,
        )
        chart.scale = scale


# ---------------------------------------------------------------------------
# Public unwrap API
# ---------------------------------------------------------------------------

class UVResult(object):
    """The unwrapped mesh: per-chart duplicated vertices plus UV coordinates."""

    __slots__ = ("vertices", "uvs", "faces", "chart_count")

    def __init__(self, vertices, uvs, faces, chart_count):
        self.vertices = vertices
        self.uvs = uvs
        self.faces = faces
        self.chart_count = chart_count


def unwrap(vertices, faces, angle_limit=DEFAULT_ANGLE_LIMIT, margin=DEFAULT_MARGIN):
    """Unwrap a triangle mesh into a non-overlapping chart atlas.

    ``vertices`` is a sequence of (x, y, z) tuples and ``faces`` a sequence
    of zero-based index triples.  The returned :class:`UVResult` contains a
    new vertex buffer (vertices are duplicated per chart), matching UVs and
    faces indexing the new buffers.
    """
    angle_limit = min(max(float(angle_limit), MIN_ANGLE), MAX_ANGLE)
    normals, areas = _face_normals_and_areas(vertices, faces)
    chart_face_lists = _grow_charts(faces, normals, areas, angle_limit)

    charts = []
    for chart_faces in chart_face_lists:
        normal = _chart_normal(normals, areas, chart_faces)
        uvs, width, height = _project_chart(vertices, faces, chart_faces, normal)
        charts.append(Chart(chart_faces, normal, uvs, width, height))

    _pack_charts(charts, margin)

    out_vertices = []
    out_uvs = []
    out_faces = []

    for chart in charts:
        local_indices = {}
        for vertex_index in sorted(chart.uvs):
            u, v = chart.uvs[vertex_index]
            local_indices[vertex_index] = len(out_vertices)
            out_vertices.append(vertices[vertex_index])
            out_uvs.append(
                (
                    chart.offset[0] + u * chart.scale,
                    chart.offset[1] + v * chart.scale,
                )
            )
        for face_index in chart.faces:
            a, b, c = faces[face_index]
            out_faces.append(
                (local_indices[a], local_indices[b], local_indices[c])
            )

    return UVResult(out_vertices, out_uvs, out_faces, len(charts))


# ---------------------------------------------------------------------------
# OBJ / MTL output
# ---------------------------------------------------------------------------

MTL_TEMPLATE = """# generated by src.uv_unwrap
newmtl Default_OBJ
Ka 1.000000 1.000000 1.000000
Kd 0.800000 0.800000 0.800000
Ks 0.500000 0.500000 0.500000
Ns 250.000000
d 1.000000
illum 2
"""


def write_uv_obj(obj_path, result, object_name=None):
    """Write an unwrapped mesh as OBJ + MTL and return both paths."""
    obj_path = os.path.abspath(obj_path)
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    directory = os.path.dirname(obj_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if object_name is None:
        object_name = os.path.splitext(os.path.basename(obj_path))[0]

    with open(obj_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# UV unwrapped by src.uv_unwrap\n")
        handle.write("mtllib %s\n" % os.path.basename(mtl_path))
        handle.write("o %s\n" % object_name)
        handle.writelines(
            "v %.6f %.6f %.6f\n" % vertex for vertex in result.vertices
        )
        handle.writelines("vt %.6f %.6f\n" % uv for uv in result.uvs)
        handle.write("usemtl Default_OBJ\n")
        handle.write("s 1\n")
        handle.writelines(
            "f %d/%d %d/%d %d/%d\n"
            % (a + 1, a + 1, b + 1, b + 1, c + 1, c + 1)
            for a, b, c in result.faces
        )

    with open(mtl_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(MTL_TEMPLATE)

    return obj_path, mtl_path


def generate_uv(input_obj, output_dir, name="mesh_alpha_uv",
                angle_limit=DEFAULT_ANGLE_LIMIT, margin=DEFAULT_MARGIN):
    """Unwrap ``input_obj`` and write ``<name>.obj`` / ``<name>.mtl``.

    Returns ``(obj_path, mtl_path)``.  This is the drop-in replacement for
    the old ``blender_uv_generating`` helper used by ``train.py``.
    """
    vertices, faces, object_name = load_obj(input_obj)
    result = unwrap(vertices, faces, angle_limit=angle_limit, margin=margin)

    obj_path = os.path.join(output_dir, name + ".obj")
    mtl_path = os.path.join(output_dir, name + ".mtl")
    obj_path, mtl_path = write_uv_obj(obj_path, result, object_name or name)

    print(
        "UV unwrap: %d charts, %d vertices, %d triangles -> %s"
        % (result.chart_count, len(result.vertices), len(result.faces), obj_path)
    )
    return obj_path, mtl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a simple chart-based UV atlas for an OBJ mesh."
    )
    parser.add_argument("input", help="input OBJ file")
    parser.add_argument("output_dir", help="directory for the generated OBJ/MTL")
    parser.add_argument(
        "--name", default="mesh_alpha_uv", help="output file stem (default: mesh_alpha_uv)"
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=DEFAULT_ANGLE_LIMIT,
        help="maximum angle between chart faces in degrees (default: 66)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="gutter between charts in UV units (default: 0.002)",
    )
    args = parser.parse_args(argv)

    obj_path, _ = generate_uv(
        args.input,
        args.output_dir,
        name=args.name,
        angle_limit=args.angle,
        margin=args.margin,
    )
    print(obj_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
