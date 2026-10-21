#!/usr/bin/env python3
"""
NON-FUNCTIONAL CONCEPT MODEL of a WEARABLE magnetic-stimulation (TMS) headset.

Outer shells only: a head band and over-head strap, a figure-8 coil SHELL at the back of the head,
and a neck collar with an EMPTY electronics bay. The bay is a placeholder box that shows where
a power, control and stimulator unit would have to fit. It contains NOTHING: no electronics, no
battery, no capacitors, no coil windings, no wiring. This is not an engineering design and it
cannot stimulate anything. Real stimulation equipment is a regulated medical device that must be
designed, built and used by qualified people under clinical supervision.

Units: millimetres. x = person's right, y = forward (the person faces +y), z = up.
Run:  python3 make_tms_wearable.py        (needs make_tms_concept.py next to it)
Outputs: tms_wearable_concept.stl / .scad / tms_wearable_preview.png
"""
import math
import sys

import numpy as np

from make_tms_concept import (C_BLACK, C_BLUE, C_DARK, C_FIGURE, C_FRAME, C_GREY, C_LIGHT, C_TEAL,
                              C_WHITE, OUT, part_mesh, signed_volume, write_preview, write_scad,
                              write_stl)

HEAD_R = 95
COIL_WING_R = 40
COIL_T = 22


def build_wearable():
    parts = []

    def box(name, center, size, color, rot=(0, 0, 0)):
        parts.append(dict(kind="box", name=name, center=center, size=size, rot=rot, color=color))

    def bar(name, p0, p1, r, color):
        parts.append(dict(kind="bar", name=name, p0=p0, p1=p1, r=r, color=color))

    def ball(name, center, r, color):
        parts.append(dict(kind="sphere", name=name, center=center, r=r, color=color))

    # ---- scale figure: head, neck, shoulders
    ball("head", (0, 0, 0), HEAD_R, C_FIGURE)
    bar("neck", (0, -5, -70), (0, -5, -200), 48, C_FIGURE)
    box("shoulders", (0, -5, -250), (430, 200, 100), C_FIGURE)

    # ---- head band (ring around the head, a little above the ears)
    ring_r, ring_z, n = HEAD_R + 14, 35, 28
    pts = [(ring_r * math.sin(2 * math.pi * i / n), ring_r * math.cos(2 * math.pi * i / n), ring_z)
           for i in range(n + 1)]
    for a, b in zip(pts[:-1], pts[1:]):
        bar("head band", a, b, 9, C_BLUE)
    # padded front plate on the forehead
    box("forehead pad", (0, ring_r + 4, ring_z), (90, 14, 34), C_TEAL)

    # ---- over-head strap from ear to ear (keeps the band from sliding down)
    arc_r, m = HEAD_R + 14, 18
    arc = [(arc_r * math.cos(math.pi * i / m), 0, arc_r * math.sin(math.pi * i / m) * 1.0)
           for i in range(m + 1)]
    arc = [(x, 0, z + 0) for x, _, z in arc]
    for a, b in zip(arc[:-1], arc[1:]):
        bar("over-head strap", a, b, 7, C_BLUE)

    # ---- occipital coil shell (solid, no windings): figure-8, face toward the head
    face_y = -(HEAD_R + 5)
    cy = face_y - COIL_T / 2
    cz = -10
    for sx in (-1, 1):
        bar("coil wing", (sx * COIL_WING_R, cy - COIL_T / 2, cz), (sx * COIL_WING_R, cy + COIL_T / 2, cz),
            COIL_WING_R, C_DARK)
    box("coil bridge", (0, cy - COIL_T / 2 - 5, cz), (2 * COIL_WING_R, 10, 30), C_DARK)
    # two brackets from the head band to the coil shell
    for sx in (-1, 1):
        bar("coil bracket", (sx * 55, -(ring_r - 5) + 0, ring_z), (sx * 38, cy - COIL_T / 2 - 8, cz + 22), 7, C_FRAME)

    # ---- neck collar with an EMPTY electronics bay
    box("electronics bay (EMPTY shell: nothing inside)", (0, -140, -135), (200, 80, 75), C_WHITE)
    box("bay lid line", (0, -140, -135 + 38), (204, 84, 4), C_GREY)
    for sx in (-1, 1):
        # collar arm wraps around the neck to a pod in front
        bar("collar arm back", (sx * 95, -140, -135), (sx * 112, -60, -150), 14, C_LIGHT)
        bar("collar arm side", (sx * 112, -60, -150), (sx * 112, 40, -150), 14, C_LIGHT)
        box("front pod (EMPTY shell)", (sx * 112, 62, -150), (46, 70, 46), C_WHITE)
    # empty conduit from the bay up to the coil shell (a cover for wiring, nothing in it)
    bar("conduit (empty)", (0, -150, -100), (0, cy - COIL_T / 2 - 12, cz - 15), 11, C_BLACK)
    return parts


def main():
    parts = build_wearable()
    meshes = [part_mesh(q) for q in parts]
    bad = [q["name"] for q, m in zip(parts, meshes) if signed_volume(m) <= 0]
    if bad:
        print("mesh check failed:", bad)
        sys.exit(1)
    n = write_stl(OUT / "tms_wearable_concept.stl", meshes)
    write_scad(OUT / "tms_wearable_concept.scad", parts, "a WEARABLE magnetic-stimulation (TMS) headset")
    write_preview(
        OUT / "tms_wearable_preview.png", parts, meshes, zoom_center=(0, -120, -40), zoom_half=190,
        suptitle="WEARABLE TMS headset: NON-FUNCTIONAL concept model (shells only, electronics bay is EMPTY)",
        titles=["3/4 view from behind", "side view (faces right)", "back view",
                "close-up: coil shell and electronics bay"])
    pts = np.concatenate(meshes).reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    print(f"{len(parts)} parts, {n} triangles")
    print("bounding box (mm): x %.0f..%.0f  y %.0f..%.0f  z %.0f..%.0f" % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))


if __name__ == "__main__":
    main()
