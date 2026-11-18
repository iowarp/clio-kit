"""Worked build123d scripts the agent can start from.

Each sets ``result``. The flow frame used by the fluid servers is x along the flow,
y up (lift), z across (span), with lengths in body lengths, so these parts are
drawn that way: a wing section in x-y extruded along z, a car body with its
height along y.
"""

from __future__ import annotations

EXAMPLES: dict[str, dict[str, str]] = {
    "naca_wing_section": {
        "description": (
            "A NACA four-digit wing section (4412: 4 % camber at 40 % chord, 12 % thick), "
            "chord 1 along x, extruded 1 along z. Change m, p, t for other sections."
        ),
        "code": """import math

m, p, t = 0.04, 0.4, 0.12   # NACA 4412
chord, span, n = 1.0, 1.0, 80


def side(sign):
    points = []
    for i in range(n + 1):
        x = (1 - math.cos(math.pi * i / n)) / 2
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                      + 0.2843 * x**3 - 0.1036 * x**4)
        if x < p:
            yc, dy = m / p**2 * (2 * p * x - x * x), 2 * m / p**2 * (p - x)
        else:
            yc = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x * x)
            dy = 2 * m / (1 - p) ** 2 * (p - x)
        th = math.atan(dy)
        points.append((chord * (x - sign * yt * math.sin(th)),
                       chord * (yc + sign * yt * math.cos(th))))
    return points


upper, lower = side(1), side(-1)
with BuildPart() as wing:
    with BuildSketch():
        with BuildLine():
            Spline(*upper[::-1], *lower[1:])   # trailing edge, over the nose, back
            Line(lower[-1], upper[-1])
        make_face()
    extrude(amount=span)
result = wing.part
""",
    },
    "ahmed_car_body": {
        "description": (
            "A simplified Ahmed car body: length 1.044 along x, height 0.288 along y, "
            "width 0.389 along z, rounded nose and a 25 degree rear slant."
        ),
        "code": """import math

L, H, W, r = 1.044, 0.288, 0.389, 0.1
slant = 0.222
drop = slant * math.tan(math.radians(25))
with BuildPart() as car:
    with BuildSketch() as side:
        Polygon((0, 0), (L, 0), (L, H - drop), (L - slant, H), (0, H), align=None)
        fillet(side.vertices().group_by(Axis.X)[0], radius=r)
    extrude(amount=W / 2, both=True)
result = car.part
""",
    },
    "cylinder": {
        "description": (
            "A circular cylinder of diameter 1 spanning z: the classic bluff body, "
            "shedding at St about 0.2 above Re 50."
        ),
        "code": """result = Cylinder(radius=0.5, height=1.0)
""",
    },
    "pipe_bend": {
        "description": (
            "A 90 degree pipe bend: bend radius 1, outer diameter 0.4, wall 0.04, swept "
            "along an arc in x-y. A part for internal flow or for flow around a pipe."
        ),
        "code": """R, d, wall = 1.0, 0.4, 0.04
with BuildPart() as bend:
    with BuildLine() as path:
        CenterArc((0, 0, 0), R, 0, 90)
    with BuildSketch(Plane(origin=path.line @ 0, z_dir=path.line % 0)):
        Circle(d / 2)
        Circle(d / 2 - wall, mode=Mode.SUBTRACT)
    sweep()
result = bend.part
""",
    },
}
