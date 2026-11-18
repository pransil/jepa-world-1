# Terrain description: x, y, z axes, z = h(x) now, z = h(x, y) later

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §3.1
- **Status:** accepted

## Question
The docs described the same terrain three ways. README called it "1-D terrain", CLAUDE.md called
it a "2D heightfield", and SPEC §3.1 called it "effectively a 2D profile extruded in depth".
"1-D" reads as a line, which doesn't fit a scene with x and z axes.

## Options
- **Count dimensions:** keep one label ("1-D" or "2-D"). That's short, but it's ambiguous: it
  could count the inputs to h, the x–z plane, or the hfield grid.
- **Name the axes:** use the usual x, y, z with z up, and give the surface as an equation.

## Decision
Name the axes. The world has standard x, y, z axes with z up. In this simple first version the
terrain is constant along y, so its surface is z = h(x) and the ball stays in the x–z plane. Later
versions will use full terrain, z = h(x, y).

## Consequences
- README.md and CLAUDE.md use this wording. The code already matches it: `sim/terrain.py`
  extrudes a profile h(x) along y, and `sim/scene.py` holds the ball to the x–z plane.
- SPEC.md describes v1 correctly (§3.1) but doesn't yet record the later z = h(x, y) plan. Adding
  it there is a SPEC edit, so it's the user's call.
- Moving to z = h(x, y) means new terrain, a free ball in 3D, and a camera that isn't side-on.
  It needs a new `sim.generator_version` and is a separate decision when it happens.
