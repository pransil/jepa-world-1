# Ball constraint: slide-x + slide-z + hinge-y joints

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §3.1 [OPEN]
- **Status:** accepted

## Question
SPEC §3.1 restricts the ball to the x-z plane "using a joint constraint or a narrow channel", and
asks for the method to be confirmed against the installed MuJoCo (4.0.1).

## Options
- **A: three joints:** slide along x, slide along z, and a hinge about y, all at the ball's centre.
  Checked on 4.0.1: nq = 3, y is exactly 0 at all times, and the ball rolls. The joints sit at the
  centre of mass, so the mass matrix is diagonal and free flight is exactly ballistic.
- **B: free joint inside a narrow channel** (walls at y = ±(r + ε)). y drifts slightly, and the
  wall contacts add friction and spurious contact events.

## Decision
A, three joints. It holds the plane exactly with no extra contacts, and its exactly-ballistic free
flight is what the event-consistency tests check against.

## Consequences
- `jepa_world/sim/scene.py` builds the ball body with these three joints. Tests assert nq = 3 and
  that the ball's world y is 0 in every frame.
- No spin about the contact normal is possible, so torsional friction is 0.
