"""Matplotlib styles for jbubble figures, in a light and a dark theme.

The package ships two Matplotlib style sheets as package data. To use one,
pass its dotted name to `matplotlib.pyplot.style.use`:

```python
import matplotlib.pyplot as plt

plt.style.use("jbubble.style.light")  # or "jbubble.style.dark"
```

Matplotlib 3.7 and later find a style sheet inside an importable package, so
you need no file paths. Neither theme changes anything outside Matplotlib's
`rcParams`, and this package doesn't import Matplotlib, so jbubble keeps
Matplotlib an optional dependency (the `examples` extra).

Both themes draw on a solid background (`#ffffff` for light, `#0d1117` for
dark), hide the top and right spines, and add a hairline grid. The colour
cycle has eight categorical slots, chosen for contrast against each
background and for colour-vision deficiencies:

| Slot | Name    | Light     | Dark      | Typical role                 |
|------|---------|-----------|-----------|------------------------------|
| `C0` | azure   | `#0d77ca` | `#3e96ea` | the bubble radius            |
| `C1` | coral   | `#e35a2d` | `#e6653c` | the first comparison         |
| `C2` | teal    | `#129483` | `#1ea28f` | the second comparison        |
| `C3` | plum    | `#9f1169` | `#b51c79` | the third comparison         |
| `C4` | green   | `#438f31` | `#4f9b3e` |                              |
| `C5` | violet  | `#7447c8` | `#8f6ce0` |                              |
| `C6` | amber   | `#da9516` | `#c0851f` | below 3:1 contrast on white  |
| `C7` | crimson | `#af2938` | `#c83846` |                              |

Every pair among the first four slots stays distinguishable in both themes.
With more than four series on one set of axes, add direct labels, line
styles, or small multiples, or use a sequential colour map for ordered data
(`Blues` in the light theme and `Blues_r` in the dark theme, which are the
image defaults).
Draw the acoustic drive in the neutral grey `#8c959f` (light) or `#6e7681`
(dark), ideally on its own axes.

The style sheets are cosmetic. Their colours and sizes can change in any
release and aren't covered by jbubble's versioning policy.
"""

# The styles are package data; the package has no Python names.
__all__: list[str] = []
