# gv4: what a slider step costs on an 1800-view scan

Date: 2026-09-09.  Script: `gv4_timing.py`.  The script takes no arguments and
prints the table below.

## The machine

The numbers were measured in a Linux container with 4 CPUs and 15 GB of
memory, on Python 3.11.15 with matplotlib 3.11.1 and numpy 2.4.6, under the Agg
backend.  There is no GPU and no display.  A laptop, which is what the plan's
gate names, has faster cores than this container, so a step that passes the
gate here passes it there.

## The measurement

The model is a helical cone-beam scan of 1800 views over two turns, with a
detector of 40 rows by 64 channels, a reconstruction of 24 by 24 by 16 voxels,
a source-detector distance of 400 ALU, a source-iso distance of 200 ALU, and
30 ALU of helical travel.  It is built from parameters alone.  No sinogram and
no reconstruction are made, because the viewer reads parameters and nothing
else.

One `set_view` call is one slider step, and the time includes the redraw.  The
redraw is what the user waits for, so a measurement that stopped at the artist
update would not be the gate.

Two figures are timed.  The first uses the partial redraw: the cached
background is restored, the artists that carry the view are drawn, and the
region is blitted.  The second is built with `blit=False` and repaints the
whole figure on every step, which is what the Increment 3 viewer did.

## The numbers

```
1800 views, detector (40, 64), recon (24, 24, 16), backend Agg
helical travel 30.0 ALU, magnification 2.00

step                                       mean ms    max ms
build the model                                1.1
build the scene                                0.1
scene.derived_quantities()                    59.4
scene.trajectory()                             0.3       0.6

build the figure, partial redraw             848.1
set_view step, partial redraw                 41.8      43.7
    against the 100 ms gate                 passes
set_view step with path, partial redraw       38.7      43.2
    against the 100 ms gate                 passes
trajectory toggle on, partial redraw         271.1
trajectory toggle off, partial redraw        272.5

build the figure, full repaint               543.1
set_view step, full repaint                  220.7     234.5
    against the 100 ms gate                  FAILS
set_view step with path, full repaint        222.2     240.2
    against the 100 ms gate                  FAILS
trajectory toggle on, full repaint           230.9
trajectory toggle off, full repaint          234.3
```

## What the numbers say

A slider step costs 42 ms on average and 44 ms at worst, against a gate of
100 ms.  The gate passes with about a factor of two in hand.

Repainting the whole figure costs 221 ms per step, so the partial redraw is
about five times cheaper.  The Increment 3 viewer, which cleared and rebuilt the
panel axes on every step, cost 402 ms on average and 494 ms at worst on this
same model and machine, measured before the change.  Most of that cost is text:
one full repaint renders about 1500 text artists, counting every tick label,
every legend entry, and the twenty-odd lines of the text panel, and the text
panel alone costs more to render than every line and marker of the four drawing
panels together.  The partial redraw skips all of it.

The source path over all 1800 views costs 0.3 ms to compute, because
`GeometryScene.trajectory` works over the whole scan at once instead of
building one scene per view.  Building one scene per view, which is what
Increment 3 did, cost 505 ms.  The path is drawn as one polyline per panel, so
1800 views cost the drawing three artists and not 5400.

A trajectory toggle costs 270 ms, because the path is part of the background
and changes the panel limits, so the whole figure repaints.  A toggle is a
deliberate click and not a drag, so this is not on the gate's path.

Building the figure costs 0.85 s, of which 0.22 s is the first full repaint,
another 0.22 s is a second repaint after the text blocks are measured and
placed, and 0.06 s is `derived_quantities`, whose
`volume_fits_detector` projects the volume's eight corners in every one of the
1800 views one view at a time.  That last one is the one loop over views left
in the scene, and vectorizing it would need the projection written a second
time, so it was left alone.
