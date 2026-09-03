<!-- provenance: confirmed (verified from envelope.py + sanitize.py source) -->
<!-- verified: 2026-09-03 · envelope.py / sanitize.py source read @add-readme -->
# The half-open envelope model + single-still keyframe markers

The four inject markers describe a half-open envelope in clip-frame time. The hold region is
`[start_keyframes, end_keyframes)` — **start-inclusive, end-exclusive** — so the last held clip
frame is `end_keyframes - 1` (`envelope.py:113-116`, `_denoise_at_frame_time`). The fade-in 1.0
anchor sits one virtual frame before `start_fade_in`, at `start_fade_in - 1`; `end_fade_out` is the
EXCLUSIVE upper bound where denoise returns to 1.0, so the last content frame is `end_fade_out - 1`.

`sanitize.py:497` validates `end_fade_out <= source_length` (exclusive upper bound). For a single
still `source_length == 1`, so `end_fade_out = 1 = source_length` is the exact allowed maximum.

## Correct markers for a SINGLE still keyframe (`source_length == 1`)

```
start_fade_in = 0, start_keyframes = 0, end_keyframes = 1, end_fade_out = 1
interpolation_type = "none"
```

- To make the hold actually CONTAIN frame 0, you need `end_keyframes = start_keyframes + 1`. With
  `0/0/1/1` the hold is `[0, 1)` = exactly frame 0. This is the honest expression of the half-open
  model.
- `end_fade_out = 1` is valid because it equals `source_length` (the exclusive upper bound at
  `sanitize.py:497`).
- With a single frame there is no fade span, so `interpolation_type = "none"` is the honest choice;
  a ramp curve is meaningless over a single point.

## Why "set all four equal" (e.g. `0/0/0/0`) is WRONG

This was a README error, now fixed. Setting all four markers equal leaves an EMPTY hold region
`[0, 0)`. The frame still lands at `min_denoise`, but only by coincidence: it is the endpoint of
the degenerate fade-in ramp (anchor `= start_fade_in - 1 = -1`, and `evaluate_curve(1.0) == 1.0`
for every curve — `envelope.py:41-85`). It yields the same denoise value but does NOT represent a
real hold and does not reflect the half-open model. Prefer `0/0/1/1` + `none`.

(`envelope.py` also has an explicit degenerate branch for all-four-equal that returns a single
`(row, min_denoise)` entry — `envelope.py:206-211` — so `0/0/0/0` does not error; it is just the
dishonest spelling.)
