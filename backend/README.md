# OpenCV trajectory backend

Browser-only tracking is fragile for golf shots because the ball is often only a
few pixels, and in this sample it becomes a dark dot against the clouds. This
backend reads every decoded frame with OpenCV, automatically estimates the
launch timing, finds the first 5-6 moving ball-like points after impact, then
renders a SmoothSwing-like projected trajectory.

Install:

```bash
python -m pip install -r backend/requirements.txt
```

Automatic example for iPhone 240fps slow-motion samples stored as 30fps
playback:

```bash
python backend/opencv_tracker.py "IMG_5999.mov" ^
  --output "IMG_5999_trajectory.mp4" ^
  --rotate none ^
  --scan-start 10 --scan-end 22 ^
  --diff-threshold 12 ^
  --draw-seconds 1.35 ^
  --projection-seconds 3.65 ^
  --detected-curve-portion 0.44 ^
  --line-delay 0.0001 ^
  --line-draw-speed 0.8 ^
  --fade-start 0.90 ^
  --speed 3.8
```

Notes:

- No tapping is required. `--launch` and `--aim` still exist as optional
  debugging overrides.
- `--scan-start` / `--scan-end` narrow the impact search window. This is useful
  when a long video contains setup time before the swing.
- `--launch-delay` defaults to `0.55` seconds because iPhone slow-motion videos
  often have the largest motion peak just before the ball visibly launches.
- The renderer intentionally uses only the first few detected points plus an
  estimated landing endpoint. This produces a stable visual trajectory even when
  the ball disappears after a few frames.
- The line grows from launch instead of appearing all at once. The detected
  portion is drawn first, then the estimated landing portion fades and narrows
  only at the end of the arc.
- `--line-delay` controls when the trajectory starts after the first detected
  ball point, in exported-video seconds. The tuned value is `0.0001`, which is
  effectively immediate.
- `--line-draw-speed 0.8` draws the trajectory at 80% of the default growth
  speed while keeping the exported video playback speed unchanged.
- `--speed 3.8` converts the provided iPhone 240fps slow-motion footage that is
  stored as 30fps playback back to approximate normal speed. Leave it at `1` to
  export the slow-motion version.
