// Golf ball detection + trajectory tracking.
//
// Pipeline (per analyzed video):
//   1. observe() is called for every frame. It estimates and cancels global
//      camera shake, frame-differences against the (shifted) previous frame,
//      keeps only "ball-coloured" moving pixels, labels them into blobs, and
//      records the total motion energy of the frame.
//   2. resolve() is called once after the scan. It finds the impact moment
//      (peak motion energy), then — starting from the tapped launch point —
//      builds a chain of detections that move *away* from the launch point,
//      and finally fits a parabola (RANSAC) for a clean, physically plausible
//      arc.
//
// All stored coordinates are normalized to [0,1] of the video frame.

export class BallTracker {
  /**
   * @param {object} opts
   * @param {number} opts.threshold     motion diff threshold (0-255).
   * @param {?{r,g,b}} opts.ballColor   ball colour sampled from the user tap.
   * @param {string} opts.colorHint     'auto' | 'white' | 'yellow' | 'orange'.
   * @param {number} opts.searchRadius  base search radius (fraction of width).
   * @param {number} opts.shakeRange    max camera-shake search (proc pixels).
   */
  constructor(opts = {}) {
    this.threshold = opts.threshold ?? 22;
    this.ballColor = opts.ballColor ?? null;
    this.colorHint = opts.colorHint ?? 'auto';
    this.searchRadius = opts.searchRadius ?? 0.16;
    this.shakeRange = opts.shakeRange ?? 6;
    this.frames = []; // [{ t, energy, blobs:[{x,y,area,aspect,r,g,b}] }]
    this.prevGray = null;
    this.w = 0;
    this.h = 0;
  }

  /** process one frame (downscaled ImageData) at presentation time t (sec). */
  observe(imageData, t) {
    const { data, width: w, height: h } = imageData;
    const n = w * h;
    this.w = w; this.h = h;

    const gray = new Uint8Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      gray[i] = (data[p] * 77 + data[p + 1] * 150 + data[p + 2] * 29) >> 8;
    }

    if (!this.prevGray) {
      this.prevGray = gray;
      this.frames.push({ t, energy: 0, blobs: [] });
      return;
    }

    // 1. cancel camera shake: find the background shift that best aligns frames
    const { dx, dy } = estimateShift(gray, this.prevGray, w, h, this.shakeRange);

    // 2. frame-difference against the shifted previous frame; keep ball pixels
    const mask = new Uint8Array(n);
    const T = this.threshold;
    let energy = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const px = x + dx, py = y + dy;
        const pv = (px >= 0 && px < w && py >= 0 && py < h)
          ? this.prevGray[py * w + px]
          : gray[i];
        if (Math.abs(gray[i] - pv) > T) {
          energy++;
          const p4 = i * 4;
          if (isBallPixel(data[p4], data[p4 + 1], data[p4 + 2], this.ballColor, this.colorHint)) {
            mask[i] = 1;
          }
        }
      }
    }
    this.prevGray = gray;

    const blobs = this._labelBlobs(mask, data, w, h);
    this.frames.push({ t, energy, blobs });
  }

  // connected-component labeling; returns ball-like blobs with mean colour.
  _labelBlobs(mask, data, w, h) {
    const visited = new Uint8Array(w * h);
    const stack = [];
    const blobs = [];
    const maxArea = Math.floor(w * h * 0.02); // ignore huge motion (body/club)
    const minArea = 1;

    for (let start = 0; start < mask.length; start++) {
      if (!mask[start] || visited[start]) continue;
      stack.length = 0;
      stack.push(start);
      visited[start] = 1;

      let area = 0, sumX = 0, sumY = 0, sumR = 0, sumG = 0, sumB = 0;
      let minX = w, maxX = 0, minY = h, maxY = 0;

      while (stack.length) {
        const idx = stack.pop();
        const x = idx % w;
        const y = (idx - x) / w;
        area++;
        sumX += x; sumY += y;
        const p4 = idx * 4;
        sumR += data[p4]; sumG += data[p4 + 1]; sumB += data[p4 + 2];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;

        if (x > 0)     { const j = idx - 1; if (mask[j] && !visited[j]) { visited[j] = 1; stack.push(j); } }
        if (x < w - 1) { const j = idx + 1; if (mask[j] && !visited[j]) { visited[j] = 1; stack.push(j); } }
        if (y > 0)     { const j = idx - w; if (mask[j] && !visited[j]) { visited[j] = 1; stack.push(j); } }
        if (y < h - 1) { const j = idx + w; if (mask[j] && !visited[j]) { visited[j] = 1; stack.push(j); } }
      }

      if (area < minArea || area > maxArea) continue;
      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      const aspect = Math.max(bw, bh) / Math.min(bw, bh);
      if (aspect > 5) continue; // reject streaky shapes (shafts, edges)

      blobs.push({
        x: (sumX / area) / w,
        y: (sumY / area) / h,
        area,
        aspect,
        r: sumR / area, g: sumG / area, b: sumB / area,
      });
    }
    return blobs;
  }

  /**
   * resolve the observed frames into a clean trajectory.
   * @param {{seed:?{x,y,t,color}}} opts  seed = tapped ball (launch) point.
   * @returns {{x,y,t}[]} fitted trajectory points (normalized).
   */
  resolve(opts = {}) {
    const seed = opts.seed || null;
    const frames = this.frames;
    if (frames.length < 3) return { points: [], impactT: null, raw: [] };

    // 1. impact = frame with peak motion energy (prefer at/after the tap time)
    let impactIdx = 0, maxE = -1;
    for (let i = 0; i < frames.length; i++) {
      if (seed && seed.t != null && frames[i].t < seed.t - 0.001) continue;
      if (frames[i].energy > maxE) { maxE = frames[i].energy; impactIdx = i; }
    }
    const impactT = frames[impactIdx].t;

    // 2. launch position: the tapped ball, or the first post-impact blob
    let launch = seed ? { x: seed.x, y: seed.y } : null;
    const post = frames.filter((f) => f.t >= impactT - 1e-6);
    if (!launch) {
      for (const f of post.slice(0, 5)) {
        if (f.blobs.length) { launch = { x: f.blobs[0].x, y: f.blobs[0].y }; break; }
      }
    }
    if (!launch) return { points: [], impactT, raw: [] };

    // 3. greedily chain detections that move away from the launch point
    const ballColor = this.ballColor;
    let lastPos = { ...launch };
    let vel = null;
    let lastDist = 0;
    let lost = 0;
    const baseR = this.searchRadius;
    const chain = [{ x: launch.x, y: launch.y, t: impactT }];

    for (const f of post) {
      if (f.t <= impactT) continue;
      const predicted = vel
        ? { x: lastPos.x + vel.x, y: lastPos.y + vel.y }
        : { ...lastPos };
      const R = baseR * (1 + lost * 0.7);

      let best = null, bestScore = Infinity;
      for (const b of f.blobs) {
        const dist = Math.hypot(b.x - predicted.x, b.y - predicted.y);
        if (dist > R) continue;
        const distFromLaunch = Math.hypot(b.x - launch.x, b.y - launch.y);
        if (vel && distFromLaunch < lastDist - 0.02) continue; // must travel outward
        let score = dist;
        if (ballColor) score += colorDist(b, ballColor) / 255 * 0.25;
        score += Math.max(0, b.area - 40) * 0.0005; // mild small-blob preference
        if (score < bestScore) { bestScore = score; best = b; }
      }

      if (best) {
        vel = { x: best.x - lastPos.x, y: best.y - lastPos.y };
        lastDist = Math.hypot(best.x - launch.x, best.y - launch.y);
        lastPos = { x: best.x, y: best.y };
        lost = 0;
        chain.push({ x: best.x, y: best.y, t: f.t });
      } else if (chain.length > 1) {
        if (++lost > 6) break;
      }
    }

    // 4. fit a parabola for a clean arc
    const raw = chain.slice();
    let points = chain;
    if (chain.length >= 3) {
      const fitted = fitCurve(chain);
      if (fitted) points = fitted;
    }
    return { points, impactT, raw };
  }
}

// ---- ball-colour classification --------------------------------------------

function isBallPixel(r, g, b, ballColor, hint) {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const luma = r * 0.299 + g * 0.587 + b * 0.114;
  const sat = mx <= 0 ? 0 : (mx - mn) / mx;

  // learned colour from the user's tap (allow some drift for motion blur)
  if (ballColor) {
    if (Math.hypot(r - ballColor.r, g - ballColor.g, b - ballColor.b) < 130) return true;
  }

  switch (hint) {
    case 'white':
      return luma > 165 && sat < 0.25;
    case 'yellow':
      return r > 150 && g > 130 && b < 150 && sat > 0.28;
    case 'orange':
      return r > 160 && g > 70 && g < 200 && b < 120 && sat > 0.40;
    default: // auto: bright white OR vivid colour (yellow/orange/etc.)
      if (luma > 175 && sat < 0.22) return true;
      if (sat > 0.45 && luma > 90 && luma < 245) return true;
      return false;
  }
}

function colorDist(blob, c) {
  return Math.hypot(blob.r - c.r, blob.g - c.g, blob.b - c.b);
}

// ---- camera-shake estimation -----------------------------------------------

// find the (dx,dy) shift of the previous frame that best matches the current
// frame's background (minimum mean absolute difference over a central region).
function estimateShift(cur, prev, w, h, R) {
  const x0 = (w * 0.2) | 0, x1 = (w * 0.8) | 0;
  const y0 = (h * 0.2) | 0, y1 = (h * 0.8) | 0;
  const step = 3;
  let bestDx = 0, bestDy = 0, bestSad = Infinity;
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      let sad = 0, cnt = 0;
      for (let y = y0; y < y1; y += step) {
        const py = y + dy;
        if (py < 0 || py >= h) continue;
        for (let x = x0; x < x1; x += step) {
          const px = x + dx;
          if (px < 0 || px >= w) continue;
          sad += Math.abs(cur[y * w + x] - prev[py * w + px]);
          cnt++;
        }
      }
      if (cnt) {
        const m = sad / cnt;
        if (m < bestSad) { bestSad = m; bestDx = dx; bestDy = dy; }
      }
    }
  }
  return { dx: bestDx, dy: bestDy };
}

// ---- parabola fitting (RANSAC) ---------------------------------------------

// fit x linear in time and y quadratic in time (gravity), robust to outliers.
function fitCurve(points) {
  const t0 = points[0].t;
  const us = points.map((p) => p.t - t0);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const n = points.length;
  const tol = 0.05;

  let bestInliers = null;
  const iters = Math.min(300, Math.max(40, n * n));
  for (let it = 0; it < iters; it++) {
    const idx = sample3(n);
    const su = idx.map((i) => us[i]);
    const cx = polyfit(su, idx.map((i) => xs[i]), 1);
    const cy = polyfit(su, idx.map((i) => ys[i]), 2);
    if (!cx || !cy) continue;
    const inliers = [];
    for (let i = 0; i < n; i++) {
      const ex = polyval(cx, us[i]) - xs[i];
      const ey = polyval(cy, us[i]) - ys[i];
      if (Math.hypot(ex, ey) < tol) inliers.push(i);
    }
    if (!bestInliers || inliers.length > bestInliers.length) bestInliers = inliers;
  }
  if (!bestInliers || bestInliers.length < 3) return null;

  const iu = bestInliers.map((i) => us[i]);
  const cx = polyfit(iu, bestInliers.map((i) => xs[i]), 1);
  const cy = polyfit(iu, bestInliers.map((i) => ys[i]), 2);
  if (!cx || !cy) return null;

  const uMax = Math.max(...us);
  const out = [];
  const N = 90;
  for (let k = 0; k <= N; k++) {
    const u = (uMax * k) / N;
    out.push({ x: polyval(cx, u), y: polyval(cy, u), t: t0 + u });
  }
  return out;
}

function sample3(n) {
  const a = Math.floor(Math.random() * n);
  let b = Math.floor(Math.random() * n);
  let c = Math.floor(Math.random() * n);
  while (b === a) b = Math.floor(Math.random() * n);
  while (c === a || c === b) c = Math.floor(Math.random() * n);
  return [a, b, c];
}

// least-squares polynomial fit via normal equations (deg 1 or 2).
function polyfit(us, vs, deg) {
  const m = deg + 1;
  const sumPow = new Array(2 * deg + 1).fill(0);
  const sumVU = new Array(m).fill(0);
  for (let k = 0; k < us.length; k++) {
    const u = us[k], v = vs[k];
    let up = 1;
    for (let p = 0; p <= 2 * deg; p++) { sumPow[p] += up; up *= u; }
    up = 1;
    for (let i = 0; i < m; i++) { sumVU[i] += v * up; up *= u; }
  }
  const A = [];
  for (let i = 0; i < m; i++) {
    A.push([]);
    for (let j = 0; j < m; j++) A[i].push(sumPow[i + j]);
  }
  return gaussSolve(A, sumVU.slice());
}

function polyval(coeffs, u) {
  let r = 0, up = 1;
  for (let i = 0; i < coeffs.length; i++) { r += coeffs[i] * up; up *= u; }
  return r;
}

// solve A x = b for small systems with partial pivoting; null if singular.
function gaussSolve(A, b) {
  const m = A.length;
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    }
    if (Math.abs(A[piv][col]) < 1e-12) return null;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = 0; r < m; r++) {
      if (r === col) continue;
      const f = A[r][col] / A[col][col];
      for (let c = col; c < m; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  return b.map((v, i) => v / A[i][i]);
}
