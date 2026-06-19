// ball detection + trajectory tracking via frame differencing.
// all stored coordinates are normalized to [0,1] relative to the video frame,
// so they can be drawn on any canvas size.

export class BallTracker {
  /**
   * @param {object} opts
   * @param {number} opts.threshold   motion diff threshold (0-255). higher = less sensitive.
   * @param {number} opts.brightness  minimum brightness for a ball pixel (0-255).
   * @param {number} opts.searchRadius search radius for the next ball, as fraction of width.
   */
  constructor(opts = {}) {
    this.threshold = opts.threshold ?? 28;
    this.brightness = opts.brightness ?? 110;
    this.searchRadius = opts.searchRadius ?? 0.3;
    this.reset();
  }

  reset(seed = null) {
    this.prevGray = null;
    this.points = [];          // [{x, y, t}] normalized + time(sec)
    this.lastPos = seed;       // {x, y} normalized or null
    this.velocity = null;      // {x, y} normalized per step
    this.lostFrames = 0;
    this.started = !!seed;     // becomes true once tracking locks on
  }

  /**
   * process one frame.
   * @param {ImageData} imageData  frame pixels at processing resolution
   * @param {number} t             presentation time in seconds
   * @returns {?{x:number,y:number,t:number}} detected point (normalized) or null
   */
  process(imageData, t) {
    const { data, width, height } = imageData;
    const n = width * height;

    const gray = new Uint8Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      gray[i] = (data[p] * 77 + data[p + 1] * 150 + data[p + 2] * 29) >> 8; // luma
    }

    if (!this.prevGray) {
      this.prevGray = gray;
      return null;
    }

    // build a mask of pixels that both MOVED and are BRIGHT (golf ball ~ white)
    const mask = new Uint8Array(n);
    const T = this.threshold;
    const B = this.brightness;
    for (let i = 0; i < n; i++) {
      if (gray[i] > B && Math.abs(gray[i] - this.prevGray[i]) > T) mask[i] = 1;
    }
    this.prevGray = gray;

    const blobs = this._labelBlobs(mask, width, height);
    if (!blobs.length) {
      this.lostFrames++;
      return null;
    }

    const cand = this._selectBlob(blobs, width, height);
    if (!cand) {
      this.lostFrames++;
      return null;
    }

    const norm = { x: cand.cx / width, y: cand.cy / height };
    if (this.lastPos && this.started) {
      this.velocity = { x: norm.x - this.lastPos.x, y: norm.y - this.lastPos.y };
    }
    this.lastPos = norm;
    this.started = true;
    this.lostFrames = 0;

    const pt = { x: norm.x, y: norm.y, t };
    this.points.push(pt);
    return pt;
  }

  // connected-component labeling (4-neighbour flood fill, stack based).
  // returns [{cx, cy, area, w, h}] in processing-pixel coords.
  _labelBlobs(mask, w, h) {
    const visited = new Uint8Array(w * h);
    const blobs = [];
    const stack = [];
    const maxArea = Math.floor(w * h * 0.03); // ignore huge motion (body / club blur)

    for (let start = 0; start < mask.length; start++) {
      if (!mask[start] || visited[start]) continue;
      stack.length = 0;
      stack.push(start);
      visited[start] = 1;

      let area = 0, sumX = 0, sumY = 0;
      let minX = w, maxX = 0, minY = h, maxY = 0;

      while (stack.length) {
        const idx = stack.pop();
        const x = idx % w;
        const y = (idx - x) / w;
        area++;
        sumX += x; sumY += y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;

        if (x > 0)     { const j = idx - 1; if (mask[j] && !visited[j]) { visited[j] = 1; stack.push(j); } }
        if (x < w - 1) { const j = idx + 1; if (mask[j] && !visited[j]) { visited[j] = 1; stack.push(j); } }
        if (y > 0)     { const j = idx - w; if (mask[j] && !visited[j]) { visited[j] = 1; stack.push(j); } }
        if (y < h - 1) { const j = idx + w; if (mask[j] && !visited[j]) { visited[j] = 1; stack.push(j); } }
      }

      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      // a ball is small and roughly compact; drop tiny noise and oversized blobs
      if (area < 1 || area > maxArea) continue;
      const fill = area / (bw * bh);
      blobs.push({ cx: sumX / area, cy: sumY / area, area, w: bw, h: bh, fill });
    }
    return blobs;
  }

  // pick the most ball-like blob, biased toward the predicted next position.
  _selectBlob(blobs, w, h) {
    // predict where the ball should be next
    let predicted = null;
    if (this.lastPos) {
      predicted = {
        x: this.lastPos.x + (this.velocity ? this.velocity.x : 0),
        y: this.lastPos.y + (this.velocity ? this.velocity.y : 0),
      };
    }
    const radius = this.searchRadius * w;

    let best = null;
    let bestScore = Infinity;
    for (const b of blobs) {
      // compactness: prefer roundish, well-filled blobs
      const aspect = Math.max(b.w, b.h) / Math.min(b.w, b.h);
      if (aspect > 6) continue; // long streaks (club shaft etc.)

      let score;
      if (predicted) {
        const dx = b.cx - predicted.x * w;
        const dy = b.cy - predicted.y * h;
        const dist = Math.hypot(dx, dy);
        if (this.started && dist > radius) continue; // too far from prediction
        score = dist + b.area * 0.05 + aspect * 2;
      } else {
        // first lock-on: prefer small compact bright blobs
        score = b.area + aspect * 4;
      }
      if (score < bestScore) {
        bestScore = score;
        best = b;
      }
    }
    return best;
  }

  /**
   * clean up the raw points into a single forward-moving trajectory.
   * removes detections that contradict the dominant direction of travel.
   * @returns {{x:number,y:number,t:number}[]}
   */
  buildTrajectory() {
    const pts = this.points.slice().sort((a, b) => a.t - b.t);
    if (pts.length < 3) return pts;

    // dominant horizontal direction (median dx sign)
    const dxs = [];
    for (let i = 1; i < pts.length; i++) dxs.push(pts[i].x - pts[i - 1].x);
    dxs.sort((a, b) => a - b);
    const medianDx = dxs[Math.floor(dxs.length / 2)];
    const dir = medianDx >= 0 ? 1 : -1;

    // keep a monotonic-in-x chain following the dominant direction
    const cleaned = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
      const prev = cleaned[cleaned.length - 1];
      const stepX = (pts[i].x - prev.x) * dir;
      // allow small backward jitter, reject big reversals
      if (stepX > -0.02) cleaned.push(pts[i]);
    }
    return cleaned.length >= 2 ? cleaned : pts;
  }
}
