// In-browser AI assist using TensorFlow.js MoveNet pose detection.
//
// Strategy (per the user's idea): recognise the PERSON, then the BALL that sits
// at the end of the club the person is holding. At address the ball is large
// and clearly visible, so locating it there gives a reliable launch point,
// colour and size — and "near the hands" disambiguates it from spare balls,
// white shoes or shirt logos. The motion tracker then takes over from impact.
//
// TF.js + pose-detection are loaded from a CDN in index.html (window.tf /
// window.poseDetection). If they (or the model) fail to load, callers fall back.

let detectorPromise = null;

export function poseAvailable() {
  return typeof window !== 'undefined' && !!window.poseDetection && !!window.tf;
}

async function getDetector() {
  if (!poseAvailable()) return null;
  if (!detectorPromise) {
    const pd = window.poseDetection;
    detectorPromise = pd.createDetector(pd.SupportedModels.MoveNet, {
      modelType: pd.movenet.modelType.SINGLEPOSE_LIGHTNING,
    });
  }
  return detectorPromise;
}

function seekTo(video, t) {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; video.removeEventListener('seeked', finish); resolve(); } };
    video.addEventListener('seeked', finish);
    video.currentTime = Math.min(Math.max(0, t), video.duration || t);
    setTimeout(finish, 400);
  });
}

function keypointMap(keypoints) {
  const m = {};
  for (const k of keypoints) m[k.name] = k;
  return m;
}

// is this pixel plausibly a golf ball? white (bright, low saturation) OR a
// vivid yellow/orange (high saturation, not green — excludes grass).
function ballness(r, g, b) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const luma = 0.299 * r + 0.587 * g + 0.114 * b;
  const sat = mx <= 0 ? 0 : (mx - mn) / mx;
  if (luma > 150 && sat < 0.30) return true;                 // white
  const green = g > r + 10 && g > b + 10;                     // grass / foliage
  if (sat > 0.40 && luma > 80 && luma < 250 && !green) return true; // yellow/orange
  return false;
}

// Find the ball nearest the hands within a search window. Returns
// { x, y, color } normalized, or null.
function findBallNear(ctx, W, H, handPx, radiusFrac) {
  const R = Math.round(radiusFrac * W);
  const x0 = Math.max(0, handPx.x - R), y0 = Math.max(0, handPx.y - R);
  const x1 = Math.min(W, handPx.x + R), y1 = Math.min(H, handPx.y + R);
  const w = x1 - x0, h = y1 - y0;
  if (w <= 0 || h <= 0) return null;

  const data = ctx.getImageData(x0, y0, w, h).data;
  const mask = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    if (ballness(data[p], data[p + 1], data[p + 2])) mask[i] = 1;
  }

  // connected components (iterative flood fill)
  const visited = new Uint8Array(w * h);
  const stack = [];
  let best = null, bestScore = Infinity;
  const minArea = Math.max(6, Math.round(w * h * 0.0008));
  const maxArea = Math.round(w * h * 0.06);

  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || visited[s]) continue;
    stack.length = 0; stack.push(s); visited[s] = 1;
    let area = 0, sx = 0, sy = 0, sr = 0, sg = 0, sb = 0;
    let mnx = w, mxx = 0, mny = h, mxy = 0;
    while (stack.length) {
      const idx = stack.pop();
      const x = idx % w, y = (idx - x) / w;
      area++; sx += x; sy += y;
      const p = idx * 4; sr += data[p]; sg += data[p + 1]; sb += data[p + 2];
      if (x < mnx) mnx = x; if (x > mxx) mxx = x;
      if (y < mny) mny = y; if (y > mxy) mxy = y;
      if (x > 0 && mask[idx - 1] && !visited[idx - 1]) { visited[idx - 1] = 1; stack.push(idx - 1); }
      if (x < w - 1 && mask[idx + 1] && !visited[idx + 1]) { visited[idx + 1] = 1; stack.push(idx + 1); }
      if (y > 0 && mask[idx - w] && !visited[idx - w]) { visited[idx - w] = 1; stack.push(idx - w); }
      if (y < h - 1 && mask[idx + w] && !visited[idx + w]) { visited[idx + w] = 1; stack.push(idx + w); }
    }
    if (area < minArea || area > maxArea) continue;
    const bw = mxx - mnx + 1, bh = mxy - mny + 1;
    const aspect = Math.max(bw, bh) / Math.min(bw, bh);
    const fill = area / (bw * bh);
    if (aspect > 1.7 || fill < 0.55) continue;   // must be roundish & solid
    const cx = sx / area, cy = sy / area;
    const dist = Math.hypot((x0 + cx) - handPx.x, (y0 + cy) - handPx.y);
    const score = dist;                          // nearest qualifying blob wins
    if (score < bestScore) {
      bestScore = score;
      best = {
        x: (x0 + cx) / W, y: (y0 + cy) / H,
        color: { r: sr / area, g: sg / area, b: sb / area },
      };
    }
  }
  return best;
}

/**
 * Pose pass + ball-at-address detection.
 * @returns {Promise<?{impactT:number, launch:{x,y}, ballColor:?{r,g,b}}>}
 */
export async function detectImpact(video, canvas, opts = {}) {
  const detector = await getDetector();
  if (!detector) return null;

  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const dur = video.duration;
  if (!dur || !isFinite(dur)) return null;

  const wasRate = video.playbackRate;
  video.pause();

  const N = 120;
  const step = Math.max(1 / 60, dur / N);
  const samples = []; // {t, hand:{x,y}px or null}

  try {
    for (let t = 0; t <= dur + 1e-3; t += step) {
      await seekTo(video, t);
      ctx.drawImage(video, 0, 0, W, H);
      const poses = await detector.estimatePoses(canvas, { maxPoses: 1, flipHorizontal: false });
      let hand = null;
      if (poses && poses[0]) {
        const kp = keypointMap(poses[0].keypoints);
        const cands = [kp.right_wrist, kp.left_wrist].filter((k) => k && k.score > 0.3);
        if (cands.length) {
          hand = {
            x: cands.reduce((a, c) => a + c.x, 0) / cands.length,
            y: cands.reduce((a, c) => a + c.y, 0) / cands.length,
          };
        }
      }
      samples.push({ t, hand });
      if (opts.onProgress) opts.onProgress(Math.min(1, t / dur));
    }
  } catch (err) {
    video.playbackRate = wasRate;
    return null;
  }

  // impact = sample where the hands move fastest
  let impactIdx = -1, best = -1;
  for (let i = 1; i < samples.length; i++) {
    if (!samples[i].hand || !samples[i - 1].hand) continue;
    const dt = samples[i].t - samples[i - 1].t || 1e-3;
    const sp = Math.hypot(
      samples[i].hand.x - samples[i - 1].hand.x,
      samples[i].hand.y - samples[i - 1].hand.y,
    ) / dt;
    if (sp > best) { best = sp; impactIdx = i; }
  }
  if (impactIdx < 0) { video.playbackRate = wasRate; return null; }

  const impactT = samples[impactIdx].t;
  const handAtImpact = samples[impactIdx].hand;

  // locate the ball at address: search near the hands in a sharp pre-impact
  // frame (a few samples before impact, before motion blur sets in).
  let ball = null;
  const addrIdx = Math.max(0, impactIdx - 5);
  const addrHand = samples[addrIdx].hand || handAtImpact;
  if (addrHand) {
    try {
      await seekTo(video, samples[addrIdx].t);
      ctx.drawImage(video, 0, 0, W, H);
      ball = findBallNear(ctx, W, H, addrHand, 0.28);
    } catch (_e) { ball = null; }
  }

  video.playbackRate = wasRate;

  const launch = ball
    ? { x: ball.x, y: ball.y }
    : { x: handAtImpact.x / W, y: handAtImpact.y / H };
  return { impactT, launch, ballColor: ball ? ball.color : null };
}
