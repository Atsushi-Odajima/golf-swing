// In-browser AI assist using TensorFlow.js MoveNet pose detection.
//
// We don't try to detect the tiny, fast ball with a model (that needs a custom
// trained network + dataset). Instead we detect the GOLFER reliably, find the
// moment the hands move fastest (= impact) and where the hands are at that
// moment (= the ball's launch area). That removes the manual taps and gives the
// motion tracker a trustworthy launch point and impact time to work from.
//
// The TF.js + pose-detection libraries are loaded from a CDN in index.html and
// exposed as window.tf / window.poseDetection. If they (or the model weights)
// fail to load, callers fall back to the non-AI path.

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
    // some browsers don't fire 'seeked' if the time barely changes
    setTimeout(finish, 400);
  });
}

function keypointMap(keypoints) {
  const m = {};
  for (const k of keypoints) m[k.name] = k;
  return m;
}

/**
 * Run a sparse pose pass over the video to find the impact moment and the
 * launch (hand) position.
 *
 * @param {HTMLVideoElement} video
 * @param {HTMLCanvasElement} canvas  scratch canvas to draw frames into
 * @param {{onProgress?:(p:number)=>void}} [opts]
 * @returns {Promise<?{impactT:number, launch:{x:number,y:number}}>} normalized
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

  const N = 120;                          // number of pose samples
  const step = Math.max(1 / 60, dur / N);
  const hands = [];                       // [{t, x, y}] hand point per sample (px)

  try {
    for (let t = 0; t <= dur + 1e-3; t += step) {
      await seekTo(video, t);
      ctx.drawImage(video, 0, 0, W, H);
      const poses = await detector.estimatePoses(canvas, { maxPoses: 1, flipHorizontal: false });
      if (poses && poses[0]) {
        const kp = keypointMap(poses[0].keypoints);
        const cands = [kp.right_wrist, kp.left_wrist].filter((k) => k && k.score > 0.3);
        if (cands.length) {
          const x = cands.reduce((a, c) => a + c.x, 0) / cands.length;
          const y = cands.reduce((a, c) => a + c.y, 0) / cands.length;
          hands.push({ t, x, y });
        }
      }
      if (opts.onProgress) opts.onProgress(Math.min(1, t / dur));
    }
  } catch (err) {
    video.playbackRate = wasRate;
    return null;
  }
  video.playbackRate = wasRate;

  if (hands.length < 3) return null;

  // impact = sample where the hands move fastest
  let impactT = null, launch = null, best = -1;
  for (let i = 1; i < hands.length; i++) {
    const dt = hands[i].t - hands[i - 1].t || 1e-3;
    const sp = Math.hypot(hands[i].x - hands[i - 1].x, hands[i].y - hands[i - 1].y) / dt;
    if (sp > best) { best = sp; impactT = hands[i].t; launch = hands[i]; }
  }
  if (impactT == null) return null;

  return { impactT, launch: { x: launch.x / W, y: launch.y / H } };
}
