import { BallTracker } from './tracker.js';
import { exportVideo, fileExtFor } from './exporter.js';
import { detectImpact, poseAvailable } from './pose.js';

// ---- element refs ----
const $ = (id) => document.getElementById(id);
const fileInput = $('file-input');
const filenameEl = $('filename');
const stageCard = $('step-stage');
const video = $('video');
const overlay = $('overlay');
const tapHint = $('tap-hint');
const seek = $('seek');
const btnTap = $('btn-tap');
const btnClearSeed = $('btn-clear-seed');
const ballColorSel = $('ball-color');
const sensitivity = $('sensitivity');
const sensitivityVal = $('sensitivity-val');
const btnAnalyze = $('btn-analyze');
const progress = $('progress');
const progressBar = $('progress-bar');
const statusEl = $('status');
const exportBlock = $('export-block');
const btnReplay = $('btn-replay');
const btnExport = $('btn-export');
const exportProgress = $('export-progress');
const exportProgressBar = $('export-progress-bar');
const resultBlock = $('result-block');
const btnDownload = $('btn-download');
const btnShare = $('btn-share');
const btnTrace = $('btn-trace');
const btnTraceUndo = $('btn-trace-undo');
const btnTraceClear = $('btn-trace-clear');
const btnTraceDone = $('btn-trace-done');
const traceStatus = $('trace-status');

const octx = overlay.getContext('2d');

// processing canvas for the CV analysis. The golf ball is tiny, so we must
// NOT downscale much or it vanishes (at 426px wide it was ~4px and undetectable;
// it only becomes reliably detectable around 1280px+). Cap the longer side at
// 1600px to keep the ball large while bounding compute.
const PROC_MAX_DIM = 1600;
const procCanvas = document.createElement('canvas');
const pctx = procCanvas.getContext('2d', { willReadFrequently: true });

// export canvas (full resolution)
const exportCanvas = document.createElement('canvas');

// ---- state ----
let tracker = null;
let trajectory = [];      // cleaned points [{x,y,t}]
let seed = null;          // { x, y, t, color, bg, aim } ball launch point + direction
let tapStage = 'idle';    // 'idle' | 'ball' | 'aim'
let analyzed = false;
let lastBlob = null;
let lastFilename = 'golf-trajectory.mp4';
let busy = false;

// manual trace mode
let traceMode = false;
let tracePoints = []; // [{x,y,t}] taps along the ball flight

// ---- helpers ----
function setStatus(msg, kind = '') {
  statusEl.textContent = msg;
  statusEl.className = 'status' + (kind ? ' ' + kind : '');
}

function sizeCanvases() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  overlay.width = w;
  overlay.height = h;
  exportCanvas.width = w;
  exportCanvas.height = h;
  const scale = Math.min(1, PROC_MAX_DIM / Math.max(w, h));
  procCanvas.width = Math.round(w * scale);
  procCanvas.height = Math.round(h * scale);
}

// average colour of a small patch (radius r) of the current frame.
function avgPatch(px, py, r, w, h) {
  const x0 = Math.max(0, px - r), y0 = Math.max(0, py - r);
  const sw = Math.min(w - x0, 2 * r + 1), sh = Math.min(h - y0, 2 * r + 1);
  if (sw <= 0 || sh <= 0) return null;
  const d = pctx.getImageData(x0, y0, sw, sh).data;
  let R = 0, G = 0, B = 0, c = 0;
  for (let i = 0; i < d.length; i += 4) { R += d[i]; G += d[i + 1]; B += d[i + 2]; c++; }
  return c ? { r: R / c, g: G / c, b: B / c } : null;
}

// learn the ball colour (centre patch) and the surrounding background/grass
// colour (a ring around the tap), from the current video frame.
function learnColors(nx, ny) {
  const w = procCanvas.width, h = procCanvas.height;
  pctx.drawImage(video, 0, 0, w, h);
  const cx = Math.round(nx * w), cy = Math.round(ny * h);
  const ball = avgPatch(cx, cy, 2, w, h);
  const rr = Math.max(4, Math.round(w * 0.035));
  const ring = [
    avgPatch(cx - rr, cy, 1, w, h), avgPatch(cx + rr, cy, 1, w, h),
    avgPatch(cx, cy - rr, 1, w, h), avgPatch(cx, cy + rr, 1, w, h),
  ].filter(Boolean);
  let R = 0, G = 0, B = 0;
  for (const s of ring) { R += s.r; G += s.g; B += s.b; }
  const bg = ring.length ? { r: R / ring.length, g: G / ring.length, b: B / ring.length } : null;
  return { ball, bg };
}

function setTapHint(text) {
  tapHint.textContent = text;
  tapHint.classList.remove('hidden');
}
function hideTapHint() {
  tapHint.classList.add('hidden');
}

// draw trajectory (+ seed marker). uptoTime limits drawing to the growing arc.
function drawOverlay(ctx, uptoTime = null) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (seed && !analyzed) {
    ctx.save();
    ctx.strokeStyle = '#ffd24a';
    ctx.lineWidth = Math.max(2, w * 0.004);
    ctx.beginPath();
    ctx.arc(seed.x * w, seed.y * h, w * 0.018, 0, Math.PI * 2);
    ctx.stroke();
    if (seed.aim) drawArrow(ctx, seed.x * w, seed.y * h, seed.aim.x * w, seed.aim.y * h, w);
    ctx.restore();
  }

  if (!trajectory.length) return;
  const pts = uptoTime == null
    ? trajectory
    : trajectory.filter((p) => p.t <= uptoTime + 1e-3);
  if (pts.length < 2) {
    if (pts.length === 1) drawBall(ctx, pts[0].x * w, pts[0].y * h, w);
    return;
  }

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(255,210,74,0.9)';
  ctx.shadowBlur = w * 0.012;
  ctx.strokeStyle = 'rgba(255,210,74,0.35)';
  ctx.lineWidth = Math.max(6, w * 0.012);
  strokePath(ctx, pts, w, h);
  ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ffd24a';
  ctx.lineWidth = Math.max(2.5, w * 0.005);
  strokePath(ctx, pts, w, h);
  const last = pts[pts.length - 1];
  drawBall(ctx, last.x * w, last.y * h, w);
  ctx.restore();
}

function strokePath(ctx, pts, w, h) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x * w, pts[0].y * h);
  for (let i = 1; i < pts.length; i++) {
    const p0 = pts[i - 1], p1 = pts[i];
    const mx = (p0.x + p1.x) / 2 * w;
    const my = (p0.y + p1.y) / 2 * h;
    ctx.quadraticCurveTo(p0.x * w, p0.y * h, mx, my);
  }
  const lp = pts[pts.length - 1];
  ctx.lineTo(lp.x * w, lp.y * h);
  ctx.stroke();
}

// dashed aim arrow from the ball toward the tapped launch direction.
function drawArrow(ctx, x0, y0, x1, y1, w) {
  ctx.save();
  ctx.strokeStyle = 'rgba(255,210,74,0.85)';
  ctx.fillStyle = 'rgba(255,210,74,0.85)';
  ctx.lineWidth = Math.max(2, w * 0.005);
  ctx.setLineDash([w * 0.02, w * 0.015]);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.setLineDash([]);
  const ang = Math.atan2(y1 - y0, x1 - x0);
  const head = w * 0.03;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(ang - 0.4), y1 - head * Math.sin(ang - 0.4));
  ctx.lineTo(x1 - head * Math.cos(ang + 0.4), y1 - head * Math.sin(ang + 0.4));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawBall(ctx, x, y, w) {
  ctx.save();
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#ffd24a';
  ctx.lineWidth = Math.max(1.5, w * 0.003);
  ctx.beginPath();
  ctx.arc(x, y, Math.max(3, w * 0.008), 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function redraw() {
  if (traceMode || (tracePoints.length && !analyzed)) { drawTracePreview(); return; }
  drawOverlay(octx, analyzed ? video.currentTime : null);
}

// ---- manual trace: tap the ball on a few frames, fit a smooth arc ----
function drawTracePreview() {
  const w = overlay.width, h = overlay.height;
  octx.clearRect(0, 0, w, h);
  // tapped points
  for (let i = 0; i < tracePoints.length; i++) {
    const p = tracePoints[i];
    octx.save();
    octx.fillStyle = '#3fd0ff';
    octx.strokeStyle = '#ffffff';
    octx.lineWidth = Math.max(1.5, w * 0.003);
    octx.beginPath();
    octx.arc(p.x * w, p.y * h, Math.max(4, w * 0.008), 0, Math.PI * 2);
    octx.fill(); octx.stroke();
    octx.restore();
  }
  // preview curve
  if (tracePoints.length >= 2) {
    const curve = fitTrace();
    if (curve.length >= 2) {
      octx.save();
      octx.strokeStyle = '#ffd24a';
      octx.lineWidth = Math.max(2.5, w * 0.005);
      octx.lineJoin = 'round'; octx.lineCap = 'round';
      octx.beginPath();
      octx.moveTo(curve[0].x * w, curve[0].y * h);
      for (let i = 1; i < curve.length; i++) octx.lineTo(curve[i].x * w, curve[i].y * h);
      octx.stroke();
      octx.restore();
    }
  }
}

// fit x linear / y quadratic (parabola) through the tapped points, sample dense.
function fitTrace() {
  const pts = tracePoints.slice().sort((a, b) => a.t - b.t);
  const t0 = pts[0].t;
  const us = pts.map((p) => p.t - t0);
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const cx = polyfit(us, xs, 1);
  const cy = polyfit(us, ys, pts.length >= 3 ? 2 : 1);
  if (!cx || !cy) return pts.map((p) => ({ x: p.x, y: p.y, t: p.t }));
  const uMax = us[us.length - 1] || 0.01;
  const out = [];
  const N = 80;
  for (let k = 0; k <= N; k++) {
    const u = (uMax * k) / N;
    out.push({ x: polyval(cx, u), y: polyval(cy, u), t: t0 + u });
  }
  return out;
}

function polyval(c, u) { let r = 0, up = 1; for (let i = 0; i < c.length; i++) { r += c[i] * up; up *= u; } return r; }
function polyfit(us, vs, deg) {
  const m = deg + 1;
  const sp = new Array(2 * deg + 1).fill(0);
  const sv = new Array(m).fill(0);
  for (let k = 0; k < us.length; k++) {
    let up = 1; const u = us[k], v = vs[k];
    for (let p = 0; p <= 2 * deg; p++) { sp[p] += up; up *= u; }
    up = 1; for (let i = 0; i < m; i++) { sv[i] += v * up; up *= u; }
  }
  const A = []; for (let i = 0; i < m; i++) { A.push([]); for (let j = 0; j < m; j++) A[i].push(sp[i + j]); }
  // gaussian elimination
  for (let c = 0; c < m; c++) {
    let piv = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (Math.abs(A[piv][c]) < 1e-12) return null;
    [A[c], A[piv]] = [A[piv], A[c]]; [sv[c], sv[piv]] = [sv[piv], sv[c]];
    for (let r = 0; r < m; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < m; k++) A[r][k] -= f * A[c][k];
      sv[r] -= f * sv[c];
    }
  }
  return sv.map((v, i) => v / A[i][i]);
}

function updateTraceUI() {
  btnTrace.textContent = traceMode ? '⏹ トレース終了' : '✏️ 手動トレース開始';
  btnTrace.classList.toggle('active', traceMode);
  const has = tracePoints.length > 0;
  btnTraceUndo.classList.toggle('hidden', !traceMode || !has);
  btnTraceClear.classList.toggle('hidden', !traceMode || !has);
  btnTraceDone.classList.toggle('hidden', !traceMode || tracePoints.length < 2);
  if (traceMode) {
    traceStatus.textContent = `タップ ${tracePoints.length} 点（スライダーでコマを進めながらボールをタップ。2点以上で確定可）`;
  }
}

btnTrace.addEventListener('click', () => {
  traceMode = !traceMode;
  if (traceMode) {
    analyzed = false;
    trajectory = [];
    exportBlock.classList.add('hidden');
    setStatus('');
  }
  updateTraceUI();
  redraw();
});

btnTraceUndo.addEventListener('click', () => { tracePoints.pop(); updateTraceUI(); redraw(); });
btnTraceClear.addEventListener('click', () => { tracePoints = []; updateTraceUI(); redraw(); });

btnTraceDone.addEventListener('click', () => {
  if (tracePoints.length < 2) return;
  trajectory = fitTrace();
  analyzed = true;
  traceMode = false;
  updateTraceUI();
  traceStatus.textContent = `軌道を確定しました（${tracePoints.length} 点）。`;
  exportBlock.classList.remove('hidden');
  video.currentTime = 0;
  redraw();
});

// ---- file loading ----
fileInput.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  filenameEl.textContent = file.name;
  lastFilename = file.name.replace(/\.[^.]+$/, '') + '-trajectory';
  resetForNewVideo();
  video.src = URL.createObjectURL(file);
  video.load();
});

function resetForNewVideo() {
  trajectory = [];
  seed = null;
  analyzed = false;
  lastBlob = null;
  tapStage = 'idle';
  btnTap.classList.remove('active');
  tapHint.classList.add('hidden');
  btnClearSeed.classList.add('hidden');
  exportBlock.classList.add('hidden');
  resultBlock.classList.add('hidden');
  setStatus('');
}

video.addEventListener('loadedmetadata', () => {
  sizeCanvases();
  stageCard.classList.remove('hidden');
  video.currentTime = 0;
});

video.addEventListener('loadeddata', () => {
  octx.clearRect(0, 0, overlay.width, overlay.height);
});

// ---- scrubbing ----
seek.addEventListener('input', () => {
  if (!video.duration) return;
  video.currentTime = (seek.value / 100) * video.duration;
});
video.addEventListener('timeupdate', () => {
  if (video.duration) seek.value = (video.currentTime / video.duration) * 100;
});
video.addEventListener('seeked', redraw);

// ---- two-tap setup: ball position (+colour), then launch direction ----
btnTap.addEventListener('click', () => {
  if (analyzed) return;
  if (tapStage === 'idle') {
    tapStage = 'ball';
    btnTap.classList.add('active');
    setTapHint('① 止まっているボールをタップ');
  } else {
    tapStage = 'idle';
    btnTap.classList.remove('active');
    hideTapHint();
  }
});

overlay.addEventListener('pointerdown', (e) => {
  const rect = overlay.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;

  // manual trace mode takes precedence: record a tap on the ball
  if (traceMode) {
    tracePoints.push({ x, y, t: video.currentTime });
    updateTraceUI();
    redraw();
    return;
  }

  if (tapStage === 'idle' || analyzed) return;

  if (tapStage === 'ball') {
    const c = learnColors(x, y);
    seed = { x, y, t: video.currentTime, color: c.ball, bg: c.bg, aim: null };
    tapStage = 'aim';
    setTapHint('② ボールが飛んだ方向をタップ（おおよそでOK）');
    btnClearSeed.classList.remove('hidden');
  } else if (tapStage === 'aim') {
    if (seed) seed.aim = { x, y };
    tapStage = 'idle';
    btnTap.classList.remove('active');
    hideTapHint();
  }
  redraw();
});

btnClearSeed.addEventListener('click', () => {
  seed = null;
  tapStage = 'idle';
  btnTap.classList.remove('active');
  hideTapHint();
  btnClearSeed.classList.add('hidden');
  redraw();
});

// ---- sensitivity ----
sensitivity.addEventListener('input', () => {
  sensitivityVal.textContent = sensitivity.value;
});

// ---- frame iteration (low playback rate so high-fps frames are not skipped) ----
// kept very low because we now process at high resolution (heavier per frame);
// a slower play-through means the compositor presents — and we capture — more
// frames, so we don't miss the few frames where the fast ball is visible.
const ANALYZE_RATE = 0.13;

function scanFrames(onFrame) {
  return new Promise((resolve, reject) => {
    const useRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
    video.muted = true;
    video.playbackRate = ANALYZE_RATE;

    const finish = () => {
      video.playbackRate = 1.0;
      video.onended = null;
      resolve();
    };
    video.onended = finish;

    const begin = () => {
      if (useRVFC) {
        const step = (_now, meta) => {
          onFrame(meta.mediaTime);
          if (!video.ended && !video.paused) video.requestVideoFrameCallback(step);
          else finish();
        };
        video.requestVideoFrameCallback(step);
      } else {
        const tick = () => {
          if (video.ended || video.paused) return finish();
          onFrame(video.currentTime);
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }
    };

    const startFromZero = () => video.play().then(begin).catch(reject);
    if (video.currentTime === 0) startFromZero();
    else {
      const onSeeked = () => { video.removeEventListener('seeked', onSeeked); startFromZero(); };
      video.addEventListener('seeked', onSeeked);
      video.currentTime = 0;
    }
  });
}

// ---- analyze ----
btnAnalyze.addEventListener('click', async () => {
  if (busy || !video.duration) return;
  busy = true;
  analyzed = false;
  trajectory = [];
  btnAnalyze.disabled = true;
  exportBlock.classList.add('hidden');
  resultBlock.classList.add('hidden');
  progress.classList.remove('hidden');
  progressBar.style.width = '0%';

  const pw = procCanvas.width;
  const ph = procCanvas.height;
  const duration = video.duration;

  try {
    // 1. AI auto-setup: if the user didn't tap, use pose detection to find the
    //    golfer, the impact moment and the launch (hand) position automatically.
    let activeSeed = seed;
    let usedAI = false;
    if (!activeSeed && poseAvailable()) {
      setStatus('AIでゴルファーとインパクトを自動検出中…');
      try {
        const det = await detectImpact(video, procCanvas, {
          onProgress: (p) => { progressBar.style.width = (p * 50).toFixed(0) + '%'; },
        });
        if (det) {
          // AI found the golfer + (ideally) the ball at the club. Use the ball's
          // position and learned colour as the launch anchor.
          activeSeed = {
            x: det.launch.x, y: det.launch.y, t: det.impactT,
            color: det.ballColor || null, bg: null, aim: null,
          };
          usedAI = true;
        }
      } catch (_e) { /* fall back below */ }
    }

    // 2. dense motion scan (shake compensation + ball blobs)
    setStatus(usedAI ? 'AIで検出した打点をもとに軌道を解析中…' : '解析中… 手ブレ補正とボール検出をしています');
    tracker = new BallTracker({
      threshold: Number(sensitivity.value),
      ballColor: activeSeed ? activeSeed.color : null,
      bgColor: activeSeed ? activeSeed.bg : null,
      colorHint: ballColorSel ? ballColorSel.value : 'auto',
    });

    await scanFrames((t) => {
      pctx.drawImage(video, 0, 0, pw, ph);
      const frame = pctx.getImageData(0, 0, pw, ph);
      tracker.observe(frame, t);
      if (duration) progressBar.style.width = (50 + Math.min(50, (t / duration) * 50)).toFixed(0) + '%';
    });

    const result = tracker.resolve({ seed: activeSeed });
    progressBar.style.width = '100%';

    const haveBallAnchor = !!(activeSeed && activeSeed.color);
    // only accept a trajectory backed by real, colour-matched ball detections
    // (raw includes the launch point, so >=3 means at least 2 tracked positions)
    const enoughDetections = result.raw.length >= 3;

    if (haveBallAnchor && enoughDetections && result.points.length >= 2) {
      trajectory = result.points;
      analyzed = true;
      const how = usedAI ? 'AI自動検出' : '手動指定';
      setStatus(
        `軌道を描きました（${how} / インパクト ${result.impactT.toFixed(2)}秒 / 検出点 ${result.raw.length}）。`,
        'ok',
      );
      exportBlock.classList.remove('hidden');
      video.currentTime = 0;
      redraw();
    } else if (usedAI && !haveBallAnchor) {
      setStatus(
        'AIはゴルファーを検出しましたが、ボールを特定できませんでした。' +
        '①手動で「ボールと方向をタップ」する、または②ボールが空を横切る構図（カメラを低く上向き）で撮り直すと改善します。',
        'error',
      );
    } else {
      setStatus(
        '打った後のボールを十分に追えませんでした（背景に溶け込んで見えていない可能性）。' +
        'ボールが空を横切る構図での撮影、または手動タップをお試しください。',
        'error',
      );
    }
  } catch (err) {
    setStatus('解析中にエラーが発生しました: ' + err.message, 'error');
  } finally {
    busy = false;
    btnAnalyze.disabled = false;
    setTimeout(() => progress.classList.add('hidden'), 600);
  }
});

// ---- replay (preview overlay growing in real time) ----
btnReplay.addEventListener('click', () => {
  if (busy || !analyzed) return;
  video.playbackRate = 1.0;
  video.muted = true;
  const useRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

  const run = () => {
    video.play();
    if (useRVFC) {
      const step = (_n, meta) => {
        drawOverlay(octx, meta.mediaTime);
        if (!video.ended && !video.paused) video.requestVideoFrameCallback(step);
      };
      video.requestVideoFrameCallback(step);
    } else {
      const tick = () => {
        if (video.ended || video.paused) return;
        drawOverlay(octx, video.currentTime);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  };

  if (video.currentTime > 0.05) {
    const onSeeked = () => { video.removeEventListener('seeked', onSeeked); run(); };
    video.addEventListener('seeked', onSeeked);
    video.currentTime = 0;
  } else run();
});

// ---- export ----
btnExport.addEventListener('click', async () => {
  if (busy || !analyzed) return;
  busy = true;
  btnExport.disabled = true;
  resultBlock.classList.add('hidden');
  exportProgress.classList.remove('hidden');
  exportProgressBar.style.width = '0%';
  setStatus('動画を書き出し中…（再生時間ぶんかかります）');

  try {
    const { blob, mimeType } = await exportVideo({
      video,
      canvas: exportCanvas,
      drawOverlay: (ctx, t) => drawOverlay(ctx, t),
      onProgress: (p) => { exportProgressBar.style.width = (p * 100).toFixed(0) + '%'; },
    });
    lastBlob = blob;
    const ext = fileExtFor(mimeType);
    lastFilename = lastFilename.replace(/\.[^.]+$/, '') + '.' + ext;

    const url = URL.createObjectURL(blob);
    btnDownload.href = url;
    btnDownload.download = lastFilename;
    resultBlock.classList.remove('hidden');
    setStatus('書き出し完了！ ダウンロードまたは共有できます。', 'ok');
  } catch (err) {
    setStatus('書き出しエラー: ' + err.message, 'error');
  } finally {
    busy = false;
    btnExport.disabled = false;
    video.playbackRate = 1.0;
    setTimeout(() => exportProgress.classList.add('hidden'), 600);
    redraw();
  }
});

// ---- share ----
btnShare.addEventListener('click', async () => {
  if (!lastBlob) return;
  const file = new File([lastBlob], lastFilename, { type: lastBlob.type });
  try {
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: 'ゴルフ弾道', text: 'ゴルフの弾道を撮影しました ⛳' });
    } else {
      setStatus('この端末は共有に未対応です。ダウンロードをご利用ください。', 'error');
    }
  } catch (err) {
    if (err.name !== 'AbortError') setStatus('共有エラー: ' + err.message, 'error');
  }
});
