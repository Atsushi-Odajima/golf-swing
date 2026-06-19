import { BallTracker } from './tracker.js';
import { exportVideo, fileExtFor } from './exporter.js';

// ---- element refs ----
const $ = (id) => document.getElementById(id);
const fileInput = $('file-input');
const filenameEl = $('filename');
const stageCard = $('step-stage');
const stage = $('stage');
const video = $('video');
const overlay = $('overlay');
const tapHint = $('tap-hint');
const seek = $('seek');
const btnTap = $('btn-tap');
const btnClearSeed = $('btn-clear-seed');
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

const octx = overlay.getContext('2d');

// processing canvas (downscaled) for the CV analysis
const PROC_MAX_W = 426;
const procCanvas = document.createElement('canvas');
const pctx = procCanvas.getContext('2d', { willReadFrequently: true });

// export canvas (full resolution)
const exportCanvas = document.createElement('canvas');

// ---- state ----
let tracker = new BallTracker();
let trajectory = [];      // cleaned points [{x,y,t}]
let seed = null;          // normalized {x,y} ball start
let tapMode = false;
let analyzed = false;
let lastBlob = null;
let lastFilename = 'golf-trajectory.mp4';
let busy = false;

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
  const scale = Math.min(1, PROC_MAX_W / w);
  procCanvas.width = Math.round(w * scale);
  procCanvas.height = Math.round(h * scale);
}

// draw trajectory (+ seed marker) onto a context sized to the video frame.
// if uptoTime is a number, only the part of the flight up to that time is drawn.
function drawOverlay(ctx, uptoTime = null) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  // seed marker (before/after analysis)
  if (seed && !analyzed) {
    ctx.save();
    ctx.strokeStyle = '#ffd24a';
    ctx.lineWidth = Math.max(2, w * 0.004);
    ctx.beginPath();
    ctx.arc(seed.x * w, seed.y * h, w * 0.018, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  if (!trajectory.length) return;

  const pts = uptoTime == null
    ? trajectory
    : trajectory.filter((p) => p.t <= uptoTime + 1e-3);
  if (pts.length < 2) {
    // still show the launch point dot
    if (pts.length === 1) drawBall(ctx, pts[0].x * w, pts[0].y * h, w);
    return;
  }

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(255,210,74,0.9)';
  ctx.shadowBlur = w * 0.012;

  // outer glow
  ctx.strokeStyle = 'rgba(255,210,74,0.35)';
  ctx.lineWidth = Math.max(6, w * 0.012);
  strokePath(ctx, pts, w, h);

  // core line
  ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ffd24a';
  ctx.lineWidth = Math.max(2.5, w * 0.005);
  strokePath(ctx, pts, w, h);

  // current ball position
  const last = pts[pts.length - 1];
  drawBall(ctx, last.x * w, last.y * h, w);
  ctx.restore();
}

function strokePath(ctx, pts, w, h) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x * w, pts[0].y * h);
  for (let i = 1; i < pts.length; i++) {
    // quadratic smoothing through midpoints
    const p0 = pts[i - 1], p1 = pts[i];
    const mx = (p0.x + p1.x) / 2 * w;
    const my = (p0.y + p1.y) / 2 * h;
    ctx.quadraticCurveTo(p0.x * w, p0.y * h, mx, my);
  }
  const lp = pts[pts.length - 1];
  ctx.lineTo(lp.x * w, lp.y * h);
  ctx.stroke();
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
  drawOverlay(octx, analyzed ? video.currentTime : null);
}

// ---- file loading ----
fileInput.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  filenameEl.textContent = file.name;
  const baseName = file.name.replace(/\.[^.]+$/, '');
  lastFilename = baseName + '-trajectory';

  resetForNewVideo();
  video.src = URL.createObjectURL(file);
  video.load();
});

function resetForNewVideo() {
  trajectory = [];
  seed = null;
  analyzed = false;
  lastBlob = null;
  tapMode = false;
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
  // show first frame
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

// ---- tap to seed ball position ----
btnTap.addEventListener('click', () => {
  if (analyzed) return;
  tapMode = !tapMode;
  btnTap.classList.toggle('active', tapMode);
  tapHint.classList.toggle('hidden', !tapMode);
});

overlay.addEventListener('pointerdown', (e) => {
  if (!tapMode) return;
  const rect = overlay.getBoundingClientRect();
  seed = {
    x: (e.clientX - rect.left) / rect.width,
    y: (e.clientY - rect.top) / rect.height,
  };
  tapMode = false;
  btnTap.classList.remove('active');
  tapHint.classList.add('hidden');
  btnClearSeed.classList.remove('hidden');
  redraw();
});

btnClearSeed.addEventListener('click', () => {
  seed = null;
  btnClearSeed.classList.add('hidden');
  redraw();
});

// ---- sensitivity ----
sensitivity.addEventListener('input', () => {
  sensitivityVal.textContent = sensitivity.value;
});

// ---- frame iteration (low playback rate so high-fps frames are not skipped) ----
const ANALYZE_RATE = 0.2;

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
  setStatus('解析中… ボールを探しています');

  tracker = new BallTracker({ threshold: Number(sensitivity.value) });
  tracker.reset(seed);

  const pw = procCanvas.width;
  const ph = procCanvas.height;
  const duration = video.duration;

  try {
    await scanFrames((t) => {
      pctx.drawImage(video, 0, 0, pw, ph);
      const frame = pctx.getImageData(0, 0, pw, ph);
      tracker.process(frame, t);
      if (duration) progressBar.style.width = Math.min(100, (t / duration) * 100) + '%';
    });

    trajectory = tracker.buildTrajectory();
    progressBar.style.width = '100%';

    if (trajectory.length >= 2) {
      analyzed = true;
      setStatus(`軌道を検出しました（${trajectory.length} 点）。再生・書き出しできます。`, 'ok');
      exportBlock.classList.remove('hidden');
      video.currentTime = 0;
      redraw();
    } else {
      setStatus(
        'ボールの軌道を十分に検出できませんでした。感度を上げ下げする、ボール位置をタップで指定する、' +
        'またはスロー撮影の動画でお試しください。',
        'error'
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
