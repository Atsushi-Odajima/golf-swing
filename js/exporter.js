// records the video + trajectory overlay into a downloadable/shareable file.

// pick the best supported recording format (prefer mp4 on iOS/Safari).
export function pickMimeType() {
  const candidates = [
    'video/mp4;codecs=h264',
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return '';
}

export function fileExtFor(mimeType) {
  return mimeType.includes('mp4') ? 'mp4' : 'webm';
}

/**
 * replay the video once at normal speed, drawing each frame + the growing
 * trajectory into a canvas, and record the canvas to a Blob.
 *
 * @param {object} cfg
 * @param {HTMLVideoElement} cfg.video
 * @param {HTMLCanvasElement} cfg.canvas      offscreen canvas at video resolution
 * @param {(ctx, t)=>void} cfg.drawOverlay    draws trajectory up to time t
 * @param {(p:number)=>void} [cfg.onProgress] 0..1
 * @returns {Promise<{blob: Blob, mimeType: string}>}
 */
export function exportVideo(cfg) {
  const { video, canvas, drawOverlay, onProgress } = cfg;
  const ctx = canvas.getContext('2d');
  const mimeType = pickMimeType();
  if (!mimeType) {
    return Promise.reject(new Error('この端末は動画の書き出しに対応していません。'));
  }

  return new Promise((resolve, reject) => {
    const fps = 30;
    const stream = canvas.captureStream(fps);
    const recorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: 8_000_000,
    });
    const chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onerror = (e) => reject(e.error || new Error('録画エラー'));
    recorder.onstop = () => resolve({ blob: new Blob(chunks, { type: mimeType }), mimeType });

    const duration = video.duration || 0;
    const useRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

    const renderFrame = (mediaTime) => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      drawOverlay(ctx, mediaTime);
      if (onProgress && duration) onProgress(Math.min(mediaTime / duration, 1));
    };

    const finish = () => {
      if (recorder.state !== 'inactive') recorder.stop();
    };

    video.muted = true;
    video.playbackRate = 1.0;
    video.currentTime = 0;
    video.onended = finish;

    const start = () => {
      recorder.start();
      if (useRVFC) {
        const step = (_now, meta) => {
          renderFrame(meta.mediaTime);
          if (!video.ended && !video.paused) video.requestVideoFrameCallback(step);
          else finish();
        };
        video.requestVideoFrameCallback(step);
      } else {
        // fallback: render on a timer while playing
        const tick = () => {
          if (video.ended || video.paused) return finish();
          renderFrame(video.currentTime);
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }
    };

    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      video.play().then(start).catch(reject);
    };
    if (video.currentTime === 0) {
      video.play().then(start).catch(reject);
    } else {
      video.addEventListener('seeked', onSeeked);
    }
  });
}
