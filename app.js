/* ═══════════════════════════════════════════════════════════════
   MemeMatch — Main Application Controller
   Handles webcam, face-api.js, pose detection, and UI updates
   ═══════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── State ─────────────────────────────────────────────────
    const state = {
        cameraActive: false,
        poseEnabled: false,
        modelsLoaded: false,
        poseModelLoaded: false,
        detecting: false,
        currentEmotion: 'neutral',
        frameCount: 0,
        fps: 0,
        lastFpsTime: Date.now(),
    };

    // ── Matcher Instance ──────────────────────────────────────
    const matcher = new MemeMatcher();

    // ── DOM References ────────────────────────────────────────
    const $ = (id) => document.getElementById(id);
    const webcam = $('webcam');
    const overlay = $('overlay');
    const startBtn = $('startCameraBtn');
    const poseBtn = $('togglePoseBtn');
    const captureBtn = $('captureBtn');
    const cameraPlaceholder = $('cameraPlaceholder');
    const expressionBadge = $('expressionBadge');
    const expressionEmoji = $('expressionEmoji');
    const expressionText = $('expressionText');
    const expressionConfidence = $('expressionConfidence');
    const poseBadge = $('poseBadge');
    const poseText = $('poseText');
    const memeGrid = $('memeGrid');
    const loadingOverlay = $('loadingOverlay');
    const loaderText = $('loaderText');
    const totalMemesEl = $('totalMemes');
    const fpsCounter = $('fpsCounter');
    const emotionFilter = $('emotionFilter');
    const lightbox = $('lightbox');
    const lightboxImg = $('lightboxImg');
    const lightboxName = $('lightboxName');
    const lightboxTags = $('lightboxTags');
    const lightboxClose = $('lightboxClose');
    const lightboxBackdrop = $('lightboxBackdrop');

    let poseDetector = null;
    let detectionInterval = null;
    let memeUpdateDebounce = null;
    let lastDetectedExpressions = null;
    let lastGesture = 'unknown';

    // ── Initialization ────────────────────────────────────────
    async function init() {
        createParticles();
        await loadMemeCatalog();
        setupEventListeners();
    }

    // ── Particles ─────────────────────────────────────────────
    function createParticles() {
        const container = $('particles');
        for (let i = 0; i < 30; i++) {
            const p = document.createElement('div');
            p.className = 'particle';
            p.style.left = Math.random() * 100 + '%';
            p.style.animationDuration = (8 + Math.random() * 12) + 's';
            p.style.animationDelay = Math.random() * 8 + 's';
            p.style.width = (1 + Math.random() * 2) + 'px';
            p.style.height = p.style.width;
            container.appendChild(p);
        }
    }

    // ── Load Meme Catalog ─────────────────────────────────────
    async function loadMemeCatalog() {
        try {
            const count = await matcher.loadCatalog();
            totalMemesEl.textContent = count;
        } catch (err) {
            showToast('Failed to load meme catalog. Run analyze_memes.py first!', 'error');
        }
    }

    // ── Event Listeners ───────────────────────────────────────
    function setupEventListeners() {
        startBtn.addEventListener('click', toggleCamera);
        if (poseBtn) poseBtn.addEventListener('click', togglePose);
        if (captureBtn) captureBtn.addEventListener('click', takeSnapshot);
        emotionFilter.addEventListener('change', onFilterChange);
        lightboxClose.addEventListener('click', closeLightbox);
        lightboxBackdrop.addEventListener('click', closeLightbox);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeLightbox();
        });
    }

    // ── Camera Toggle ─────────────────────────────────────────
    async function toggleCamera() {
        if (state.cameraActive) {
            stopCamera();
        } else {
            await startCamera();
        }
    }

    async function startCamera() {
        startBtn.disabled = true;
        startBtn.innerHTML = '<iconify-icon icon="lucide:loader-2" class="animate-spin"></iconify-icon> Starting...';

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                },
            });

            webcam.style.display = 'block';
            cameraPlaceholder.style.display = 'none';

            webcam.srcObject = stream;
            await webcam.play();

            // Set canvas size to match video
            overlay.width = webcam.videoWidth;
            overlay.height = webcam.videoHeight;

            state.cameraActive = true;

            startBtn.innerHTML = '<div class="w-2.5 h-2.5 bg-white rounded-sm"></div> Stop Camera';
            startBtn.disabled = false;
            if (poseBtn) poseBtn.disabled = false;
            if (captureBtn) captureBtn.disabled = false;

            // Load face-api models
            await loadFaceModels();

            // Start detection loop
            startDetection();

            // Body gestures are a third of the catalog's vocabulary (flexing,
            // pointing, facepalm), so pose comes up with the camera instead of
            // waiting for someone to find the toggle. Loads in the background:
            // no blocking overlay, and the button still switches it off.
            enablePose(true);

        } catch (err) {
            console.error('Camera error:', err);
            showToast('Camera access denied. Please allow camera permissions.', 'error');
            startBtn.innerHTML = '<iconify-icon icon="lucide:play"></iconify-icon> Start Camera';
            startBtn.disabled = false;
        }
    }

    function stopCamera() {
        state.cameraActive = false;
        state.detecting = false;

        if (detectionInterval) {
            cancelAnimationFrame(detectionInterval);
            detectionInterval = null;
        }

        const stream = webcam.srcObject;
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
        }
        webcam.srcObject = null;
        webcam.style.display = 'none';
        cameraPlaceholder.style.display = 'flex';

        expressionBadge.classList.remove('visible');
        poseBadge.classList.remove('visible');

        // Clear canvas
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        startBtn.innerHTML = '<iconify-icon icon="lucide:play"></iconify-icon> Start Camera';
        if (poseBtn) poseBtn.disabled = true;
        if (captureBtn) captureBtn.disabled = true;
        state.poseEnabled = false;
        if (poseBtn) poseBtn.classList.remove('active');
    }

    // ── Snapshot ──────────────────────────────────────────────
    function takeSnapshot() {
        if (!state.cameraActive) {
            showToast('Start the camera first!', 'error');
            return;
        }

        // ── Flash feedback — the user knows the capture happened instantly ──
        const flashEl = document.createElement('div');
        flashEl.style.cssText = 'position:fixed;inset:0;background:#fff;z-index:99999;pointer-events:none;opacity:0.85;transition:opacity 0.25s';
        document.body.appendChild(flashEl);
        requestAnimationFrame(() => { flashEl.style.opacity = '0'; });
        setTimeout(() => flashEl.remove(), 300);

        // ── Build the composite canvas ──
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const cw = webcam.videoWidth;
        const ch = webcam.videoHeight;

        const memeImg = memeGrid.querySelector('img');
        const hasMeme = memeImg && memeImg.naturalWidth > 0;

        const padding = 20;

        if (hasMeme) {
            const memeRatio = memeImg.naturalWidth / memeImg.naturalHeight;
            const memeW = ch * memeRatio;
            canvas.width = cw + memeW + (padding * 3);
            canvas.height = ch + (padding * 2);

            ctx.fillStyle = '#f3f4f6';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Webcam (flipped)
            ctx.save();
            ctx.translate(padding + cw, padding);
            ctx.scale(-1, 1);
            ctx.drawImage(webcam, 0, 0, cw, ch);
            ctx.restore();

            // Meme image
            ctx.drawImage(memeImg, padding * 2 + cw, padding, memeW, ch);
        } else {
            // No meme yet — just capture the webcam
            canvas.width = cw + (padding * 2);
            canvas.height = ch + (padding * 2);

            ctx.fillStyle = '#f3f4f6';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.save();
            ctx.translate(padding + cw, padding);
            ctx.scale(-1, 1);
            ctx.drawImage(webcam, 0, 0, cw, ch);
            ctx.restore();
        }

        // ── Fire-and-forget upload to Supabase ──
        const dataUrl = canvas.toDataURL('image/png');
        fetch('/api/upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imageBase64: dataUrl }),
        })
        .then(r => {
            if (!r.ok) {
                return r.text().then(t => { throw new Error(`HTTP ${r.status}: ${t}`); });
            }
            return r.json();
        })
        .then(data => {
            if (data.success) {
                console.log('Snapshot saved:', data.url);
                showToast('Captured! 📸', 'info');
            } else {
                console.error('Upload failed:', data);
                showToast('Failed to save: ' + (data.error || 'unknown'), 'error');
            }
        })
        .catch(err => {
            console.error('Upload error:', err);
            showToast('Error saving: ' + err.message, 'error');
        });
    }

    // ── Face-API Model Loading ────────────────────────────────
    async function loadFaceModels() {
        if (state.modelsLoaded) return;

        showLoading('Loading face detection models...');

        try {
            // Weights ship in ./models, so the app runs with no network at all.
            const MODEL_URL = './models';

            await Promise.all([
                faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
                faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
                faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
            ]);

            state.modelsLoaded = true;
            hideLoading();
        } catch (err) {
            console.error('Model loading error:', err);
            hideLoading();
            showToast('Failed to load face models from ./models.', 'error');
        }
    }

    // ── Pose Model Loading ────────────────────────────────────
    async function loadPoseModel(silent = false) {
        if (state.poseModelLoaded) return;

        if (!silent) showLoading('Loading pose detection model...');

        try {
            // MediaPipe Tasks ships as an ES module; pull it in on demand so
            // the rest of the app stays a plain classic script.
            const vision = await import('./vendor/mediapipe/vision_bundle.mjs');
            const fileset = await vision.FilesetResolver.forVisionTasks('./vendor/mediapipe/wasm');

            poseDetector = await vision.PoseLandmarker.createFromOptions(fileset, {
                baseOptions: { modelAssetPath: './models/pose_landmarker_lite.task' },
                runningMode: 'VIDEO',
                numPoses: 1,
            });

            state.poseModelLoaded = true;
            if (!silent) hideLoading();
        } catch (err) {
            console.error('Pose model error:', err);
            if (!silent) hideLoading();
            showToast('Failed to load pose model.', 'error');
        }
    }

    // ── Pose Toggle ───────────────────────────────────────────
    async function enablePose(silent = false) {
        state.poseEnabled = true;
        if (poseBtn) poseBtn.classList.add('active');
        poseBadge.style.display = '';
        poseBadge.classList.add('visible');

        await loadPoseModel(silent);
        if (!state.poseModelLoaded) disablePose();   // load failed - roll the UI back
    }

    function disablePose() {
        state.poseEnabled = false;
        if (poseBtn) poseBtn.classList.remove('active');
        poseBadge.classList.remove('visible');
        lastGesture = 'unknown';
    }

    async function togglePose() {
        if (state.poseEnabled) disablePose();
        else await enablePose();
    }

    // ── Detection Loop ────────────────────────────────────────
    function startDetection() {
        state.detecting = true;

        async function detectFrame() {
            if (!state.detecting || !state.cameraActive) return;

            try {
                // Face expression detection
                if (state.modelsLoaded) {
                    const detections = await faceapi
                        .detectAllFaces(webcam, new faceapi.TinyFaceDetectorOptions({
                            inputSize: 320,
                            scoreThreshold: 0.4,
                        }))
                        .withFaceLandmarks()
                        .withFaceExpressions();

                    drawDetections(detections);

                    if (detections.length > 0) {
                        const expressions = detections[0].expressions;
                        lastDetectedExpressions = expressions;
                        updateExpressionUI(expressions);
                    }
                }

                // Pose detection
                if (state.poseEnabled && poseDetector) {
                    try {
                        const result = poseDetector.detectForVideo(webcam, performance.now());
                        const landmarks = result.landmarks && result.landmarks[0];
                        if (landmarks) {
                            const keypoints = toMoveNetKeypoints(landmarks);
                            const gesture = matcher.classifyPose(keypoints);
                            lastGesture = gesture;
                            drawPose(keypoints);
                            updatePoseUI(gesture);
                        }
                    } catch (poseErr) {
                        // Silently handle pose errors
                    }
                }

                // Update meme matches (debounced)
                updateMemeMatches();

                // FPS counter
                state.frameCount++;
                const now = Date.now();
                if (now - state.lastFpsTime >= 1000) {
                    state.fps = state.frameCount;
                    state.frameCount = 0;
                    state.lastFpsTime = now;
                    fpsCounter.textContent = state.fps + ' FPS';
                }

            } catch (err) {
                console.error('Detection error:', err);
            }

            detectionInterval = requestAnimationFrame(detectFrame);
        }

        detectFrame();
    }

    // ── Draw Face Detections ──────────────────────────────────
    function drawDetections(detections) {
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        // Visual debug overlays disabled per user request
    }

    // ── MediaPipe → MoveNet Keypoints ─────────────────────────
    // MediaPipe returns 33 normalized landmarks by index. The matcher and the
    // skeleton renderer both speak MoveNet's 17 named keypoints in pixels, so
    // translate once here rather than teaching them a second format.
    const MEDIAPIPE_TO_MOVENET = [
        [0, 'nose'], [2, 'left_eye'], [5, 'right_eye'], [7, 'left_ear'], [8, 'right_ear'],
        [11, 'left_shoulder'], [12, 'right_shoulder'], [13, 'left_elbow'], [14, 'right_elbow'],
        [15, 'left_wrist'], [16, 'right_wrist'], [23, 'left_hip'], [24, 'right_hip'],
        [25, 'left_knee'], [26, 'right_knee'], [27, 'left_ankle'], [28, 'right_ankle'],
    ];

    function toMoveNetKeypoints(landmarks) {
        const w = overlay.width, h = overlay.height;
        return MEDIAPIPE_TO_MOVENET.map(([idx, name]) => {
            const lm = landmarks[idx];
            if (!lm) return { x: 0, y: 0, score: 0, name };
            return {
                x: lm.x * w,
                y: lm.y * h,
                // `visibility` is how MediaPipe expresses keypoint confidence.
                score: lm.visibility === undefined ? 1 : lm.visibility,
                name,
            };
        });
    }

    // ── Draw Pose Keypoints ───────────────────────────────────
    function drawPose(keypoints) {
        const ctx = overlay.getContext('2d');

        // Visual debug overlays disabled per user request
    }

    // ── Update Expression UI ──────────────────────────────────
    function updateExpressionUI(expressions) {
        const dominant = matcher.getDominantExpression(expressions);

        // Update badge
        expressionBadge.classList.add('visible');
        expressionEmoji.textContent = dominant.emoji;
        expressionText.textContent = dominant.name;
        expressionConfidence.textContent = Math.round(dominant.score * 100) + '%';

        // Update bars
        for (const emotion of ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']) {
            const score = expressions[emotion] || 0;
            const bar = $('bar-' + emotion);
            const val = $('val-' + emotion);
            if (bar) bar.style.width = (score * 100) + '%';
            if (val) val.textContent = Math.round(score * 100) + '%';
        }

        state.currentEmotion = dominant.name;
    }

    // ── Update Pose UI ────────────────────────────────────────
    function updatePoseUI(gesture) {
        if (gesture !== 'unknown') {
            poseBadge.classList.add('visible');
            const label = gesture.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            poseText.textContent = label;
        } else {
            poseText.textContent = 'No pose';
        }
    }

    // ── Update Meme Matches (Debounced + Stability Check) ────
    let lastMatchedEmotion = null;
    let lastMatchedConfidence = 0;
    let lastMatchedGesture = 'unknown';

    function updateMemeMatches() {
        if (memeUpdateDebounce) return;

        memeUpdateDebounce = setTimeout(() => {
            memeUpdateDebounce = null;

            // Stability check: re-rank when the *query* moves, and the query is
            // expression + gesture. Leaving the gesture out of this test meant a
            // held face pinned the results, so flexing or pointing in front of a
            // steady expression changed nothing — the body half of the catalog
            // (flexing, pointing, facepalm) was unreachable in practice.
            const gestureChanged = lastGesture !== lastMatchedGesture;
            if (lastDetectedExpressions) {
                const dominant = matcher.getDominantExpression(lastDetectedExpressions);
                const emotionChanged = dominant.name !== lastMatchedEmotion;
                const confidenceShift = Math.abs(dominant.score - lastMatchedConfidence) > 0.20;

                if (!emotionChanged && !confidenceShift && !gestureChanged) {
                    return; // Skip update — expression and gesture are both stable
                }

                lastMatchedEmotion = dominant.name;
                lastMatchedConfidence = dominant.score;
            }
            lastMatchedGesture = lastGesture;

            const filterValue = emotionFilter.value;
            const manualEmotion = filterValue !== 'auto' ? filterValue : null;

            const results = matcher.findMatches(
                lastDetectedExpressions,
                lastGesture,
                manualEmotion
            );

            renderMemeGrid(results);
        }, 500); // Fast updates — capture is user-triggered via Capture button
    }

    // ── Filter Change Handler ─────────────────────────────────
    function onFilterChange() {
        matcher.resetCooldowns();

        const filterValue = emotionFilter.value;
        if (filterValue !== 'auto') {
            const manualEmotion = filterValue;
            const results = matcher.findMatches(null, 'unknown', manualEmotion);
            renderMemeGrid(results);
        }
    }

    // ── Render Meme Grid ──────────────────────────────────────
    function renderMemeGrid(results) {
        if (!results || results.length === 0) {
            memeGrid.innerHTML = `
                <div class="text-center p-8">
                    <span class="text-5xl block mb-4">🤷</span>
                    <p class="font-bold text-[#333333]">No matching memes found</p>
                    <p class="text-sm text-gray-500 font-medium mt-2">Try a different expression!</p>
                </div>
            `;
            return;
        }

        const r = results[0];
        const name = r.meme.name;
        // `score` is the fraction of the library this meme outranks, so it
        // reads as "better than 98% of 1300 memes" rather than a bare cosine.
        const score = Math.round(r.score * 100);
        const tags = [r.category, ...r.meme.emotions.filter(e => e !== r.category)].slice(0, 3);
        const emotionTags = tags.map(e =>
            `<span class="px-3 py-1.5 bg-white border-2 border-gray-200 rounded-full text-xs md:text-sm font-bold text-gray-600 flex items-center gap-1.5 shadow-sm whitespace-nowrap">
                <span class="text-base md:text-lg">${MemeMatcher.EMOTION_EMOJIS[e] || ''}</span> ${e}
            </span>`
        ).join('');

        memeGrid.innerHTML = `
            <div class="single-meme-display w-full flex-1 flex flex-col items-center justify-center cursor-pointer group" data-path="${encodeURIComponent(r.meme.path)}">
                <div class="relative w-full rounded-2xl bg-gray-100 overflow-hidden border-[6px] border-[#333333] shadow-[6px_6px_0_#333333] rotate-1 group-hover:rotate-0 transition-transform duration-300 flex items-center justify-center">
                    <div class="absolute top-3 right-3 md:top-4 md:right-4 bg-[#FF6B6B] text-white px-2 py-1 md:px-3 md:py-1.5 rounded-lg text-xs md:text-sm font-black z-10 shadow-lg border-2 border-white rotate-6">
                        ${score}% MATCH
                    </div>
                    <img src="${encodeURI(r.meme.path)}"
                         class="w-full h-auto max-h-[50vh] md:max-h-[400px] object-contain"
                         alt="${escapeHtml(name)}"
                         onerror="this.src='data:image/svg+xml,<svg xmlns=http://www.w3.org/2000/svg/>'">
                </div>
            </div>
        `;

        // Click to open lightbox
        memeGrid.querySelector('.single-meme-display').addEventListener('click', () => {
            openLightbox(r.meme);
        });
    }

    // ── Lightbox ──────────────────────────────────────────────
    function openLightbox(meme) {
        lightboxImg.src = encodeURI(meme.path);
        lightboxName.textContent = meme.name;

        const tags = meme.emotions.map(e =>
            `<span class="lightbox-tag">${MemeMatcher.EMOTION_EMOJIS[e] || ''} ${e}</span>`
        );
        if (meme.folder) tags.push(`<span class="lightbox-tag">📁 ${escapeHtml(meme.folder)}</span>`);
        // The OCR'd caption is what the meme actually says — worth showing.
        if (meme.text) tags.push(`<span class="lightbox-tag">💬 ${escapeHtml(meme.text.slice(0, 90))}</span>`);
        lightboxTags.innerHTML = tags.join('');

        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
    }

    // ── Loading Overlay ───────────────────────────────────────
    function showLoading(text) {
        loaderText.textContent = text;
        loadingOverlay.classList.add('active');
    }

    function hideLoading() {
        loadingOverlay.classList.remove('active');
    }

    // ── Toast Notifications ───────────────────────────────────
    function showToast(message, type = 'info') {
        const container = $('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 4200);
    }

    // ── Utility ───────────────────────────────────────────────
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Boot ──────────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
