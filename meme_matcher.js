/* ═══════════════════════════════════════════════════════════════
   MemeMatch — Meme Matching Engine

   Consumes the v3 catalog written by analyze_memes.py: every meme
   carries a probability distribution over the 18 taxonomy categories
   (CLIP on pixels + OCR'd caption + curated folder + filename), plus a
   `face` prominence ratio.

   The live query is a face-api expression distribution (7 emotions),
   optionally nudged by a body gesture. Matching is a dot product
   between that query and the meme's category vector — but taken over
   *column-standardized* scores, because raw category columns have very
   different means and a plain dot product just returns whichever
   category the dataset happens to be saturated with.
   ═══════════════════════════════════════════════════════════════ */

/**
 * @typedef {Object} MemeEntry
 * @property {string} path      - Path relative to the app root
 * @property {string} name      - Filename without extension
 * @property {string} folder    - Curated Reactions/ folder, or ""
 * @property {string} text      - OCR'd caption ("" if none)
 * @property {number} face      - Largest face as a fraction of image area
 * @property {string[]} top     - Top-3 category names
 * @property {number[]} scores  - Probability per catalog category
 * @property {string} filename  - Basename with extension (added on load)
 * @property {string[]} emotions - Alias of `top` (added on load)
 */

class MemeMatcher {
    constructor() {
        /** @type {MemeEntry[]} */
        this.memes = [];
        /** @type {string[]} */
        this.categories = [];
        /** @type {Object<string, number>} */
        this.catIndex = {};
        /** Column-standardized scores, row-major [n * C]. @type {Float32Array} */
        this.z = new Float32Array(0);

        this.cooldownMap = new Map();   // path -> timestamp last shown
        this.cooldownMs = 15000;
        this.maxResults = 1;

        // How far down the ranking randomness is allowed to reach, so the
        // same expression does not always resolve to the same meme.
        this.varietyPool = 12;
        this.varietyTemp = 0.35;

        // Weight of the "is this actually a close-up face" bonus, in the
        // same z-score units as the category match itself.
        this.faceWeight = 0.35;
        this.faceSaturation = 0.18;     // face ratio at which the bonus maxes out

        // Gesture share of the query when a body pose is recognized, as a
        // multiple of the mass the face reading carries.
        //
        // Above parity on purpose: a gesture is something you *did*, while the
        // face is whatever your face happened to be doing, and the face signal
        // is near-saturated (a resting face reads neutral: 0.8+). At the old
        // 0.5 the body was decorative - holding a flex in front of a neutral
        // face put a gesture meme on top 0/8 draws. At 1.25 it is 6-8/8, and
        // the face still decides *which* one: angry+flexing and happy+flexing
        // resolve to different memes.
        this.poseWeight = 1.25;
    }

    /**
     * Load and index the catalog produced by analyze_memes.py.
     * @param {string} [url]
     * @returns {Promise<number>} number of memes indexed
     */
    async loadCatalog(url = './meme_catalog.json') {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status} loading ${url}`);
        const data = await response.json();

        if (Array.isArray(data) || !Array.isArray(data.memes)) {
            throw new Error('Catalog is not in v3 format — re-run analyze_memes.py');
        }

        this.categories = data.categories;
        this.catIndex = {};
        this.categories.forEach((c, i) => { this.catIndex[c] = i; });
        this.memes = data.memes;
        this.model = data.model || '';
        this.generated = data.generated || '';

        for (const m of this.memes) {
            m.filename = m.path.slice(m.path.lastIndexOf('/') + 1);
            m.emotions = m.top;
        }

        this._standardize();

        console.log(`[MemeMatcher] ${this.memes.length} memes, ` +
                    `${this.categories.length} categories, model ${this.model}`);
        return this.memes.length;
    }

    /**
     * Z-score each category column across the whole library.
     *
     * Categories are not equally represented and their prompt banks do not
     * sit equally close to the image manifold, so a raw score of 0.14 means
     * something very different for "love" than for "facepalm". Standardizing
     * per column makes the columns comparable, which is the difference
     * between a usable ranking and every query returning the same memes.
     */
    _standardize() {
        const n = this.memes.length;
        const C = this.categories.length;
        const z = new Float32Array(n * C);
        if (!n) { this.z = z; return; }

        for (let c = 0; c < C; c++) {
            let mean = 0;
            for (let i = 0; i < n; i++) mean += this.memes[i].scores[c];
            mean /= n;

            let varSum = 0;
            for (let i = 0; i < n; i++) {
                const d = this.memes[i].scores[c] - mean;
                varSum += d * d;
            }
            const sd = Math.sqrt(varSum / n) + 1e-8;

            for (let i = 0; i < n; i++) {
                z[i * C + c] = (this.memes[i].scores[c] - mean) / sd;
            }
        }
        this.z = z;
    }

    // ── Taxonomy ──────────────────────────────────────────────

    static EMOTION_EMOJIS = {
        happy: '😊', sad: '😢', angry: '😠', surprised: '😲', fearful: '😨',
        disgusted: '🤢', neutral: '😐', laughing: '😂', crying: '😭',
        smug: '😏', confused: '🤔', bored: '🥱', love: '😍', awkward: '😬',
        flexing: '💪', pointing: '👉', facepalm: '🤦', mocking: '😜',
    };

    /**
     * face-api reports 7 emotions; the catalog has 18 categories. Each
     * detected emotion is spread over the catalog categories it can
     * plausibly justify, which is what lets a plain "neutral" face still
     * pull up bored/smug reaction memes instead of only literal blank stares.
     */
    static EXPRESSION_SPREAD = {
        happy:     { happy: 0.55, laughing: 0.30, love: 0.15 },
        sad:       { sad: 0.55, crying: 0.30, bored: 0.15 },
        angry:     { angry: 0.60, mocking: 0.20, disgusted: 0.20 },
        surprised: { surprised: 0.60, confused: 0.25, fearful: 0.15 },
        fearful:   { fearful: 0.60, awkward: 0.20, surprised: 0.20 },
        disgusted: { disgusted: 0.60, mocking: 0.25, angry: 0.15 },
        neutral:   { neutral: 0.45, bored: 0.35, smug: 0.20 },
    };

    /** Body gesture -> catalog categories. */
    static POSE_SPREAD = {
        arms_raised:    { happy: 0.40, surprised: 0.30, flexing: 0.30 },
        t_pose:         { neutral: 0.50, smug: 0.50 },
        pointing_right: { pointing: 1.00 },
        pointing_left:  { pointing: 1.00 },
        hands_on_head:  { surprised: 0.40, fearful: 0.30, sad: 0.30 },
        facepalm:       { facepalm: 0.70, bored: 0.30 },
        thinking:       { confused: 0.60, smug: 0.40 },
        flexing:        { flexing: 0.70, smug: 0.30 },
        crossed_arms:   { angry: 0.50, smug: 0.50 },
        unknown:        {},
    };

    // ── Pose classification ───────────────────────────────────

    /**
     * Classify a body pose into a gesture name.
     *
     * Keypoints use MoveNet naming and pixel coordinates; app.js adapts
     * MediaPipe's 33 landmarks into that shape. All thresholds are
     * expressed as multiples of shoulder width, so the classifier behaves
     * the same at any camera resolution or distance from the lens.
     *
     * @param {Array<{x:number,y:number,score:number,name:string}>} keypoints
     * @returns {string} gesture name
     */
    classifyPose(keypoints) {
        if (!keypoints || keypoints.length < 11) return 'unknown';

        const kp = {};
        for (const p of keypoints) kp[p.name] = p;

        const MIN = 0.3;
        const ok = (name) => kp[name] && kp[name].score > MIN;
        if (!ok('left_shoulder') || !ok('right_shoulder')) return 'unknown';

        const lSh = kp.left_shoulder, rSh = kp.right_shoulder;
        const shoulderY = (lSh.y + rSh.y) / 2;
        const unit = Math.max(Math.hypot(lSh.x - rSh.x, lSh.y - rSh.y), 1e-3);

        const lWr = kp.left_wrist, rWr = kp.right_wrist;
        const lEl = kp.left_elbow, rEl = kp.right_elbow;
        const nose = kp.nose;

        const bothWristsUp = ok('left_wrist') && ok('right_wrist') &&
            lWr.y < shoulderY - 0.35 * unit && rWr.y < shoulderY - 0.35 * unit;

        // Hands pulled in tight above the head, rather than thrown wide.
        if (bothWristsUp && ok('nose') &&
            Math.abs(lWr.x - nose.x) < 0.8 * unit &&
            Math.abs(rWr.x - nose.x) < 0.8 * unit) {
            return 'hands_on_head';
        }

        // Double bicep flex. Checked before arms_raised because a flex also
        // puts the wrists above the shoulders — what separates them is that a
        // flex keeps the elbows out wide and at shoulder height, with the
        // forearms vertical, while raised arms carry the elbows up too.
        if (ok('left_wrist') && ok('left_elbow') && ok('right_wrist') && ok('right_elbow')) {
            const vertical = (wr, el) => wr.y < el.y - 0.15 * unit &&
                                         Math.abs(wr.x - el.x) < 0.45 * unit;
            const elbowsDown = lEl.y > shoulderY - 0.20 * unit && rEl.y > shoulderY - 0.20 * unit;
            const elbowsWide = Math.abs(lEl.x - rEl.x) > 1.3 * unit;
            if (vertical(lWr, lEl) && vertical(rWr, rEl) && elbowsDown && elbowsWide) {
                return 'flexing';
            }
        }

        if (bothWristsUp) return 'arms_raised';

        // A hand at the face: covering it is a facepalm, resting near the
        // chin is thinking. Distance to the nose separates the two.
        if (ok('nose')) {
            for (const wr of [rWr, lWr]) {
                if (!wr || wr.score <= MIN) continue;
                const d = Math.hypot(wr.x - nose.x, wr.y - nose.y) / unit;
                if (d < 0.30) return 'facepalm';
                if (d < 0.60) return 'thinking';
            }
        }

        // One arm extended sideways at roughly shoulder height. Keypoints are
        // named anatomically, so which screen direction "right_wrist" travels
        // depends on mirroring; measuring outward from the body midline
        // instead keeps this correct either way.
        const midX = (lSh.x + rSh.x) / 2;
        for (const [wr, sh] of [[lWr, lSh], [rWr, rSh]]) {
            if (!wr || wr.score <= MIN) continue;
            const outward = Math.sign(sh.x - midX) || 1;
            if ((wr.x - sh.x) * outward > 1.0 * unit &&
                Math.abs(wr.y - sh.y) < 0.55 * unit) {
                // The preview is displayed mirrored, so a low raw x reads as
                // the viewer's right.
                return wr.x < midX ? 'pointing_right' : 'pointing_left';
            }
        }

        // Arms held out level and wide.
        if (ok('left_elbow') && ok('right_elbow') &&
            Math.abs(lEl.y - shoulderY) < 0.3 * unit &&
            Math.abs(rEl.y - shoulderY) < 0.3 * unit &&
            Math.abs(lEl.x - rEl.x) > 1.8 * unit) {
            return 't_pose';
        }

        // Wrists crossed in front of the chest, below the shoulders.
        if (ok('left_wrist') && ok('right_wrist') &&
            lWr.y > shoulderY && rWr.y > shoulderY &&
            lWr.y < shoulderY + 0.9 * unit && rWr.y < shoulderY + 0.9 * unit &&
            lWr.x < rSh.x + 0.35 * unit && rWr.x > lSh.x - 0.35 * unit) {
            return 'crossed_arms';
        }

        return 'unknown';
    }

    // ── Query construction ────────────────────────────────────

    /**
     * Turn detected expressions + gesture into a normalized weight vector
     * over the catalog categories.
     * @returns {Float32Array}
     */
    buildTarget(expressions, gesture = 'unknown', manualEmotion = null) {
        const C = this.categories.length;
        const target = new Float32Array(C);
        const add = (cat, w) => {
            const i = this.catIndex[cat];
            if (i !== undefined) target[i] += w;
        };

        if (manualEmotion) {
            add(manualEmotion, 1.0);
        } else {
            let faceMass = 0;
            if (expressions) {
                for (const [expr, prob] of Object.entries(expressions)) {
                    if (prob <= 0.05) continue;
                    const spread = MemeMatcher.EXPRESSION_SPREAD[expr];
                    if (!spread) continue;
                    for (const [cat, share] of Object.entries(spread)) add(cat, prob * share);
                    faceMass += prob;
                }
            }

            const spread = MemeMatcher.POSE_SPREAD[gesture];
            if (spread && Object.keys(spread).length) {
                // The gesture takes a fixed share of the query, independent
                // of how confident the face detector happened to be.
                const poseMass = faceMass > 0 ? this.poseWeight * faceMass : 1.0;
                for (const [cat, share] of Object.entries(spread)) add(cat, poseMass * share);
            }
        }

        let total = 0;
        for (let i = 0; i < C; i++) total += target[i];
        if (total <= 0) {
            add('neutral', 1.0);
            total = 1.0;
        }
        for (let i = 0; i < C; i++) target[i] /= total;
        return target;
    }

    // ── Matching ──────────────────────────────────────────────

    /**
     * Rank the library against the current expression/gesture.
     * @returns {Array<{meme: MemeEntry, score: number, raw: number, category: string}>}
     */
    findMatches(expressions, gesture = 'unknown', manualEmotion = null) {
        const n = this.memes.length;
        if (!n) return [];

        const C = this.categories.length;
        const now = Date.now();
        const target = this.buildTarget(expressions, gesture, manualEmotion);

        // Only the categories carrying real weight are worth touching.
        const active = [];
        for (let c = 0; c < C; c++) if (target[c] > 1e-4) active.push(c);

        const raw = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const base = i * C;
            let s = 0;
            for (const c of active) s += target[c] * this.z[base + c];

            // A close-up reaction face answers "match my expression" better
            // than a wide multi-panel template does.
            const face = this.memes[i].face || 0;
            if (face > 0) s += this.faceWeight * Math.min(face / this.faceSaturation, 1);

            raw[i] = s;
        }

        // Rank on a copy that carries the cooldown penalty, so recently shown
        // memes drop out of contention without distorting the reported match.
        const ranked = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const last = this.cooldownMap.get(this.memes[i].path);
            ranked[i] = (last && now - last < this.cooldownMs) ? raw[i] - 10 : raw[i];
        }

        const order = Array.from({ length: n }, (_, i) => i)
            .sort((a, b) => ranked[b] - ranked[a]);

        const results = [];
        const pool = order.slice(0, Math.max(this.varietyPool, this.maxResults));
        while (results.length < this.maxResults && pool.length) {
            const pick = this._sampleByScore(pool, ranked);
            const i = pool.splice(pick, 1)[0];
            const meme = this.memes[i];
            this.cooldownMap.set(meme.path, now);
            results.push({
                meme,
                score: this._confidence(raw[i]),
                raw: raw[i],
                category: this._bestCategory(target, i),
            });
        }
        return results;
    }

    /** Softmax sample over the candidate pool — top-ranked usually wins, not always. */
    _sampleByScore(pool, ranked) {
        const best = ranked[pool[0]];
        const weights = pool.map(i => Math.exp((ranked[i] - best) / this.varietyTemp));
        const total = weights.reduce((a, b) => a + b, 0);
        let r = Math.random() * total;
        for (let k = 0; k < weights.length; k++) {
            r -= weights[k];
            if (r <= 0) return k;
        }
        return 0;
    }

    /**
     * Map a match score to a 0..1 confidence for display.
     *
     * `raw` is already in z-score units — a weighted blend of standardized
     * category columns — so a plain logistic on it spreads sensibly across
     * the range the scorer actually produces (roughly -2 to +4.5). A
     * percentile would not: the top of a 1300-meme library always sits above
     * the 99th percentile, so every match would read as "99%".
     */
    _confidence(value) {
        return 1 / (1 + Math.exp(-(value - 1.0) / 0.9));
    }

    /** Which requested category this meme actually answered on. */
    _bestCategory(target, i) {
        const C = this.categories.length;
        let bestCat = this.memes[i].top[0], best = -Infinity;
        for (let c = 0; c < C; c++) {
            if (target[c] <= 1e-4) continue;
            const contrib = target[c] * this.z[i * C + c];
            if (contrib > best) { best = contrib; bestCat = this.categories[c]; }
        }
        return bestCat;
    }

    /**
     * Dominant face-api expression, for the on-screen badge.
     * @returns {{name: string, score: number, emoji: string}}
     */
    getDominantExpression(expressions) {
        if (!expressions) return { name: 'neutral', score: 0, emoji: '😐' };

        let maxName = 'neutral', maxScore = 0;
        for (const [name, score] of Object.entries(expressions)) {
            if (score > maxScore) { maxScore = score; maxName = name; }
        }
        return {
            name: maxName,
            score: maxScore,
            emoji: MemeMatcher.EMOTION_EMOJIS[maxName] || '😐',
        };
    }

    resetCooldowns() {
        this.cooldownMap.clear();
    }
}
