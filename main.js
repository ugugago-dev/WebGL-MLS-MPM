import { isMobile, dpr, PARTICLE_RADIUS, PARTICLE_COUNT, urlNum, urlFlag } from './config.js';
import { FluidGL } from './fluid-gl.js';
import { NRFRenderer } from './nrf-renderer.js';
import { getGL2, GLCapabilityError, loadEquirectTexture } from './gl-utils.js';
import { GLProfiler } from './gl-profiler.js';
import {
    mat4Perspective, mat4LookAt, mat4Multiply,
    cameraVectors, screenToWorld, worldToScreen, simBoundsOnScreen,
} from './math.js';

// ─────────────────────────────────────────────────────────────
//  粒子の初期化
// ─────────────────────────────────────────────────────────────
const particleCount = Math.round(urlNum('p', PARTICLE_COUNT));

const fluid = new FluidGL(2, 2, 3, urlNum('r', PARTICLE_RADIUS), particleCount);
const spacing = 0.03;
fluid.fillBlock(spacing, spacing, spacing, 1.0 - spacing, 1.0 - spacing, 0.6 - spacing);

// ─────────────────────────────────────────────────────────────
//  Canvas / overlay
// ─────────────────────────────────────────────────────────────
const c       = document.querySelector('#gl');
const overlay = document.querySelector('#overlay');
const octx    = overlay.getContext('2d');
overlay.style.touchAction = 'none';

let logicalW = window.innerWidth, logicalH = window.innerHeight;

function resizeCanvases() {
    logicalW = window.innerWidth;
    logicalH = window.innerHeight;
    c.width  = overlay.width  = Math.round(logicalW * dpr);
    c.height = overlay.height = Math.round(logicalH * dpr);
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (renderer) renderer.resize(c.width, c.height);
}

// ─────────────────────────────────────────────────────────────
//  オービットカメラ + 操作状態
// ─────────────────────────────────────────────────────────────
const camera = {
    target: [fluid.grid_X_num / 2, fluid.grid_Y_num / 2, fluid.grid_Z_num / 2],
    theta: Math.PI / 2,
    phi: 0.15,
    radius: Math.max(fluid.grid_X_num, fluid.grid_Y_num, fluid.grid_Z_num) * 1.6,
    fovy: Math.PI / 4,
    near: 0.1,
    far: 10000,
};

const handState = {
    pos: [0, 0, 0], vel: [0, 0, 0],
    radius: 6.0, strength: 1, active: false,
};

// カメラフレームの共通計算。毎フレーム呼ぶので事前確保したスクラッチへ書き込む
// (math.js の各関数は out 引数を取らないと新規確保する)。呼び出し側は結果を
// 同じ呼び出しの中で同期的に読むだけで、保持はしない。
const _camView = new Float32Array(16), _camProj = new Float32Array(16), _camViewProj = new Float32Array(16);
const _camCv = {
    eye: new Float32Array(3), right: new Float32Array(3), up: new Float32Array(3),
    forward: new Float32Array(3), // shade パスの環境マップサンプル (viewToWorldDir) 用
};
const _camFrame = { cv: _camCv, fovY: 0, view: _camView, proj: _camProj, viewProj: _camViewProj };
const WORLD_UP = [0, 1, 0];
function computeCameraFrame(aspect) {
    cameraVectors(camera, _camCv);
    // FOV は短辺基準 — 縦長画面でもシムが横にはみ出さないようにする。
    _camFrame.fovY = 2 * Math.atan(Math.tan(camera.fovy / 2) / Math.min(aspect, 1));
    mat4LookAt(_camCv.eye, camera.target, WORLD_UP, _camView);
    mat4Perspective(_camFrame.fovY, aspect, camera.near, camera.far, _camProj);
    mat4Multiply(_camProj, _camView, _camViewProj);
    return _camFrame;
}

// ─────────────────────────────────────────────────────────────
//  ポインタイベント
// ─────────────────────────────────────────────────────────────
{
    const orbit = { active: false, pointerId: null, lastX: 0, lastY: 0 };
    const iact  = { active: false, pointerId: null };
    // pointermove はドラッグ中 rAF より高頻度で発火し得るので、computeCameraFrame と
    // 同じ「事前確保したスクラッチへ out 引数で書く」パターンで毎イベントの配列確保を避ける。
    // _camCv は rAF ループ (computeCameraFrame) とも共有するが、JS はシングルスレッドで
    // 次に上書きされるのは次の rAF フレームなので競合しない。
    const _pmPos = new Float32Array(3);

    // ピンチズーム (2本指)。タッチのみ対象 — マウス/ペンは touches に一切乗らないので
    // 以下の分岐は素通りし、既存の1本指ロジックがそのまま動く。
    const touches = new Map(); // touch pointerId → {x, y} (クライアント座標)
    const pinch = { active: false, startDist: 0, startRadius: 0 };
    const touchDist = () => {
        const [a, b] = touches.values();
        return Math.hypot(a.x - b.x, a.y - b.y);
    };
    // 2本目の指が触れた瞬間、直前に1本目の指で始まっていたオービット/押す操作を打ち切る
    // (ピンチと同時に水を押し続けたり回転し続けたりすると意図しない操作になるため)。
    const cancelSingleFingerGestures = () => {
        if (iact.active) {
            iact.active = false; iact.pointerId = null;
            handState.active = false; handState.vel[0] = handState.vel[1] = handState.vel[2] = 0;
        }
        if (orbit.active) { orbit.active = false; orbit.pointerId = null; }
    };

    overlay.addEventListener('contextmenu', (e) => e.preventDefault());
    // iOS Safari は touch-action に関係なく pinch-zoom の gesture イベントを出すので塞ぐ。
    document.addEventListener('gesturestart', (e) => e.preventDefault());
    document.addEventListener('gesturechange', (e) => e.preventDefault());

    overlay.addEventListener('pointerdown', (e) => {
        if (e.pointerType === 'touch') {
            touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
            overlay.setPointerCapture(e.pointerId);
            if (touches.size >= 2) {
                cancelSingleFingerGestures();
                pinch.active = true;
                pinch.startDist = touchDist() || 1;
                pinch.startRadius = camera.radius;
                e.preventDefault();
                return;
            }
        }

        const aspect = c.width / c.height;
        const { cv, viewProj: vp } = computeCameraFrame(aspect);

        // モバイルは投影したシム境界の外をタップしたらオービット、内側なら押す。
        let wantOrbit = e.button === 2;
        if (!wantOrbit && isMobile) {
            const b = simBoundsOnScreen(
                [fluid.grid_X_num, fluid.grid_Y_num, fluid.grid_Z_num], vp, logicalW, logicalH);
            wantOrbit = b !== null && (
                e.offsetX < b[0] || e.offsetX > b[2] || e.offsetY < b[1] || e.offsetY > b[3]);
        }

        if (wantOrbit) {
            orbit.active = true; orbit.pointerId = e.pointerId;
            orbit.lastX = e.clientX; orbit.lastY = e.clientY;
            overlay.setPointerCapture(e.pointerId);
            e.preventDefault();
        } else {
            iact.active = true; iact.pointerId = e.pointerId;
            const p0 = screenToWorld(e.offsetX, e.offsetY, logicalW, logicalH, camera, cv);
            handState.pos = [
                Math.max(0, Math.min(fluid.grid_X_num, p0[0])),
                Math.max(0, Math.min(fluid.grid_Y_num, p0[1])),
                Math.max(0, Math.min(fluid.grid_Z_num, p0[2])),
            ];
            handState.active = true;
            overlay.setPointerCapture(e.pointerId);
        }
    });

    overlay.addEventListener('pointermove', (e) => {
        if (e.pointerType === 'touch' && touches.has(e.pointerId)) {
            touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
            if (pinch.active) {
                const d = touchDist() || 1;
                camera.radius = Math.max(1, Math.min(100000, pinch.startRadius * (pinch.startDist / d)));
                return;
            }
        }
        if (orbit.active && e.pointerId === orbit.pointerId) {
            const dx = e.clientX - orbit.lastX, dy = e.clientY - orbit.lastY;
            orbit.lastX = e.clientX; orbit.lastY = e.clientY;
            camera.theta -= dx * 0.01;
            camera.phi = Math.max(-(Math.PI / 2 - 0.01), Math.min(Math.PI / 2 - 0.01, camera.phi + dy * 0.01));
        }
        if (iact.active && e.pointerId === iact.pointerId) {
            // 絶対位置を毎回引き直す (差分を累積するとドリフトする)。事前確保した
            // _camCv/_pmPos へ書き込み、handState.pos は要素ごとに上書きする
            // (参照を差し替えると次のイベントで _pmPos の中身と衝突するため)。
            cameraVectors(camera, _camCv);
            screenToWorld(e.offsetX, e.offsetY, logicalW, logicalH, camera, _camCv, _pmPos);
            const vs = 2.0;
            handState.vel[0] = (_pmPos[0] - handState.pos[0]) * vs;
            handState.vel[1] = (_pmPos[1] - handState.pos[1]) * vs;
            handState.vel[2] = (_pmPos[2] - handState.pos[2]) * vs;
            handState.pos[0] = _pmPos[0];
            handState.pos[1] = _pmPos[1];
            handState.pos[2] = _pmPos[2];
        }
    });

    const endPointer = (e) => {
        if (e.pointerType === 'touch') {
            touches.delete(e.pointerId);
            if (touches.size < 2) pinch.active = false;
        }
        if (e.pointerId === iact.pointerId) {
            iact.active = false; iact.pointerId = null;
            handState.active = false; handState.vel[0] = handState.vel[1] = handState.vel[2] = 0;
        }
        if (e.pointerId === orbit.pointerId) { orbit.active = false; orbit.pointerId = null; }
    };
    overlay.addEventListener('pointerup', endPointer);
    overlay.addEventListener('pointercancel', endPointer);

    overlay.addEventListener('wheel', (e) => {
        e.preventDefault();
        camera.radius = Math.max(1, Math.min(100000, camera.radius * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
}

// ─────────────────────────────────────────────────────────────
//  メインループ
// ─────────────────────────────────────────────────────────────
let renderer = null;
let profiler = null;
let avgFrameMs = 0, lastFrameTime = performance.now();

function drawOverlay(cv, viewProj, frameMs) {
    octx.clearRect(0, 0, logicalW, logicalH);

    if (handState.active) {
        const sc = worldToScreen(handState.pos, viewProj, logicalW, logicalH);
        if (sc) {
            const dist = Math.hypot(
                handState.pos[0] - cv.eye[0], handState.pos[1] - cv.eye[1], handState.pos[2] - cv.eye[2]) || 1;
            const focal = 1 / Math.tan(camera.fovy / 2);
            const sr = Math.max(4, handState.radius / dist * focal * (logicalH / 2));
            octx.beginPath();
            octx.arc(sc[0], sc[1], sr, 0, Math.PI * 2);
            octx.strokeStyle = 'rgba(255,255,255,0.7)';
            octx.lineWidth = 1.5;
            octx.stroke();
        }
    }

    octx.font = '14px monospace';
    octx.fillStyle = '#aaa';
    let ln = 0;
    const nextY = () => 28 + 18 * ln++;
    octx.fillText(`frame: ${frameMs.toFixed(2)} ms`, 16, nextY());
    octx.fillText(`particles: ${fluid.active_particle_num} / ${fluid.particle_num}`, 16, nextY());
    octx.fillText(`grid: ${fluid.grid_X_num}x${fluid.grid_Y_num}x${fluid.grid_Z_num} (${fluid.grid_num} cells)`, 16, nextY());
    octx.fillText(isMobile
        ? 'drag in sim: push  drag outside: orbit  pinch: zoom'
        : 'L-drag: push fluid  R-drag: orbit  wheel: zoom', 16, nextY());

    // ?prof でパス単位の内訳を出す (GPU 時間 = EXT_disjoint_timer_query_webgl2)。
    if (profiler) {
        ln++;
        if (!profiler.gpuEnabled) {
            octx.fillStyle = '#f88';
            octx.fillText('GPU timer 拡張なし — CPU 時間のみ', 16, nextY());
        }
        octx.fillStyle = '#8cf';
        let total = 0;
        for (const r of profiler.report()) {
            if (r.name.startsWith('CPU')) continue;
            total += r.ms;
        }
        for (const r of profiler.report()) {
            const share = total > 0 && !r.name.startsWith('CPU') ? ` (${(r.ms / total * 100).toFixed(0)}%)` : '';
            octx.fillText(`${r.name.padEnd(18)} ${r.ms.toFixed(3)} ms${share}`, 16, nextY());
        }
        octx.fillText(`${'GPU total'.padEnd(18)} ${total.toFixed(3)} ms`, 16, nextY());
        // dropped が増え続けている = 重いパスの結果が回収できていない。表から消えている
        // 項目は「速い」のではなく「測れていない」(gl-profiler.js poll() のコメント参照)。
        if (profiler.dropped > 0) {
            octx.fillStyle = '#fc8';
            octx.fillText(`dropped (disjoint): ${profiler.dropped}`, 16, nextY());
        }
        octx.fillStyle = '#aaa';
    }
}

function startLoop() {
    // 固定タイムステップアキュムレータ。SIM_STEP_S = simFrame 1回ぶんの実時間、
    // MAX_SIM_STEPS で遅いフレームの追いつきを打ち切る (spiral-of-death 防止)。
    const SIM_STEP_S    = 1 / 60;
    const MAX_SIM_STEPS = 3;
    let simAccum = 0;

    function loop(timestamp) {
        const aspect = c.width / c.height;
        const deltaS = Math.min((timestamp - lastFrameTime) / 1000, 0.1);
        simAccum += deltaS;

        const { cv, fovY, view, proj, viewProj } = computeCameraFrame(aspect);

        let simSteps = 0;
        const tSim = performance.now();
        while (simAccum >= SIM_STEP_S && simSteps < MAX_SIM_STEPS) {
            fluid.simFrame(handState, cv.eye);
            simAccum -= SIM_STEP_S;
            simSteps++;
        }
        // GPU タイマーに映らないコスト (JS・ドライバの検証・コマンド積み) を切り分けるため
        // CPU 時間も別に測る。GL 呼び出しは非同期なのでこれは「発行にかかった時間」。
        if (profiler && simSteps > 0) profiler.cpu('CPU sim submit', performance.now() - tSim);
        // 実フレームコストが SIM_STEP_S*MAX_SIM_STEPS を超え続けると simAccum が
        // 際限なく溜まるので、追いつけなかったぶんは捨てる。
        const MAX_SIM_ACCUM = SIM_STEP_S * MAX_SIM_STEPS;
        if (simAccum > MAX_SIM_ACCUM) simAccum = MAX_SIM_ACCUM;
        handState.vel[0] = handState.vel[1] = handState.vel[2] = 0;

        renderer.draw(view, proj, viewProj, fovY, cv, fluid.active_particle_num);
        if (profiler) profiler.poll();

        const frameMs = timestamp - lastFrameTime;
        lastFrameTime = timestamp;
        avgFrameMs = avgFrameMs * 0.9 + frameMs * 0.1;
        drawOverlay(cv, viewProj, avgFrameMs);

        requestAnimationFrame(loop);
    }
    lastFrameTime = performance.now();
    requestAnimationFrame(loop);
}

// ─────────────────────────────────────────────────────────────
//  エントリポイント
// ─────────────────────────────────────────────────────────────
// スマホにはコンソールが無いので、例外は全部画面に出す。
function showError(err) {
    const msg = err instanceof Error ? `${err.message}\n\n${err.stack || ''}` : String(err);
    console.error(err);
    let el = document.querySelector('#error');
    if (!el) {
        el = document.createElement('pre');
        el.id = 'error';
        el.style.cssText = 'position:fixed;inset:0;margin:0;padding:16px;background:#200;color:#fbb;'
            + 'font:13px/1.5 monospace;white-space:pre-wrap;overflow:auto;z-index:10';
        document.body.appendChild(el);
    }
    el.textContent = msg;
}
window.addEventListener('error', (e) => showError(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => showError(e.reason));

async function init() {
    const gl = getGL2(c);
    fluid.initGL(gl);
    renderer = new NRFRenderer(gl, fluid);
    if (urlFlag('prof')) {
        profiler = new GLProfiler(gl);
        fluid.profiler = profiler;
        renderer.profiler = profiler;
    }
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);
    // 背景・反射・屈折用の環境マップ。ループ開始前に読み終える (起動が数百msブロックされる
    // だけなので、真っ黒な初期フレームを許容するより単純)。
    renderer.envTex = await loadEquirectTexture(gl, 'textures/suburban_garden_equirect.png');
    startLoop();
}

init().catch((err) => {
    if (err instanceof GLCapabilityError) showError(err);
    else throw err;
});
