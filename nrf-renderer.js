import { OFF, STATE_STRIDE } from './gl-shaders.js';
import { buildNRFShaders, PARTICLE_RADIUS, NRF_ITERATIONS, RENDER_SCALE, BG_DEPTH } from './nrf-shaders.js';
import {
    makeProgram, uniformLocations, makeFloatTexture, makeColorTexture, makeFBO, makeDepthRenderbuffer,
    makeBuffer, makeVAO,
} from './gl-utils.js';

// WebGPU版 fluid-renderer.js のNRF (Narrow-Range Filter) 表面レンダリングを移植した
// ParticleRenderer の後継。壁のデバッグ描画・spray/foam(diffuse粒子)は Sim_WebGL に
// 土台が無いため対象外。shade の反射・屈折はWebGPU版と厳密に同じ式・同じ定数値
// (固定skyColor反射 + 単色背景屈折、詳細は nrf-shaders.js SHADE_FS のコメント参照)。
//
// パス構成 (draw() 内、WebGPU版のPass1/T1/NRF/ThB/shadeに対応):
//   1. 深度   : 粒子ビルボード → R32F (rawTex) + 実深度テスト                [RENDER_SCALE解像度]
//   2. 厚み   : 粒子ビルボード → R16F (thickRawTex, 加算ブレンド)             [RENDER_SCALE解像度]
//   3. NRF    : 深度のみで決めるbilateralカーネルをH/V×NRF_ITERATIONS、MRTで厚みも同時フィルタ [同上]
//   4. 厚みブラー: 固定半径ガウシアンH/V (グリッド周波数の厚みムラ対策)         [同上]
//   5. shade  : 法線再構成 + Fresnel + Beer-Lambert → sceneColorTex (RGBA8)  [同上]
//   6. blit   : バイリニアで実解像度へアップスケール + FXAA → キャンバス
// RENDER_SCALE (既定0.5、nrf-shaders.js) は1〜5の全パスに共通の解像度スケール。
// WebGPU版のFLUID_RES_SCALE/RENDER_SCALEを1本化したもの — shadeがdepth/thicknessを
// texelFetchで直接読む構造 (アップサンプル無し) なので、shadeまでの全パスを同じ解像度に
// 揃えるのが最も単純 (詳細は nrf-shaders.js の RENDER_SCALE コメント参照)。
export class NRFRenderer {
    constructor(gl, fluid) {
        this.gl = gl;
        this.fluid = fluid;
        this.profiler = null;   // GLProfiler を挿すと各パスのGPU時間を計測する

        const src = buildNRFShaders({ hardMin: fluid.HARD_MIN });

        this.progDepth = makeProgram(gl, src.BILLBOARD_VS, src.DEPTH_FS, { label: 'nrf depth' });
        this.progThick = makeProgram(gl, src.BILLBOARD_VS, src.THICK_FS, { label: 'nrf thickness' });
        this.progNrfH  = makeProgram(gl, src.FULLSCREEN_VS, src.NRF_H_FS, { label: 'nrf filter H' });
        this.progNrfV  = makeProgram(gl, src.FULLSCREEN_VS, src.NRF_V_FS, { label: 'nrf filter V' });
        this.progThickSmoothH = makeProgram(gl, src.FULLSCREEN_VS, src.THICK_SMOOTH_H_FS, { label: 'thick smooth H' });
        this.progThickSmoothV = makeProgram(gl, src.FULLSCREEN_VS, src.THICK_SMOOTH_V_FS, { label: 'thick smooth V' });
        this.progShade = makeProgram(gl, src.FULLSCREEN_VS, src.SHADE_FS, { label: 'nrf shade' });
        this.progBlit  = makeProgram(gl, src.FULLSCREEN_VS, src.BLIT_FS, { label: 'nrf blit (FXAA)' });

        this.uDepth = uniformLocations(gl, this.progDepth);
        this.uThick = uniformLocations(gl, this.progThick);
        this.uNrfH  = uniformLocations(gl, this.progNrfH);
        this.uNrfV  = uniformLocations(gl, this.progNrfV);
        this.uThickSmoothH = uniformLocations(gl, this.progThickSmoothH);
        this.uThickSmoothV = uniformLocations(gl, this.progThickSmoothV);
        this.uShade = uniformLocations(gl, this.progShade);
        this.uBlit  = uniformLocations(gl, this.progBlit);

        // 半径・静止密度は実行中に変わらないので depth/thickness プログラムへ一度だけ送る
        // (fluid-gl.js _setStaticUniforms と同じ理由・パターン)。
        gl.useProgram(this.progDepth);
        gl.uniform1f(this.uDepth.uHalfSize, PARTICLE_RADIUS);
        gl.uniform1f(this.uDepth.uRestDensity, fluid.REST_DENSITY);
        gl.useProgram(this.progThick);
        gl.uniform1f(this.uThick.uHalfSize, PARTICLE_RADIUS);
        gl.uniform1f(this.uThick.uRestDensity, fluid.REST_DENSITY);
        gl.useProgram(this.progNrfH);
        gl.uniform1i(this.uNrfH.uDepthTex, 0);
        gl.uniform1i(this.uNrfH.uThickTex, 1);
        gl.useProgram(this.progNrfV);
        gl.uniform1i(this.uNrfV.uDepthTex, 0);
        gl.uniform1i(this.uNrfV.uThickTex, 1);
        gl.useProgram(this.progThickSmoothH);
        gl.uniform1i(this.uThickSmoothH.uSrcTex, 0);
        gl.useProgram(this.progThickSmoothV);
        gl.uniform1i(this.uThickSmoothV.uSrcTex, 0);
        gl.useProgram(this.progShade);
        gl.uniform1i(this.uShade.uDepthTex, 0);
        gl.uniform1i(this.uShade.uThickTex, 1);
        gl.useProgram(this.progBlit);
        gl.uniform1i(this.uBlit.uSceneTex, 0);
        gl.useProgram(null);

        // 3頂点の外接三角形コーナー (WebGPU版 particleShader.vs と同じ形)。divisor=0固定の
        // 静的VBO — WebGL2の「インスタンス描画にはdivisor=0の属性が最低1つ必要」という
        // 制約 (P2G散布の ineighbor と同じ役目) をこれで満たす。
        this.cornerBuf = makeBuffer(
            gl, gl.ARRAY_BUFFER,
            new Float32Array([0, 2, -1.7320508, -1, 1.7320508, -1]),
            gl.STATIC_DRAW,
        );

        // 状態バッファ (A/B ping-pong) それぞれに対応するビルボードVAO。
        // aPos(loc0)/aVel(loc1)/aDensity(loc2) はdivisor=1で fluid.stateBuf を直接読む
        // (fluid-gl.js の STATE_STRIDE/OFF と同じレイアウト)、aCorner(loc3)はdivisor=0
        // (gl-utils.js の makeVAO ヘルパーをそのまま流用)。
        this.billboardVAO = fluid.stateBuf.map((buf) => makeVAO(gl, [
            { loc: 0, size: 3, buffer: buf, stride: STATE_STRIDE, offset: OFF.pos * 4, divisor: 1 },
            { loc: 1, size: 3, buffer: buf, stride: STATE_STRIDE, offset: OFF.vel * 4, divisor: 1 },
            { loc: 2, size: 1, buffer: buf, stride: STATE_STRIDE, offset: OFF.density * 4, divisor: 1 },
            { loc: 3, size: 2, buffer: this.cornerBuf, stride: 0, offset: 0, divisor: 0 },
        ]));

        this._w = 0; this._h = 0;
    }

    _beginPass(name) { if (this.profiler) this.profiler.begin(name); }
    _endPass() { if (this.profiler) this.profiler.end(); }

    // Pass1(深度)・Pass T1(厚み)の両ビルボードパスが共有するカメラuniformの設定
    // (uProjだけはPass1限定で呼び出し側が別途設定する — THICK_FSにはuProjが無い)。
    _setBillboardCamera(u, view, viewProj, cv) {
        const gl = this.gl;
        gl.uniformMatrix4fv(u.uViewProj, false, viewProj);
        gl.uniformMatrix4fv(u.uView, false, view);
        gl.uniform3f(u.uCamRight, cv.right[0], cv.right[1], cv.right[2]);
        gl.uniform3f(u.uCamUp, cv.up[0], cv.up[1], cv.up[2]);
    }

    // 頂点バッファ無しのフルスクリーン三角形パス (NRFフィルタ・厚みブラー) が共有する
    // FBO/プログラム/入力テクスチャのbind定型作業。fluid-gl.js の _runTF/_runSplat と
    // 同じ「パスの機械的な部分だけ共通化する」役割 (パス固有のuniformとdraw呼び出しは
    // 呼び出し側に残す)。
    _runFullscreen(fbo, prog, textures) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.useProgram(prog);
        textures.forEach((tex, i) => {
            gl.activeTexture(gl.TEXTURE0 + i);
            gl.bindTexture(gl.TEXTURE_2D, tex);
        });
    }

    // canvas解像度に合わせて全オフスクリーンリソースを(再)確保する。サイズ不変なら
    // 何もしない (WebGPU版 rebuildTextures と同じガード)。Pass1〜shadeは
    // RENDER_SCALE倍のサイズ (sw,sh) で確保し、blitがそこから実解像度 (w,h) へ
    // アップスケールする。
    resize(w, h) {
        this.viewW = w; this.viewH = h;
        if (w === this._w && h === this._h) return;
        this._w = w; this._h = h;

        const sw = Math.max(1, Math.round(w * RENDER_SCALE));
        const sh = Math.max(1, Math.round(h * RENDER_SCALE));
        this.renderW = sw; this.renderH = sh;

        const gl = this.gl;
        this._destroyTextures();

        // texs/fbos は生成した端からその場で積む (末尾でまとめて列挙し直さない) —
        // 新しいテクスチャ/FBOを足したときに破棄リストへの追記を忘れるミスを構造的に防ぐ。
        const texs = [], fbos = [];
        const tex2d = (make, ...args) => { const t = make(gl, ...args); texs.push(t); return t; };
        const fbo   = (...args) => { const f = makeFBO(gl, ...args); fbos.push(f); return f; };

        this.rawTex      = tex2d(makeFloatTexture, sw, sh, gl.R32F);
        this.depthRb     = makeDepthRenderbuffer(gl, sw, sh);
        this.depthFBO    = fbo([this.rawTex], this.depthRb);

        this.thickRawTex = tex2d(makeFloatTexture, sw, sh, gl.R16F);
        this.thickRawFBO = fbo([this.thickRawTex]);

        this.filtA  = tex2d(makeFloatTexture, sw, sh, gl.R32F);
        this.filtB  = tex2d(makeFloatTexture, sw, sh, gl.R32F);
        this.thickA = tex2d(makeFloatTexture, sw, sh, gl.R16F);
        this.thickB = tex2d(makeFloatTexture, sw, sh, gl.R16F);
        this.nrfFBO_A = fbo([this.filtA, this.thickA]);
        this.nrfFBO_B = fbo([this.filtB, this.thickB]);
        // 厚みブラー用の単一アタッチメントFBO (NRFのMRT FBOと同じテクスチャを指す別FBOオブジェクト)。
        this.thickOnlyFBO_A = fbo([this.thickA]);
        this.thickOnlyFBO_B = fbo([this.thickB]);

        // shadeの出力先 (RENDER_SCALE解像度、blitがバイリニア+FXAAで実解像度へ拡大する)。
        // texelFetchではなく texture() でバイリニアサンプルするので makeFloatTexture
        // ではなく makeColorTexture (RGBA8、LINEAR) を使う。
        this.sceneColorTex = tex2d(makeColorTexture, sw, sh);
        this.sceneFBO      = fbo([this.sceneColorTex]);

        this._texs = texs; this._fbos = fbos;
    }

    _destroyTextures() {
        const gl = this.gl;
        (this._texs || []).forEach((t) => gl.deleteTexture(t));
        (this._fbos || []).forEach((f) => gl.deleteFramebuffer(f));
        if (this.depthRb) gl.deleteRenderbuffer(this.depthRb);
    }

    // view/proj/viewProj: Float32Array(16)。cv: {eye,right,up} (cameraVectors の出力)。
    // fovY: ラジアン (短辺基準に補正済み)。count: 描画する粒子数。
    draw(view, proj, viewProj, fovY, cv, count) {
        const gl = this.gl;
        const fluid = this.fluid;
        if (count === 0) return;
        const w = this.viewW, h = this.viewH;
        const sw = this.renderW, sh = this.renderH;
        const vao = this.billboardVAO[fluid.cur];
        const aspect = w / h;
        const tanHalfFovY = Math.tan(fovY * 0.5);

        gl.viewport(0, 0, sw, sh);

        // ── Pass1: 深度 ──────────────────────────────────────────────
        this._beginPass('1 depth');
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.depthFBO);
        gl.clearColor(BG_DEPTH, 0, 0, 1);
        gl.clearDepth(1.0);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LESS);
        gl.disable(gl.BLEND);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(this.progDepth);
        this._setBillboardCamera(this.uDepth, view, viewProj, cv);
        gl.uniformMatrix4fv(this.uDepth.uProj, false, proj);
        gl.bindVertexArray(vao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 3, count);
        this._endPass();

        // ── Pass T1: 厚み (加算ブレンド、深度テストなし) ──────────────
        this._beginPass('2 thickness');
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.thickRawFBO);
        gl.clearColor(0, 0, 0, 0);
        gl.disable(gl.DEPTH_TEST);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.useProgram(this.progThick);
        this._setBillboardCamera(this.uThick, view, viewProj, cv);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 3, count);
        gl.disable(gl.BLEND);
        this._endPass();

        // ── NRF 1Dフィルタ (H→V を NRF_ITERATIONS 回、ping-pong) ──────
        // ここから先はどのパスも頂点バッファ無しのフルスクリーン三角形 (gl_VertexID駆動) —
        // fluid.emptyVAO を一度bindしたまま、フレーム末尾まで使い回す
        // (fluid-gl.js の GRID_UPDATE_FS と同じ空VAOを流用、重複確保しない)。
        this._beginPass('3 nrf');
        gl.bindVertexArray(fluid.emptyVAO);
        // uProjはH/Vそれぞれのプログラムに一度だけ送れば十分 (ループ内では不変)。
        gl.useProgram(this.progNrfH); gl.uniformMatrix4fv(this.uNrfH.uProj, false, proj);
        gl.useProgram(this.progNrfV); gl.uniformMatrix4fv(this.uNrfV.uProj, false, proj);
        let srcDepth = this.rawTex, srcThick = this.thickRawTex;
        for (let iter = 0; iter < NRF_ITERATIONS; iter++) {
            this._runFullscreen(this.nrfFBO_A, this.progNrfH, [srcDepth, srcThick]);
            gl.drawArrays(gl.TRIANGLES, 0, 3);

            this._runFullscreen(this.nrfFBO_B, this.progNrfV, [this.filtA, this.thickA]);
            gl.drawArrays(gl.TRIANGLES, 0, 3);

            srcDepth = this.filtB; srcThick = this.thickB;
        }
        this._endPass();

        // ── 厚み専用ガウシアンブラー (H→V、thickB→thickA→thickB) ──────
        this._beginPass('4 thick blur');
        this._runFullscreen(this.thickOnlyFBO_A, this.progThickSmoothH, [this.thickB]);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        this._runFullscreen(this.thickOnlyFBO_B, this.progThickSmoothV, [this.thickA]);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        this._endPass();

        // ── shade: 最終深度(filtB) + 最終厚み(thickB) → sceneColorTex (RENDER_SCALE解像度) へ描画 ──
        // viewportはPass1〜4と同じ (sw,sh) のまま変わっていないので再設定不要。
        this._beginPass('5 shade');
        this._runFullscreen(this.sceneFBO, this.progShade, [this.filtB, this.thickB]);
        gl.clearColor(0.08, 0.09, 0.11, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.uniform2f(this.uShade.uScreenRes, sw, sh);
        gl.uniform1f(this.uShade.uTanHalfFovY, tanHalfFovY);
        gl.uniform1f(this.uShade.uAspect, aspect);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        this._endPass();

        // ── blit: sceneColorTex (RENDER_SCALE解像度) をバイリニアで実解像度へ
        // アップスケールしつつFXAAをかけてキャンバスへ最終書き込み ──────────
        this._beginPass('6 blit');
        this._runFullscreen(null, this.progBlit, [this.sceneColorTex]);
        gl.viewport(0, 0, w, h);
        gl.clearColor(0.08, 0.09, 0.11, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.uniform2f(this.uBlit.uCanvasRes, w, h);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        this._endPass();
    }
}
