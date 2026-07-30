import {
    STATE_FLOATS, STATE_STRIDE, OFF, ATTR, TF_VARYINGS,
    PROGRAM_ATTRIBS, buildShaders,
} from './gl-shaders.js';
import {
    makeBuffer, makeFloatTexture, makeFBO, makeVAO,
    makeProgram, uniformLocations, NOOP_FS,
} from './gl-utils.js';

// WebGL2 版の MLS-MPM。WebGPU 版 (../fluid-gpu.js) と物理パラメータ・境界条件は同じだが、
// compute shader も atomic も無いので実行機構が根本的に違う:
//   - 粒子状態の更新は Transform Feedback (状態バッファを A/B で ping-pong)
//   - グリッドは RGBA32F テクスチャ、P2G は GL_POINTS 散布 + 加算ブレンド
//   - グリッド更新はフルスクリーン三角形 (read/write 分離のためテクスチャも 2 枚)
// 壁セル (SDF)・diffuse 粒子・E/R の粒子追加削除は未実装 (スコープ外)。
export class FluidGL {
    constructor(aspectX, aspectY, aspectZ, particleRadius, particleNum) {
        const normToGrid = 1 / (particleRadius * 4);
        this.grid_X_num = Math.ceil(aspectX * normToGrid);
        this.grid_Y_num = Math.ceil(aspectY * normToGrid);
        this.grid_Z_num = Math.ceil(aspectZ * normToGrid);
        this.grid_num   = this.grid_X_num * this.grid_Y_num * this.grid_Z_num;

        this.particle_num = particleNum;
        this.active_particle_num = 0;

        // 粒子状態のインターリーブ配列 (レイアウトは gl-shaders.js OFF/STATE_FLOATS)。
        // 初期状態を1回アップロードするためだけの CPU 側バッファで、以降は GPU 上で
        // TF が回すので読み返さない。
        this.state = new Float32Array(particleNum * STATE_FLOATS);

        // 粒子固有の固定ジッタ (grid imprinting 対策)。生成時に1回だけ決まる固定ベクトルを
        // G2P で毎サブステップ位置に微小加算し続ける。乱数を毎フレーム引き直すのではなく
        // 「スロットごとに固定」なのが要点 — 固定方向なら粒子同士が quadratic B-spline の
        // 周期的に「きれいな」相対位置へ完全に揃うことが構造的に起きなくなる。
        // TF では書き換えないので状態バッファには入れず、独立した静的 VBO にする。
        this.jitter = new Float32Array(particleNum * 3);
        for (let i = 0; i < particleNum * 3; i++) this.jitter[i] = Math.random() * 2 - 1;

        // ── シミュレーションパラメータ (WebGPU 版から 1:1、DT/SUBSTEPS のみ縮小案) ──
        // sub_dt = DT/SUBSTEPS = 0.3。WebGPU 版の 0.2 より粗いので CFL クランプ到達率が
        // 上がる。暴れる場合は SUBSTEPS を増やすのではなく DT を下げること
        // (1 substep が GL 7 パスなので substep 追加のコストが WebGPU 版より遥かに高い)。
        this.DT           = 0.3;
        this.SUBSTEPS     = 1;
        this.REST_DENSITY = 8.0;
        this.STIFFNESS    = 100.0;
        this.EOS_POWER    = 1.0;
        this.VISCOSITY    = 0.01;
        this.GRAVITY      = -0.98;

        // Splash 方式の予測位置バネ補正。床バネ (Y軸) が使う。
        this.WALL_STIFFNESS = 1.0;
        this.LOOKAHEAD_K    = 2.0;

        const HM = 2;
        this.HARD_MIN   = HM;
        this.HARD_MAX_X = this.grid_X_num - HM;
        this.HARD_MAX_Y = this.grid_Y_num - HM;
        this.HARD_MAX_Z = this.grid_Z_num - HM;

        this.gl = null;
        this.cur = 0;   // 状態バッファの ping-pong インデックス

        // true にすると simFrame の各パス直後に getError する。GPU との同期を強制するので
        // 常用不可だが、どのパスで落ちたかを特定できないと WebGL のエラーは追えない。
        this.debugPasses = false;

        // GLProfiler を挿すとパス単位の GPU 時間を計測する (?prof で有効化)。
        this.profiler = null;
    }

    _check(where) {
        if (!this.debugPasses) return;
        const err = this.gl.getError();
        if (err !== this.gl.NO_ERROR) throw new Error(`GL error 0x${err.toString(16)} after ${where}`);
    }

    // パスの計測区間。TIME_ELAPSED は同時に1つしかアクティブにできないので、
    // 必ず _beginPass → _endPass の対で、入れ子にせず順番に使うこと。
    _beginPass(name) { if (this.profiler) this.profiler.begin(name); }
    _endPass(name) {
        if (this.profiler) this.profiler.end();
        this._check(name);
    }

    // x0..z1: 各軸 0–1 の正規化座標。粒子は 0.5 セル間隔で配置する。
    // 複数回呼ぶと現在の active_particle_num から積み上がる。
    fillBlock(x0, y0, z0, x1, y1, z1) {
        const step = 0.5, jitter = 0.05;
        const gx0 = x0 * this.grid_X_num, gx1 = x1 * this.grid_X_num;
        const gy0 = y0 * this.grid_Y_num, gy1 = y1 * this.grid_Y_num;
        const gz0 = z0 * this.grid_Z_num, gz1 = z1 * this.grid_Z_num;
        const s = this.state;
        let p = this.active_particle_num;
        outer:
        for (let gz = gz0 + step / 2; gz < gz1; gz += step) {
            for (let gy = gy0 + step / 2; gy < gy1; gy += step) {
                for (let gx = gx0 + step / 2; gx < gx1; gx += step) {
                    if (p >= this.particle_num) break outer;
                    const b = p * STATE_FLOATS;
                    s[b + OFF.pos]     = gx + (Math.random() - 0.5) * jitter;
                    s[b + OFF.pos + 1] = gy + (Math.random() - 0.5) * jitter;
                    s[b + OFF.pos + 2] = gz + (Math.random() - 0.5) * jitter;
                    s[b + OFF.mass]    = 1.0;
                    p++;
                }
            }
        }
        this.active_particle_num = p;
        return p;
    }

    // GPU リソースを確保して初期状態をアップロードする。fillBlock() の後に1回だけ呼ぶ。
    initGL(gl) {
        this.gl = gl;

        // ── 粒子状態バッファ (A/B ping-pong) ──
        // TF は読み込み中のバッファへ書けないので必ず2枚要る。DYNAMIC_COPY = GPU が書き
        // GPU が読む (CPU からは触らない) というヒント。
        const byteLen = this.particle_num * STATE_STRIDE;
        this.stateBuf = [
            makeBuffer(gl, gl.ARRAY_BUFFER, byteLen, gl.DYNAMIC_COPY),
            makeBuffer(gl, gl.ARRAY_BUFFER, byteLen, gl.DYNAMIC_COPY),
        ];
        gl.bindBuffer(gl.ARRAY_BUFFER, this.stateBuf[0]);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.state);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        this.jitterBuf = makeBuffer(gl, gl.ARRAY_BUFFER, this.jitter, gl.STATIC_DRAW);

        // P2G 散布のインスタンス描画で使う 0..26 の近傍番号 (ATTR.ineighbor のコメント参照)。
        this.neighborBuf = makeBuffer(
            gl, gl.ARRAY_BUFFER, Float32Array.from({ length: 27 }, (_, i) => i), gl.STATIC_DRAW);

        // ── グリッドテクスチャ ──
        // massTex (R32F)  : P2G 質量散布の加算ブレンド先
        // momTex  (RGBA32F): P2G 運動量散布の加算ブレンド先 (xyz、w は未使用)
        // velTex  (RGBA32F): グリッド更新の出力 = 速度 (xyz)。G2P が読む
        //
        // 質量を運動量と同じ RGBA32F に相乗りさせず分けてあるのは、ブレンドが
        // read-modify-write だから — RGBA32F なら 1 フラグメント 32B 動くところ、
        // R32F なら 8B で済む。**RGB32F は WebGL2 では color-renderable ではない**ので
        // (R32F/RG32F/RGBA32F だけ)、運動量・速度側は RGBA32F のままにする。
        // 読みながら書けないので、更新の入力 (mass/mom) と出力 (vel) も別テクスチャ。
        const tw = this.grid_X_num, th = this.grid_Y_num * this.grid_Z_num;
        this.gridTexW = tw; this.gridTexH = th;
        this.massTex = makeFloatTexture(gl, tw, th, gl.R32F);
        this.momTex  = makeFloatTexture(gl, tw, th, gl.RGBA32F);
        this.velTex  = makeFloatTexture(gl, tw, th, gl.RGBA32F);
        this.massFBO = makeFBO(gl, [this.massTex]);
        this.momFBO  = makeFBO(gl, [this.momTex]);
        this.velFBO  = makeFBO(gl, [this.velTex]);

        this.tf = gl.createTransformFeedback();

        this._buildVAOs();
        this._buildPrograms(gl);
    }

    _buildPrograms(gl) {
        const src = buildShaders({
            gridX: this.grid_X_num, gridY: this.grid_Y_num, gridZ: this.grid_Z_num,
        });

        // TF プログラムはラスタライズしない (RASTERIZER_DISCARD) のでフラグメント
        // シェーダは空実装でよいが、リンクには必要。
        const tf = { tfVaryings: TF_VARYINGS };
        this.progPressure = makeProgram(gl, src.PRESSURE_VS, NOOP_FS, { ...tf, label: 'pressure (TF)' });
        this.progG2P      = makeProgram(gl, src.G2P_VS, NOOP_FS,      { ...tf, label: 'g2p (TF)' });
        this.progP2GMass  = makeProgram(gl, src.P2G_MASS_VS, src.SPLAT_FS, { label: 'p2g mass' });
        this.progP2GMom   = makeProgram(gl, src.P2G_MOM_VS, src.SPLAT_FS,  { label: 'p2g momentum' });
        this.progGridUpd  = makeProgram(gl, src.FULLSCREEN_VS, src.GRID_UPDATE_FS, { label: 'grid update' });

        this.uPressure = uniformLocations(gl, this.progPressure);
        this.uG2P      = uniformLocations(gl, this.progG2P);
        this.uGridUpd  = uniformLocations(gl, this.progGridUpd);

        // 属性を1つも使わない描画 (フルスクリーン三角形) 用の空 VAO。直前の VAO の
        // 有効な属性配列が残っていると描画エラーになるので、専用に1つ持っておく。
        this.emptyVAO = gl.createVertexArray();

        this._setStaticUniforms();
    }

    // pressure/gridUpd/g2p の uniform のうち、マウス操作 (uHandOn 系) を除く全ては
    // コンストラクタで決まる定数 (dt, 物理パラメータ, hard clamp 境界, テクスチャユニット
    // 番号) で実行中に変わらない。以前は simFrame が毎substep丸ごと再送信していたが、
    // GLの uniform 値はプログラムにひも付いて保持されるので、build 直後に一度だけ
    // 送っておけば十分 — hand 関連だけを simFrame 側の動的送信として残す。
    _setStaticUniforms() {
        const gl = this.gl;
        const subDt = this.DT / this.SUBSTEPS;

        gl.useProgram(this.progPressure);
        gl.uniform1i(this.uPressure.uMass, 0);
        gl.uniform1f(this.uPressure.uDt, subDt);
        gl.uniform1f(this.uPressure.uRestDensity, this.REST_DENSITY);
        gl.uniform1f(this.uPressure.uStiffness, this.STIFFNESS);
        gl.uniform1f(this.uPressure.uEosPower, this.EOS_POWER);
        gl.uniform1f(this.uPressure.uViscosity, this.VISCOSITY);

        gl.useProgram(this.progGridUpd);
        gl.uniform1i(this.uGridUpd.uMass, 0);
        gl.uniform1i(this.uGridUpd.uMom, 1);
        gl.uniform1f(this.uGridUpd.uDt, subDt);
        gl.uniform1f(this.uGridUpd.uGravity, this.GRAVITY);

        gl.useProgram(this.progG2P);
        gl.uniform1i(this.uG2P.uVel, 0);
        gl.uniform1f(this.uG2P.uDt, subDt);
        gl.uniform1f(this.uG2P.uWallStiffness, this.WALL_STIFFNESS);
        gl.uniform1f(this.uG2P.uLookaheadK, this.LOOKAHEAD_K);
        // E/R (粒子の追加・削除) 未実装で粒子数が固定なので、床バネの水量スケールは常に 1.0。
        // E/R を実装したら active/最大観測値の比を動的に送るよう戻すこと。
        gl.uniform1f(this.uG2P.uFloorFillFactor, 1.0);
        gl.uniform1f(this.uG2P.uHardMin, this.HARD_MIN);
        gl.uniform3f(this.uG2P.uHardMax, this.HARD_MAX_X, this.HARD_MAX_Y, this.HARD_MAX_Z);

        gl.useProgram(null);
    }

    // プログラムごとに「そのシェーダが実際に宣言している属性だけ」を有効にした VAO を組む。
    // 全属性を常に有効化していた頃は、質量散布 (使うのは pos と mass だけ) でも 27 頂点が
    // ステート全体を読み直していた。
    //
    // 属性の集合は gl-shaders.js の PROGRAM_ATTRIBS が単一の宣言で、シェーダのソースも
    // そこから生成される。**ここで名前を手書きで足し引きしてはいけない** —
    // シェーダが宣言していて VAO 側で無効な属性は GL エラーにならず定数 (0,0,0,1) として
    // 読まれるので、落ちずに物理だけ静かに壊れる。
    _vaoFor(names, buf, divisor) {
        const st = STATE_STRIDE;
        // 状態バッファ内の属性 (divisor は用途による) と、外部バッファの属性 (常に divisor 0)。
        const inState = {
            ipos:     { loc: ATTR.ipos,     size: 3, offset: OFF.pos * 4 },
            ivel:     { loc: ATTR.ivel,     size: 3, offset: OFF.vel * 4 },
            imass:    { loc: ATTR.imass,    size: 1, offset: OFF.mass * 4 },
            iC0:      { loc: ATTR.iC0,      size: 3, offset: OFF.C0 * 4 },
            iC1:      { loc: ATTR.iC1,      size: 3, offset: OFF.C1 * 4 },
            iC2:      { loc: ATTR.iC2,      size: 3, offset: OFF.C2 * 4 },
            idensity: { loc: ATTR.idensity, size: 1, offset: OFF.density * 4 },
        };
        const external = {
            ijitter:   { loc: ATTR.ijitter,   size: 3, buffer: this.jitterBuf,   stride: 12 },
            ineighbor: { loc: ATTR.ineighbor, size: 1, buffer: this.neighborBuf, stride: 4 },
        };
        return makeVAO(this.gl, names.map((n) => {
            if (inState[n]) return { ...inState[n], buffer: buf, stride: st, divisor };
            if (external[n]) return { ...external[n], offset: 0, divisor: 0 };
            throw new Error(`未知の属性名: ${n}`);
        }));
    }

    // 用途ごと × 状態バッファ A/B。散布パスは 27頂点 × 粒子数のインスタンス描画なので
    // 粒子属性は divisor 1、それ以外は 1頂点 = 1粒子なので divisor 0。
    _buildVAOs() {
        const perBuf = (names, divisor) =>
            this.stateBuf.map((buf) => this._vaoFor(names, buf, divisor));

        this.massSplatVAO = perBuf(PROGRAM_ATTRIBS.p2gMass, 1);
        this.momSplatVAO  = perBuf(PROGRAM_ATTRIBS.p2gMom, 1);
        this.pressureVAO  = perBuf(PROGRAM_ATTRIBS.pressure, 0);
        this.g2pVAO       = perBuf(PROGRAM_ATTRIBS.g2p, 0);
    }

    // Transform Feedback を1パス実行し、状態バッファを ping-pong する。
    // 読み込み中のバッファへは書けないので、必ず「今読んでいない側」へ出力する。
    _runTF(prog, vaos, count) {
        const gl = this.gl;
        gl.enable(gl.RASTERIZER_DISCARD);
        gl.useProgram(prog);
        gl.bindVertexArray(vaos[this.cur]);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.stateBuf[1 - this.cur]);
        gl.beginTransformFeedback(gl.POINTS);
        gl.drawArrays(gl.POINTS, 0, count);
        gl.endTransformFeedback();
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
        gl.bindVertexArray(null);
        gl.disable(gl.RASTERIZER_DISCARD);
        this.cur = 1 - this.cur;
    }

    // 1粒子あたり 27 点 (3×3×3 近傍) を加算ブレンドで散布する = P2G。
    // WebGL2 には atomic が無いのでこれが唯一の手段。
    _runSplat(prog, vaos, count) {
        const gl = this.gl;
        gl.useProgram(prog);
        gl.bindVertexArray(vaos[this.cur]);
        gl.drawArraysInstanced(gl.POINTS, 0, 27, count);
        gl.bindVertexArray(null);
    }

    // hand: { pos, vel, radius, strength, active }、eye: カメラ位置 (押し円柱の軸に使う)。
    simFrame(hand = null, eye = null) {
        const gl = this.gl;
        const count = this.active_particle_num;
        if (!gl || count === 0) return;

        // グリッドへの描画中は深度・カリング・ディザ等が邪魔をしないようにしておく。
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE);
        gl.viewport(0, 0, this.gridTexW, this.gridTexH);

        for (let step = 0; step < this.SUBSTEPS; step++) {
            // 1. 質量グリッドをクリアして質量を散布 (R32F へ加算ブレンド)
            this._beginPass('1 splat mass');
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.massFBO);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE);
            this._runSplat(this.progP2GMass, this.massSplatVAO, count);
            this._endPass('1 splat mass');

            // 2. 密度・圧力・応力 (TF)。質量グリッドを読む。
            // 描画先の FBO を bind したままそのテクスチャをサンプルするとフィードバック
            // ループ扱いで INVALID_OPERATION になる (RASTERIZER_DISCARD 中でも draw 時に
            // 検証される)。読む前に必ず FBO を外すこと。
            this._beginPass('2 pressure TF');
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.disable(gl.BLEND);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.massTex);
            // dt・物理パラメータ・テクスチャユニットは initGL 時に一度だけ送信済み
            // (_setStaticUniforms、実行中に変わらないため)。
            this._runTF(this.progPressure, this.pressureVAO, count);
            this._endPass('2 pressure TF');

            // 3. 運動量グリッドをクリアして運動量を散布
            this._beginPass('3 splat momentum');
            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.momFBO);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE);
            this._runSplat(this.progP2GMom, this.momSplatVAO, count);
            gl.disable(gl.BLEND);
            this._endPass('3 splat momentum');

            // 4. グリッド更新 (mass, mom) → vel (重力・境界・摩擦・CFL・マウス押し)
            this._beginPass('4 grid update');
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.velFBO);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.massTex);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, this.momTex);
            gl.useProgram(this.progGridUpd);
            // dt・重力・テクスチャユニットは initGL 時に一度だけ送信済み (_setStaticUniforms)。
            // マウス操作だけが実際にフレームごとに変わる値。
            const handOn = hand && hand.active && eye ? 1 : 0;
            gl.uniform1i(this.uGridUpd.uHandOn, handOn);
            if (handOn) {
                gl.uniform3f(this.uGridUpd.uHandPos, hand.pos[0], hand.pos[1], hand.pos[2]);
                gl.uniform3f(this.uGridUpd.uHandVel, hand.vel[0], hand.vel[1], hand.vel[2]);
                gl.uniform3f(this.uGridUpd.uHandEye, eye[0], eye[1], eye[2]);
                gl.uniform1f(this.uGridUpd.uHandRadius, hand.radius);
                gl.uniform1f(this.uGridUpd.uHandStrength, hand.strength);
            }
            gl.bindVertexArray(this.emptyVAO);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
            gl.bindVertexArray(null);
            this._endPass('4 grid update');

            // 5. G2P + 移流 (TF)。速度グリッドを読む。
            this._beginPass('5 g2p TF');
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.velTex);
            // dt・境界パラメータ・テクスチャユニットは initGL 時に一度だけ送信済み
            // (_setStaticUniforms、E/R 未実装で uFloorFillFactor は固定 1.0 のまま)。
            this._runTF(this.progG2P, this.g2pVAO, count);
            this._endPass('5 g2p TF');
        }

        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // デバッグ用: 現在の状態バッファを CPU に読み戻す (同期する = ホットパスでは使わない)。
    readState(count = this.active_particle_num) {
        const gl = this.gl;
        const out = new Float32Array(count * STATE_FLOATS);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.stateBuf[this.cur]);
        gl.getBufferSubData(gl.ARRAY_BUFFER, 0, out);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        return out;
    }
}
