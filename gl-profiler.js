// パス単位の GPU 時間計測 (EXT_disjoint_timer_query_webgl2)。
//
// WebGL には WebGPU の timestamp-query に相当する「任意の2点間」の計測が無く、
// TIME_ELAPSED クエリで囲んだ区間の経過時間しか取れない。しかも
// **同時にアクティブにできるクエリは1つだけ**なので、パスを入れ子にせず順番に
// 囲んでいく必要がある (begin→end→begin→end)。
//
// 結果は数フレーム遅れて利用可能になるので、投げっぱなしにして poll() で回収する。
// 同期待ちするとパイプラインストールになり、計測そのものが結果を壊す。
//
// CPU 側の時間も別途測る: ドライバの検証コストや JS のオーバーヘッドが支配的な場合、
// GPU タイマーには何も映らないため。
//
// ── 拡張が無い環境のフォールバック (syncMode、2026-07-31) ──────────────────────
// **モバイルブラウザの多くはこの拡張を出さない** (タイミング攻撃対策で無効化されている)。
// その場合は各パスをGPU完了待ちで挟んで実時間を測る方式に自動で切り替える。
//
// **待ちは 1×1 FBO からの `readPixels`** (`_waitGPU()`)。他の2つは Chrome では効かない
// ことをヘッドレスで実測済み — **再挑戦しないこと**:
//   - `gl.finish()`: コマンドバッファの処理完了までしか待たない。全パス 0.0xx ms になり、
//     同条件の TIME_ELAPSED (14〜56ms) と全く合わなかった
//   - `fenceSync` + `clientWaitSync(sync, 0, 0)` のビジーポーリング: フェンスの状態は
//     タスク境界でしか更新されないので、同一タスク内でいくら回しても TIMEOUT_EXPIRED の
//     まま。全パスが打ち切り上限に張り付いた
// readPixels だけは CPU が実データを受け取る必要があるため、ブラウザが必ずGPU完了を待つ。
// 実測 (SwiftShader、TIME_ELAPSED との突き合わせ) では**順位は完全一致**、値は同期
// オーバーヘッドぶん +2〜9ms 上振れした。
//
// **syncMode の数値は絶対値として読んではいけない**: パスごとにパイプラインを空にするので
//   ①パス間のオーバーラップが消える ②readPixels 1回ぶんの往復コストが各パスに乗る
//   (軽いパスほど相対的に大きく水増しされる) ③フレーム全体も当然遅くなる
//   (計測中の fps は信用しない)。
// **読めるのは「どのパスが重いか」の順位と比率だけ**。それが分かれば十分なので割り切る。
export class GLProfiler {
    // forceSync: 拡張があっても syncMode を使う (desktop で両者を突き合わせる診断用)。
    constructor(gl, { forceSync = false } = {}) {
        this.gl = gl;
        this.ext = forceSync ? null : gl.getExtension('EXT_disjoint_timer_query_webgl2');
        this.gpuEnabled = !!this.ext;
        this.syncMode = !this.gpuEnabled;
        this.pool = [];          // 使い終わった WebGLQuery の再利用プール
        this.pending = [];       // { name, query } — 結果待ち
        this.active = null;      // 現在アクティブなクエリ (常に高々1つ)
        this.stats = new Map();  // name -> { ms, n } — 指数移動平均
        this.order = [];         // 表示順 (初出順)
        this.dropped = 0;        // disjoint で捨てたクエリ数 (下の poll() 参照)

        if (this.syncMode) {
            // GPU完了待ち用の 1×1 RGBA8 FBO (_waitGPU 参照)。中身は読み捨てる。
            const tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 1, 1);
            gl.bindTexture(gl.TEXTURE_2D, null);
            this.syncFBO = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.syncFBO);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            this._syncPx = new Uint8Array(4);
        }
    }

    _record(name, ms) {
        let s = this.stats.get(name);
        if (!s) { s = { ms, n: 0 }; this.stats.set(name, s); this.order.push(name); }
        // 指数移動平均。フレームごとの揺れを均しつつ、変化には数十フレームで追従する。
        s.ms = s.ms * 0.9 + ms * 0.1;
        s.n++;
    }

    // syncMode 用: 発行済みコマンドのGPU実行完了までブロックする。
    // 1×1 の専用FBOから readPixels する — CPUが実データを受け取る必要があるので、
    // ブラウザは必ずGPUの完了を待つ (WebGLで同期を強制する唯一の確実な手段)。
    _waitGPU() {
        const gl = this.gl;
        const prevFBO = gl.getParameter(gl.FRAMEBUFFER_BINDING);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.syncFBO);
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this._syncPx);
        gl.bindFramebuffer(gl.FRAMEBUFFER, prevFBO);
    }

    // GPU 区間の開始。end() を呼ぶまで次の begin() はできない。
    begin(name) {
        if (this.active) return;
        const gl = this.gl;
        if (this.syncMode) {
            // 直前までの仕事を出し切ってから測り始める (前のパスの残りを
            // このパスに計上しないため)。
            this._waitGPU();
            this.active = { name, t0: performance.now() };
            return;
        }
        const query = this.pool.pop() || gl.createQuery();
        gl.beginQuery(this.ext.TIME_ELAPSED_EXT, query);
        this.active = { name, query };
    }

    end() {
        if (!this.active) return;
        if (this.syncMode) {
            this._waitGPU();
            this._record(this.active.name, performance.now() - this.active.t0);
            this.active = null;
            return;
        }
        this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
        this.pending.push(this.active);
        this.active = null;
    }

    // CPU 時間を直接記録する (GPU クエリで囲めない区間や、JS 側のコスト用)。
    cpu(name, ms) { this._record(name, ms); }

    // 結果が揃ったクエリを回収する。毎フレーム末に1回呼ぶ。
    poll() {
        if (!this.gpuEnabled) return;
        const gl = this.gl;
        // GPU_DISJOINT はこのフレーム中に GPU のクロックが乱れた (省電力遷移など) ことを
        // 示す。どのクエリが影響を受けたかは特定できないので、待機中を全部捨てる。
        //
        // 注意: 1フレームで終わらない重いパスは、結果が揃う前に disjoint に当たり続けて
        // **永久に回収されない**ことがある (ヘッドレスの SwiftShader で実際に起きた —
        // 270ms かかる散布パスだけがオーバーレイから消え、軽いパスだけが表示された)。
        // 表から項目が消えていたら「速い」のではなく「測れていない」ので、dropped を見ること。
        if (gl.getParameter(this.ext.GPU_DISJOINT_EXT)) {
            this.dropped += this.pending.length;
            for (const p of this.pending) this.pool.push(p.query);
            this.pending.length = 0;
            return;
        }
        let i = 0;
        while (i < this.pending.length) {
            const p = this.pending[i];
            if (!gl.getQueryParameter(p.query, gl.QUERY_RESULT_AVAILABLE)) { i++; continue; }
            this._record(p.name, gl.getQueryParameter(p.query, gl.QUERY_RESULT) / 1e6);
            this.pool.push(p.query);
            this.pending.splice(i, 1);
        }
    }

    // [{ name, ms }] を初出順で返す。
    report() {
        return this.order.map((name) => ({ name, ms: this.stats.get(name).ms }));
    }
}
