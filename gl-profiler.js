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
export class GLProfiler {
    constructor(gl) {
        this.gl = gl;
        this.ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
        this.gpuEnabled = !!this.ext;
        this.pool = [];          // 使い終わった WebGLQuery の再利用プール
        this.pending = [];       // { name, query } — 結果待ち
        this.active = null;      // 現在アクティブなクエリ (常に高々1つ)
        this.stats = new Map();  // name -> { ms, n } — 指数移動平均
        this.order = [];         // 表示順 (初出順)
        this.dropped = 0;        // disjoint で捨てたクエリ数 (下の poll() 参照)
    }

    _record(name, ms) {
        let s = this.stats.get(name);
        if (!s) { s = { ms, n: 0 }; this.stats.set(name, s); this.order.push(name); }
        // 指数移動平均。フレームごとの揺れを均しつつ、変化には数十フレームで追従する。
        s.ms = s.ms * 0.9 + ms * 0.1;
        s.n++;
    }

    // GPU 区間の開始。end() を呼ぶまで次の begin() はできない。
    begin(name) {
        if (!this.gpuEnabled || this.active) return;
        const gl = this.gl;
        const query = this.pool.pop() || gl.createQuery();
        gl.beginQuery(this.ext.TIME_ELAPSED_EXT, query);
        this.active = { name, query };
    }

    end() {
        if (!this.active) return;
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
