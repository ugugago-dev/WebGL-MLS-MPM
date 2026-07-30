// WebGL2 の定型処理をまとめた薄いヘルパ群。状態を持たず、gl を第1引数に取る。

// P2G の散布加算 (RGBA32F への additive blend) に必須の拡張。EXT_color_buffer_float が
// あれば「32F へ描ける」、EXT_float_blend があれば「32F へブレンドできる」。後者は
// 別拡張で、これが無いと blendFunc(ONE,ONE) が INVALID_OPERATION になる (= P2G が成立しない)。
// RGBA16F へのフォールバックは作らない: 1セルあたり数百回の加算になるので half float では
// 精度が破綻する (plan のリスク #1)。
const REQUIRED_EXTENSIONS = ['EXT_color_buffer_float', 'EXT_float_blend'];

export class GLCapabilityError extends Error {}

// canvas から WebGL2 コンテキストを取り、必須拡張を有効化して返す。
// 足りないものがあれば GLCapabilityError を投げる (呼び出し側が画面に出す)。
export function getGL2(canvas, opts = {}) {
    const gl = canvas.getContext('webgl2', {
        alpha: false,
        antialias: false,
        depth: true,
        stencil: false,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: false,
        ...opts,
    });
    if (!gl) throw new GLCapabilityError('WebGL2 に対応していません (getContext("webgl2") が null)');

    const missing = REQUIRED_EXTENSIONS.filter((name) => !gl.getExtension(name));
    if (missing.length) {
        throw new GLCapabilityError(
            `必須の WebGL2 拡張がありません: ${missing.join(', ')}\n` +
            `浮動小数点テクスチャへの加算ブレンドが使えないため、P2G (粒子→グリッドの散布) が実装できません。`
        );
    }
    return gl;
}

// Transform Feedback 専用プログラムのフラグメントシェーダ。RASTERIZER_DISCARD 下では
// 実行されないが、リンクには必要なので空実装を置く。
export const NOOP_FS = `#version 300 es
precision highp float;
void main() {}
`;

function compile(gl, type, src, label) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error(`${label} のコンパイルに失敗:\n${log}\n${numberLines(src)}`);
    }
    return sh;
}

// コンパイルエラーのログは行番号でしか位置を示さないので、ソースにも行番号を振って出す。
function numberLines(src) {
    return src.split('\n').map((l, i) => `${String(i + 1).padStart(4)}| ${l}`).join('\n');
}

// vs/fs からプログラムを作る。tfVaryings を渡すと Transform Feedback 付き
// (INTERLEAVED_ATTRIBS 固定 — SEPARATE_ATTRIBS は WebGL2 の保証値が 4 varying しかなく、
// 粒子状態の 10 varying を出せない)。
export function makeProgram(gl, vsSrc, fsSrc, { tfVaryings = null, label = 'program' } = {}) {
    const vs = compile(gl, gl.VERTEX_SHADER, vsSrc, `${label} (vertex)`);
    const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc, `${label} (fragment)`);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    if (tfVaryings) gl.transformFeedbackVaryings(prog, tfVaryings, gl.INTERLEAVED_ATTRIBS);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(prog);
        gl.deleteProgram(prog);
        throw new Error(`${label} のリンクに失敗:\n${log}`);
    }
    return prog;
}

// プログラムの全 uniform ロケーションを名前 → location のオブジェクトで返す。
// 毎フレーム getUniformLocation を呼ばないための事前解決。
//
// 未知のキーには undefined ではなく null を返す (Proxy)。GLSL コンパイラは使われて
// いない uniform を消してしまうことがあり、そうなると ACTIVE_UNIFORMS に現れない。
// WebGL は location=null の uniform 設定を黙って無視する仕様なので null なら安全に
// 素通りするが、undefined を渡すと実装によっては例外になる。
export function uniformLocations(gl, prog) {
    const out = {};
    const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
        const info = gl.getActiveUniform(prog, i);
        // 配列 uniform は "name[0]" で列挙されるので、素の名前でも引けるようにする。
        const base = info.name.replace(/\[0\]$/, '');
        out[base] = gl.getUniformLocation(prog, info.name);
    }
    return new Proxy(out, {
        get(target, key) { return key in target ? target[key] : null; },
    });
}

export function makeBuffer(gl, target, dataOrSize, usage) {
    const buf = gl.createBuffer();
    gl.bindBuffer(target, buf);
    gl.bufferData(target, dataOrSize, usage);
    gl.bindBuffer(target, null);
    return buf;
}

// attribs: [{ loc, size, buffer, stride, offset, divisor }] (stride/offset はバイト)
// divisor 省略時は 0 (頂点ごと)。size は 1..4 の float 成分数。
export function makeVAO(gl, attribs) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    for (const a of attribs) {
        gl.bindBuffer(gl.ARRAY_BUFFER, a.buffer);
        gl.enableVertexAttribArray(a.loc);
        gl.vertexAttribPointer(a.loc, a.size, gl.FLOAT, false, a.stride | 0, a.offset | 0);
        gl.vertexAttribDivisor(a.loc, a.divisor | 0);
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vao;
}

// 2D テクスチャ生成の共通部分 (ミップマップ無し・CLAMP_TO_EDGE固定)。
// makeFloatTexture/makeColorTexture はフィルタと既定フォーマットだけが違う薄いラッパー。
function makeTexture(gl, w, h, internalFormat, filter) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, internalFormat, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
}

// 浮動小数点テクスチャ。フィルタは NEAREST 固定 (グリッドは texelFetch で読むし、
// 32F は線形フィルタ非対応の実装が多い)。
export function makeFloatTexture(gl, w, h, internalFormat = gl.RGBA32F) {
    return makeTexture(gl, w, h, internalFormat, gl.NEAREST);
}

// 正規化8bitカラーテクスチャ (RGBA8、LINEAR フィルタ)。FXAA blit の入力など、
// texelFetch ではなく texture() でバイリニアサンプルしたい通常のカラー中間バッファ用
// (makeFloatTexture は texelFetch 専用の NEAREST 固定なので分けてある)。
export function makeColorTexture(gl, w, h, internalFormat = gl.RGBA8) {
    return makeTexture(gl, w, h, internalFormat, gl.LINEAR);
}

// カラーアタッチメント (1枚以上) + 任意の深度レンダーバッファを持つ FBO。
// textures.length > 1 (MRT) のときは drawBuffers を明示しないと WebGL2 は
// COLOR_ATTACHMENT0 以外への書き込みを黙って無効化する (デフォルトの描画バッファは
// [COLOR_ATTACHMENT0, NONE, NONE, ...])。depthRb は makeDepthRenderbuffer() の
// 戻り値 (省略可、渡すとテストのみ・読み出さない深度バッファとして attach する)。
export function makeFBO(gl, textures, depthRb = null) {
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    textures.forEach((tex, i) => {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tex, 0);
    });
    if (textures.length > 1) {
        gl.drawBuffers(textures.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    }
    if (depthRb) {
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRb);
    }
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`FBO が不完全です (status=0x${status.toString(16)})`);
    }
    return fbo;
}

// 深度専用レンダーバッファ (書き込み専用、後から読まない用途 — NRFレンダラーの
// Pass1 が粒子同士のZテストにだけ使う)。テクスチャではなくレンダーバッファなのは
// サンプルする必要がないため (WebGPU 版の hwDepthTex/wallDepthTex と同じ位置づけ)。
export function makeDepthRenderbuffer(gl, w, h) {
    const rb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    return rb;
}

// equirectangular環境マップ (PNG) を2Dテクスチャとして読み込む。ミップマップ無し・LINEAR、
// 横方向(経度)はシームで繋がるようREPEAT、縦方向(緯度)は極で折り返さないようCLAMP_TO_EDGE。
// UNPACK_FLIP_Y_WEBGLは既定値(false)のまま — PNGの先頭行(パノラマの上端=空)がそのまま
// テクスチャの先頭行になり、シェーダ側のequirectUV()はv=0が空になるよう書いてある
// (nrf-shaders.js SHADE_FS 参照)。
export async function loadEquirectTexture(gl, url) {
    const res = await fetch(url);
    const bitmap = await createImageBitmap(await res.blob());
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, bitmap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    bitmap.close();
    return tex;
}

// デバッグ用: 直近の GL エラーを投げる。ホットパスでは呼ばないこと
// (getError は GPU との同期を強制する)。
export function assertNoGLError(gl, where) {
    const err = gl.getError();
    if (err !== gl.NO_ERROR) throw new Error(`GL error 0x${err.toString(16)} at ${where}`);
}
