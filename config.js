// WebGL2 版の設定。WebGPU 版 (../config.js) とは意図的に別ファイル — グリッド解像度も
// 粒子数もタイムステップも WebGL2 の性能に合わせて下げてあり (CLAUDE.md「今後の展望」#4 の
// 縮小案)、共通化すると片方を触ったときにもう片方が巻き添えになる。

export const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.innerWidth < 768;

export const dpr = isMobile ? 1 : Math.min(window.devicePixelRatio || 1, 2);

// グリッドの粒子半径パラメータ。FluidGL のコンストラクタが 1/(radius*4) をセル数の
// スケールに使うので、値を上げるほどグリッドが粗くなる。
// desktop 0.015 → 34×34×50 = 57,800 セル (WebGPU 版 desktop は 0.0125 で 40×40×60 = 96,000)。
export const PARTICLE_RADIUS = isMobile ? 0.02 : 0.0175;

export const PARTICLE_COUNT = isMobile ? 20000 : 100000;

// ─────────────────────────────────────────────────────────────
//  Diagnostic URL overrides (?p=50000 など)。実機の負荷切り分け用で、
//  未指定なら呼び出し側の既定値がそのまま使われる。
// ─────────────────────────────────────────────────────────────
const _q = new URLSearchParams(location.search);
export function urlNum(key, def)  { const v = parseFloat(_q.get(key)); return Number.isFinite(v) ? v : def; }
export function urlFlag(key)      { return _q.has(key); }
