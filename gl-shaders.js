// GLSL ES 3.00 シェーダ群。WebGPU 版の shaders.js に対応する。
//
// グリッド寸法はテンプレートリテラルで焼き込む (WebGPU 版が uniform で渡しているのに対し、
// こちらは定数化して texelFetch のインデックス計算をコンパイル時に畳めるようにする)。
// 生成は buildShaders() 一箇所で行い、寸法違いのソースが混ざらないようにする。

// ── 粒子状態のインターリーブレイアウト (Transform Feedback の varying 順と一致必須) ──
// TF は「読んでいるバッファへ書けない」ため状態バッファを A/B で ping-pong する。
// 2 つの TF パス (pressure / g2p) は**全て同じ varying 一式**を出力する
// (更新しないものは素通しでコピー) — こうしないとパスごとにバッファのレイアウトが
// 変わってしまい ping-pong できない。ref/refjs.txt の copyAll() が同じ理由で存在する。
//
// **quadratic B-spline の重みはここに入れない** (2026-07-30 の最適化)。以前は
// wX/wY/wZ の 9 float を専用パスで焼いて全消費者が読んでいたが、散布パスは 1 粒子
// 27 頂点なので同じ 36 バイトを 27 回フェッチすることになり、属性帯域が支配的だった。
// pos はサブステップ中不変なので、各パスで cellAndWeights() を呼んで導出すれば
// 同じ値が得られる (下の「重みは全パスで同じ式から導く」も参照)。
export const STATE_FLOATS = 17;
export const STATE_STRIDE = STATE_FLOATS * 4;
export const OFF = {
    pos: 0, vel: 3, mass: 6,
    C0: 7, C1: 10, C2: 13,      // affine (速度勾配) 行列 3×3
    density: 16,
};
// TF varying の順序。OFF の並びと厳密に一致していること。
export const TF_VARYINGS = ['pos', 'vel', 'mass', 'C0', 'C1', 'C2', 'density'];

// 頂点属性ロケーション。粒子状態を読む全プログラムで共通。
export const ATTR = {
    ipos: 0, ivel: 1, imass: 2,
    iC0: 3, iC1: 4, iC2: 5,
    idensity: 6,
    ijitter: 7,
    // 3×3×3近傍のうちどのZスライス(0..2)を担当するかを示す番号 (2026-07-31、27点→
    // 3点散布への最適化で「近傍27個のどれか」から意味が変わった)。gl_VertexID でも
    // 同じ値は得られるが、WebGL2 のインスタンス描画は「divisor が 0 の有効な属性が
    // 最低1つ」必要で、粒子属性を全部 divisor=1 にすると INVALID_OPERATION になる。
    // この属性が divisor=0 側を兼ねる。
    islice: 8,
};

// ── プログラムごとの入力属性 (単一の宣言) ──
// fluid-gl.js はこの表から VAO を組み、シェーダのソースもここから生成する。
// **VAO とシェーダで属性の集合がズレると GL エラーにはならず、無効な属性は定数
// (0,0,0,1) として読まれる** = 落ちずに物理だけ静かに壊れる。だから二重管理せず
// ここ 1 箇所から両方を導く。
const DECL = {
    ipos:      `layout(location = ${ATTR.ipos})      in vec3  ipos;`,
    ivel:      `layout(location = ${ATTR.ivel})      in vec3  ivel;`,
    imass:     `layout(location = ${ATTR.imass})     in float imass;`,
    iC0:       `layout(location = ${ATTR.iC0})       in vec3  iC0;`,
    iC1:       `layout(location = ${ATTR.iC1})       in vec3  iC1;`,
    iC2:       `layout(location = ${ATTR.iC2})       in vec3  iC2;`,
    idensity:  `layout(location = ${ATTR.idensity})  in float idensity;`,
    ijitter:   `layout(location = ${ATTR.ijitter})   in vec3  ijitter;`,
    islice:    `layout(location = ${ATTR.islice})    in float islice;`,
};

// 状態を丸ごと読む TF パス用 (copyAll() が全 varying を埋めるので全属性が要る)。
const STATE_NAMES = ['ipos', 'ivel', 'imass', 'iC0', 'iC1', 'iC2', 'idensity'];

export const PROGRAM_ATTRIBS = {
    // 質量散布は pos と mass しか使わない。重みは pos から導出する。
    p2gMass:  ['ipos', 'imass', 'islice'],
    // 運動量散布は M (=C0..C2) と vel/mass、それに density (棄却判定) を使う。
    p2gMom:   ['ipos', 'ivel', 'imass', 'iC0', 'iC1', 'iC2', 'idensity', 'islice'],
    pressure: STATE_NAMES,
    g2p:      [...STATE_NAMES, 'ijitter'],
};

function declare(names) { return names.map((n) => DECL[n]).join('\n'); }

const STATE_OUT = `
out vec3  pos;
out vec3  vel;
out float mass;
out vec3  C0;
out vec3  C1;
out vec3  C2;
out float density;

// 全 varying を入力のまま埋める。各 TF パスはこの後、自分が更新する分だけ上書きする
// (更新しない varying も必ず書かないとレイアウトが崩れる)。
void copyAll() {
    pos = ipos; vel = ivel; mass = imass;
    C0 = iC0; C1 = iC1; C2 = iC2;
    density = idensity;
}
`;

// 頂点バッファ無しのフルスクリーン三角形。グリッド非依存 (シム側の buildShaders() と
// NRFレンダラー側の両方が使うので、独立した named export として持つ — 重複させない)。
export const FULLSCREEN_VS = `#version 300 es
precision highp float;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export function buildShaders({ gridX, gridY, gridZ }) {
    // 3D グリッドの 2D テクスチャ展開: texel = (cx, cz*gridY + cy)。
    // 参照実装 (ref/refjs.txt) はビットマスク/シフトで展開しており寸法が 2 の冪である
    // 必要があるが、この素直な展開なら 34×34×50 のような非 2 冪でもそのまま扱える。
    const GRID = `
const ivec3 GRID = ivec3(${gridX}, ${gridY}, ${gridZ});
const ivec2 GRID_TEX = ivec2(${gridX}, ${gridY * gridZ});

ivec2 cellTexel(ivec3 c) { return ivec2(c.x, c.z * GRID.y + c.y); }

// ── 不変条件: 3×3×3 ステンシルは常にグリッド内 (STENCIL_IN_GRID) ────────────────
// 粒子位置は G2P の hardClampAxis が毎 substep [HARD_MIN, dim-HARD_MIN] = [2, dim-2] へ
// 押し込み、初期配置の fillBlock も同じ範囲にクランプしている。したがって
// cellI = floor(pos) ∈ [2, dim-2] で、ステンシルの範囲 [cellI-1, cellI+1] ⊂ [1, dim-1] は
// 必ずグリッド内に収まる。**これが全パスから 27 回/頂点の範囲チェックを外せる根拠**
// (P2G 散布の「隣の Z スライスへ染み出さない」議論と同じ不変条件)。
// 物理が NaN を出した場合だけ texelFetch が範囲外になるが、WebGL2 の texelFetch は
// 範囲外で未定義値を返すだけでクラッシュも安全性の問題も起こさない。
// **この不変条件を崩す変更 (HARD_MIN を 2 未満にする、fillBlock のクランプを外す等) を
// するなら、範囲チェックを全パスへ戻すこと。**
`;

    // WebGPU 版 WGSL_COMMON から移植した共通部分。
    const COMMON = GRID + `
// EOS 圧力の下限 (凝集圧)。静止密度付近だけ弱い負圧を許し、まばらな領域では 0 にする。
// まばらな粒子が負圧で自己クラスター化する (tensile instability) のを防ぐゲート。
const float COHESION_PRESSURE     = -0.1;
const float COHESION_DENSITY_GATE = 0.7;

// quadratic B-spline 重み (WGSL の quadratic_weights と同一)。
void quadraticWeights(vec3 f, out vec3 wx, out vec3 wy, out vec3 wz) {
    wx = vec3(0.5*(0.5-f.x)*(0.5-f.x), 0.75-f.x*f.x, 0.5*(0.5+f.x)*(0.5+f.x));
    wy = vec3(0.5*(0.5-f.y)*(0.5-f.y), 0.75-f.y*f.y, 0.5*(0.5+f.y)*(0.5+f.y));
    wz = vec3(0.5*(0.5-f.z)*(0.5-f.z), 0.75-f.z*f.z, 0.5*(0.5+f.z)*(0.5+f.z));
}

// **重みは全パスでこの関数から導く**。MLS-MPM は P2G と G2P が厳密に同じ重みを使うことに
// 依存しているので、パスごとに式を書き分けてはいけない (コンパイラの FMA の畳み方が変わって
// 値が 1 ulp ずれ得る)。散布パスは 27 頂点それぞれ 1 成分しか使わないが、そこでも
// 「3 成分作って添字で引く」— 式が同一なら値も必ず同一になる。
void cellAndWeights(vec3 p, out ivec3 cellI, out vec3 f, out vec3 wx, out vec3 wy, out vec3 wz) {
    cellI = ivec3(floor(p));
    f = p - vec3(cellI) - 0.5;
    quadraticWeights(f, wx, wy, wz);
}
`;

    // P2G 散布 (頂点シェーダ側のみで使う共通部分)。
    const SPLAT_COMMON = `
// 範囲外セルへ散布しようとした点をクリップ空間の外へ飛ばして捨てる
// (discard より安く、ラスタライズ自体が起きない)。
const vec4 CULLED = vec4(-2.0, -2.0, -2.0, 1.0);

// セル座標 → クリップ空間。1 セル = 1 texel なので点の大きさは常に 1。
vec4 cellClipPos(ivec3 c) {
    vec2 t = vec2(cellTexel(c)) + 0.5;
    return vec4(t / vec2(GRID_TEX) * 2.0 - 1.0, 0.0, 1.0);
}

// Zスライス dz(0..2) から3×3ブロックの基準セル (cellI-1, cellI-1, cellI.z-1+dz) を組み立てる。
// P2G_MASS_VS/P2G_MOM_VS で共通のロジックなのでここに1箇所だけ書く。
// cz の範囲チェックは不要 — STENCIL_IN_GRID の保証による。
ivec3 sliceBaseCell(ivec3 cellI, int dz) {
    return ivec3(cellI.x - 1, cellI.y - 1, cellI.z - 1 + dz);
}
`;

    // P2G 散布 (フラグメントシェーダ側のみで使う共通部分。discard はフラグメント
    // シェーダでのみ有効な文なので、VS用の SPLAT_COMMON とは別の文字列に分けてある)。
    const SPLAT_FS_COMMON = `
// gl_FragCoordから3×3ブロック内の相対位置(dx,dy)を復元する。P2G_MASS_FS/P2G_MOM_FS で共通。
// 点の大きさは常に 3.0 なので local は必ず [0,2]、さらに baseCell が指す 3×3 は
// STENCIL_IN_GRID によりグリッド内に収まるので、セル自体の範囲チェックは不要。
// local の範囲だけは残してある — ラスタライズが 1 テクセルでもずれると vec3 の範囲外
// 添字 (GLSL では未定義動作) になるため、そこだけは保険で捨てる。
ivec2 recoverLocalXY(ivec3 baseCell) {
    ivec2 t = ivec2(gl_FragCoord.xy);
    ivec2 local = ivec2(t.x - baseCell.x, t.y - (baseCell.z * GRID.y + baseCell.y));
    if (any(lessThan(local, ivec2(0))) || any(greaterThan(local, ivec2(2)))) discard;
    return local;
}
`;

    // ── パス1: P2G 質量散布 ── WGSL_P2G_MASS 相当。
    // atomicAdd の代わりに加算ブレンドで散布する。2026-07-31 に 27頂点/粒子 → 3頂点/粒子へ
    // 最適化済み: テクスチャ配置が texel=(cx, cz*gridY+cy) なので、Zスライスを固定すれば
    // (dx,dy) の3×3近傍がテクセル空間でちょうど連続した3×3ブロックになる。そこで
    // gl_PointSize=3.0 の点1個でその9セルをまとめてラスタライズする (WebGL2の点は非AAなので
    // 3×3が正確に埋まる)。ALIASED_POINT_SIZE_RANGEが3未満の環境では成立しないため、
    // gl-utils.js getGL2() が起動時に検証して足りなければ GLCapabilityError で落とす。
    // フラグメント側で gl_FragCoord から実際のテクセル位置を復元して重みを掛け直す。
    // ドロー回数は1/9、ブレンド書き込み回数(=P2Gの実コスト)自体は変わらない。
    // 出力先は R32F の質量専用テクスチャ (RGBA32F だとブレンドの read-modify-write が
    // 1 フラグメント 32B になるが、R32F なら 8B で済む)。
    const P2G_MASS_VS = `#version 300 es
precision highp float;
${COMMON}
${declare(PROGRAM_ATTRIBS.p2gMass)}
${SPLAT_COMMON}

flat out ivec3 vBaseCell;  // このインスタンスの3×3ブロックの基準セル (cellI-1, cellI-1, cz)
flat out vec3  vWx, vWy;   // FSがdx/dyで引く重み
flat out float vWzImass;   // wz[dz]*imass — このインスタンスで固定な係数は先に畳んでおく

void main() {
    int dz = int(islice); // 0, 1, 2 (Zスライスごとに1点、3インスタンス分)
    ivec3 cellI; vec3 f, wx, wy, wz;
    cellAndWeights(ipos, cellI, f, wx, wy, wz);

    // STENCIL_IN_GRID により棄却条件が無いので、このVSは完全に分岐なし。
    ivec3 baseCell = sliceBaseCell(cellI, dz);

    vBaseCell  = baseCell;
    vWx = wx; vWy = wy;
    vWzImass = wz[dz] * imass;
    gl_Position  = cellClipPos(ivec3(cellI.x, cellI.y, baseCell.z)); // 3×3ブロックの中心セル
    gl_PointSize = 3.0;
}
`;

    // FSはCOMMON(GRID/inGrid)とSPLAT_FS_COMMON(recoverLocalXY)だけで足りる —
    // wx/wy/wzはVSで計算済みの値をそのまま受け取るので quadraticWeights の呼び直しは無い
    // (「重みは全パスで同じ関数から導く」原則はVS側のcellAndWeights呼び出し1箇所を
    // 全消費者が共有する形で満たしている、CULLED/cellClipPosはVS専用)。
    const P2G_MASS_FS = `#version 300 es
precision highp float;
${COMMON}
${SPLAT_FS_COMMON}
flat in ivec3 vBaseCell;
flat in vec3  vWx, vWy;
flat in float vWzImass;
layout(location = 0) out vec4 oColor;

void main() {
    ivec2 xy = recoverLocalXY(vBaseCell);
    float w = vWx[xy.x] * vWy[xy.y];
    oColor = vec4(w * vWzImass, 0.0, 0.0, 0.0); // R32F なので .r だけが書かれる
}
`;

    // ── パス2: 密度・圧力 (TF) ── WGSL_P2G_MOM の前半。
    // WebGL2 では1点が1テクセルしか書けないので運動量散布は 27 点描画になるが、
    // その各点で密度を 27 タップし直すと 729 タップになる。密度・圧力・応力の算出を
    // この先行 TF パスへ切り出し、散布パスは結果を読むだけにする
    // (ref/refjs.txt の同名パスが存在する理由と同じ)。
    //
    // 散布パスが必要なのは w*(mass*vel + M*d) の M = mass*A + coef*S だけなので、
    // affine A を持つ C0..C2 スロットをそのまま M で上書きする。A はこの substep の
    // 運動量散布でしか使われず、G2P が毎 substep 新しい A を書き直すので安全
    // (ref/refjs.txt も gvel0..2 を同じように潰している)。
    const PRESSURE_VS = `#version 300 es
precision highp float;
${COMMON}
${declare(PROGRAM_ATTRIBS.pressure)}
${STATE_OUT}

uniform sampler2D uMass;
uniform float uDt;
uniform float uRestDensity;
uniform float uStiffness;
uniform float uEosPower;
uniform float uViscosity;

void main() {
    copyAll();

    ivec3 cellI; vec3 f, wx, wy, wz;
    cellAndWeights(ipos, cellI, f, wx, wy, wz);
    ivec3 base = cellI - 1;

    // 範囲チェックは STENCIL_IN_GRID により不要。テクセルのY座標 (cz*GRID.y + cy) も
    // k/j ループへ巻き上げて、27 回の整数乗算を 3 回に減らしてある。
    // **重みの掛け算の順序 (wx*wy)*wz は変えないこと** — MLS-MPM は P2G と G2P が
    // 厳密に同じ重みを使うことに依存しており、wy*wz を先に畳むと丸めが 1ulp ずれ得る
    // (散布側 P2G_MOM_FS の w も (vWx*vWy)*vWzD の順序)。
    float dens = 0.0;
    int ty0 = base.z * GRID.y + base.y;
    for (int k = 0; k < 3; k++) {
        int tyk = ty0 + k * GRID.y;
        for (int j = 0; j < 3; j++) {
            int ty = tyk + j;
            for (int i = 0; i < 3; i++) {
                dens += wx[i] * wy[j] * wz[k] * texelFetch(uMass, ivec2(base.x + i, ty), 0).r;
            }
        }
    }
    density = dens;
    if (dens <= 1e-8) { C0 = vec3(0.0); C1 = vec3(0.0); C2 = vec3(0.0); return; }

    // 体積推定に使う密度だけクランプする: 孤立した粒子は局所密度がほぼ 0 になり、
    // vol = mass/density が巨大な力に化ける (静止しているのに弾かれて見える)。
    // varying の density は未クランプのまま — 描画側が孤立粒子の縮小に使う。
    float vol = imass / max(dens, uRestDensity * 0.15);
    float pressure = uStiffness * (pow(dens / uRestDensity, uEosPower) - 1.0);
    // 凝集圧の下限は静止密度付近でだけ有効。まばらな領域で負圧をかけると
    // 粒子が自己クラスター化する (tensile instability)。
    float pFloor = dens > uRestDensity * COHESION_DENSITY_GATE ? COHESION_PRESSURE : 0.0;
    pressure = max(pressure, pFloor);

    float visc = uViscosity;
    float s00 = -pressure + visc * (iC0.x + iC0.x);
    float s11 = -pressure + visc * (iC1.y + iC1.y);
    float s22 = -pressure + visc * (iC2.z + iC2.z);
    float s01 = visc * (iC0.y + iC1.x);
    float s02 = visc * (iC0.z + iC2.x);
    float s12 = visc * (iC1.z + iC2.y);

    float coef = -vol * 4.0 * uDt;

    // M = mass*A + coef*S (行優先で C0/C1/C2 に格納)
    C0 = imass * iC0 + coef * vec3(s00, s01, s02);
    C1 = imass * iC1 + coef * vec3(s01, s11, s12);
    C2 = imass * iC2 + coef * vec3(s02, s12, s22);
}
`;

    // ── パス3: P2G 運動量散布 ── WGSL_P2G_MOM の後半。質量散布と同じ27→3頂点/粒子の
    // 最適化を適用 (詳細はP2G_MASS_VSのコメント参照)。d(=セル中心相対位置)のうちZ成分は
    // インスタンスごとに固定なので、FSで復元するdx/dyだけからd全体を組み直す。
    const P2G_MOM_VS = `#version 300 es
precision highp float;
${COMMON}
${declare(PROGRAM_ATTRIBS.p2gMom)}
${SPLAT_COMMON}

flat out ivec3 vBaseCell;
flat out vec3  vWx, vWy;
flat out float vWzD;     // wz[dz] (このインスタンスで固定)
flat out vec2  vFxy;     // d.x/d.y をFSで組むための端数位置
flat out float vDzTerm;  // d.z = float(dz)-1.0-f.z (このインスタンスで固定なので先に計算)
flat out vec3  vMassVel; // imass*ivel
flat out vec3  vC0, vC1, vC2;

void main() {
    // 密度がほぼ 0 の粒子は運動量を散布しない (WGSL 版が early return する条件と同じ)。
    int dz = int(islice);
    ivec3 cellI; vec3 f, wx, wy, wz;
    cellAndWeights(ipos, cellI, f, wx, wy, wz);

    // 棄却は密度条件のみ (セル範囲は STENCIL_IN_GRID で保証されている)。
    if (idensity <= 1e-8) {
        gl_Position = CULLED; gl_PointSize = 1.0;
        return;
    }
    ivec3 baseCell = sliceBaseCell(cellI, dz);

    vBaseCell = baseCell;
    vWx = wx; vWy = wy; vWzD = wz[dz];
    vFxy = f.xy;
    vDzTerm  = float(dz) - 1.0 - f.z;
    vMassVel = imass * ivel;
    vC0 = iC0; vC1 = iC1; vC2 = iC2;
    gl_Position  = cellClipPos(ivec3(cellI.x, cellI.y, baseCell.z));
    gl_PointSize = 3.0;
}
`;

    const P2G_MOM_FS = `#version 300 es
precision highp float;
${COMMON}
${SPLAT_FS_COMMON}
flat in ivec3 vBaseCell;
flat in vec3  vWx, vWy;
flat in float vWzD;
flat in vec2  vFxy;
flat in float vDzTerm;
flat in vec3  vMassVel;
flat in vec3  vC0, vC1, vC2;
layout(location = 0) out vec4 oColor;

void main() {
    ivec2 xy = recoverLocalXY(vBaseCell);
    float w = vWx[xy.x] * vWy[xy.y] * vWzD;

    // M は行優先で C0/C1/C2 に入っている (PRESSURE_VS 参照)。mat3() は列優先で
    // 組むので、取り違えないよう dot で明示的に行×ベクトルにする。
    vec3 d = vec3(float(xy.x) - 1.0 - vFxy.x, float(xy.y) - 1.0 - vFxy.y, vDzTerm);
    vec3 md = vec3(dot(vC0, d), dot(vC1, d), dot(vC2, d));
    oColor = vec4(w * (vMassVel + md), 0.0);
}
`;

    // ── パス4: グリッド更新 ── WGSL_UPDATE_GRID + WGSL_APPLY_HAND 相当。
    // 質量 (R32F) と運動量 (RGBA32F) を読んで速度グリッドへ書く。同じテクスチャを
    // 読みながら書けないので出力は別の RGBA32F。
    const GRID_UPDATE_FS = `#version 300 es
precision highp float;
${COMMON}

uniform sampler2D uMass;
uniform sampler2D uMom;
uniform float uDt;
uniform float uGravity;

// マウス押し (WGSL_APPLY_HAND)。カメラ光線を軸とする無限円柱との距離で影響範囲を決める
// ので、水面がその光線上のどこにあっても正しく効く。
uniform int   uHandOn;
uniform vec3  uHandPos;
uniform vec3  uHandVel;
uniform vec3  uHandEye;
uniform float uHandRadius;
uniform float uHandStrength;

layout(location = 0) out vec4 oColor;

void main() {
    ivec2 t = ivec2(gl_FragCoord.xy);
    int cx = t.x;
    int cy = t.y % GRID.y;
    int cz = t.y / GRID.y;

    float cm = texelFetch(uMass, t, 0).r;
    if (cm <= 0.0) { oColor = vec4(0.0); return; }

    vec3 v = texelFetch(uMom, t, 0).xyz / cm;
    v.y += uGravity * uDt;

    // Splash 方式の壁際2セル処理。X/Z は無条件ゼロ化、Y のみ方向限定
    // (床に向かう成分だけ止め、押し返す成分は生かす) — 薄い層だと床際の帯がほぼ全高を
    // 覆ってしまい、密度過多を解消する上向きの圧力速度まで潰して暴発するため。
    if (cx < 2 || cx > GRID.x - 3) v.x = 0.0;
    if (cy < 2           && v.y < 0.0) v.y = 0.0;
    if (cy > GRID.y - 3  && v.y > 0.0) v.y = 0.0;
    if (cz < 2 || cz > GRID.z - 3) v.z = 0.0;

    // 床際は XZ に摩擦減衰をかけ、床に沿って滑って角に集まるのを抑える。
    if (cy < 2) { v.x *= 0.85; v.z *= 0.9; }

    // CFL クランプ: 1サブステップの移動量を最大 1.5 セルに制限する全セル共通の安全網。
    // 1.0 だと単位系換算で約 2.09 m/s になり、水深 0.28m からの自由落下だけで
    // 日常的に頭打ちになるため 1.5 (約 3.1 m/s)。
    const float CFL_MAX_CELLS = 1.5;
    v = clamp(v * uDt, -CFL_MAX_CELLS, CFL_MAX_CELLS) / uDt;

    // マウス押しは CFL クランプの後 (WebGPU 版のパス順 UPDATE_GRID → APPLY_HAND と同じ)。
    if (uHandOn != 0) {
        vec3 cell = vec3(float(cx), float(cy), float(cz)) + 0.5;
        vec3 rayDir = normalize(uHandPos - uHandEye);
        vec3 toCell = cell - uHandPos;
        float dist = length(toCell - dot(toCell, rayDir) * rayDir);
        if (dist < uHandRadius) {
            float s = 1.0 - dist / uHandRadius;
            v += uHandVel * (s * s * uHandStrength);
        }
    }

    oColor = vec4(v, cm);
}
`;

    // ── パス5: G2P + 移流 (TF) ── WGSL_G2P 相当。
    const G2P_VS = `#version 300 es
precision highp float;
${COMMON}
${declare(PROGRAM_ATTRIBS.g2p)}
${STATE_OUT}

uniform sampler2D uVel;
uniform float uDt;
uniform float uWallStiffness;
uniform float uLookaheadK;
uniform float uFloorFillFactor;
uniform float uHardMin;
uniform vec3  uHardMax;

// 床バネ帯の幅 (セル)。試行履歴 (縮めないこと):
//  - 食い込み時のみ: クッション喪失で全粒子が床面にパンケーキ化 → 底セルの密度スパイクを
//    EOS が弾き、薄い層が縦振動
//  - 1セル・2セル: いずれも実機で薄層の縦振動が残った
//  - 3セル (現行): 振動なし。代償として静止高さが水深依存になる (薄い水は床から ~1.4 セル
//    浮く) が、振動より害が小さいので許容
const float FLOOR_SPRING_BAND = 3.0;

// 予測位置を [lo,hi] にハードクランプし、クランプした軸の速度成分を 0 にする。
vec2 hardClampAxis(float p, float v, float lo, float hi) {
    if (p < lo) return vec2(lo, 0.0);
    if (p > hi) return vec2(hi, 0.0);
    return vec2(p, v);
}

void main() {
    copyAll();

    ivec3 cellI; vec3 f, wx, wy, wz;
    cellAndWeights(ipos, cellI, f, wx, wy, wz);
    ivec3 base = cellI - 1;

    vec3 gv = vec3(0.0);
    vec3 B0 = vec3(0.0), B1 = vec3(0.0), B2 = vec3(0.0);

    // PRESSURE_VS と同じ巻き上げ (STENCIL_IN_GRID により範囲チェック不要、テクセルの
    // Y座標とセル中心相対位置 d の Y/Z 成分をループ外へ)。重みの掛け算の順序
    // (wx*wy)*wz は P2G 側と一致させたまま変えないこと。
    int ty0 = base.z * GRID.y + base.y;
    for (int k = 0; k < 3; k++) {
        int tyk = ty0 + k * GRID.y;
        float dz = float(base.z + k) - ipos.z + 0.5;
        for (int j = 0; j < 3; j++) {
            int ty = tyk + j;
            float dy = float(base.y + j) - ipos.y + 0.5;
            for (int i = 0; i < 3; i++) {
                float w = wx[i] * wy[j] * wz[k];
                vec3 cv = texelFetch(uVel, ivec2(base.x + i, ty), 0).xyz;
                vec3 d  = vec3(float(base.x + i) - ipos.x + 0.5, dy, dz);
                vec3 wv = w * cv;
                gv += wv;
                B0 += wv.x * d;
                B1 += wv.y * d;
                B2 += wv.z * d;
            }
        }
    }

    // APIC の affine 行列。次の substep の PRESSURE_VS が使う。
    C0 = B0 * 4.0; C1 = B1 * 4.0; C2 = B2 * 4.0;

    vec3 np = ipos + uDt * gv;
    vec3 nv = gv;

    // 粒子固有の固定ジッタ (grid imprinting 対策)。位置にだけ毎 substep 微小加算し、
    // 速度には影響させない。0.0001 セルは物理的に無視できる大きさ。
    np += ijitter * 0.0001;

    // 床バネ (Splash 方式): 予測位置ベースの常時バネ。押し返しの強さだけ
    // uFloorFillFactor (ドメイン全体の水量比) でスケールし、水量が少ないときは弱める。
    float floorZoneY = uHardMin + FLOOR_SPRING_BAND;
    vec3 xn = np + nv * uDt * uLookaheadK;
    if (xn.y < floorZoneY) {
        float distBelow = floorZoneY - xn.y;
        float blend = clamp(distBelow / FLOOR_SPRING_BAND, 0.0, 1.0);
        nv.y += uWallStiffness * uFloorFillFactor * blend * distBelow;
    }

    // ハードクランプは最後の安全弁 (バネで押し戻してもなお壁を超える場合のみ発動)。
    vec2 rx = hardClampAxis(np.x, nv.x, uHardMin, uHardMax.x);
    vec2 ry = hardClampAxis(np.y, nv.y, uHardMin, uHardMax.y);
    vec2 rz = hardClampAxis(np.z, nv.z, uHardMin, uHardMax.z);
    pos = vec3(rx.x, ry.x, rz.x);
    vel = vec3(rx.y, ry.y, rz.y);
}
`;

    return {
        FULLSCREEN_VS,
        P2G_MASS_VS, P2G_MASS_FS, P2G_MOM_VS, P2G_MOM_FS,
        PRESSURE_VS,
        GRID_UPDATE_FS,
        G2P_VS,
    };
}
