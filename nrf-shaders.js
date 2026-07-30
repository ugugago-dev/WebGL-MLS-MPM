// NRF (Narrow-Range Filter) 流体サーフェスレンダリングの GLSL ES 3.00 ソース群。
// WebGPU版 (../fluid-renderer.js) のPass1/T1/NRF/ThB/shadeを移植したもの。
// 壁のデバッグ描画・spray/foam(diffuse粒子)は Sim_WebGL では未実装/対象外なので、
// それらに依存するパス(Pass W、diffuse合成)は含まない。反射・屈折の式・定数は
// WebGPU版と厳密に同じ (固定skyColor反射+単色背景屈折、詳細は下記 SHADE_FS のコメント参照)。
//
// 生成は buildNRFShaders() 一箇所。restDensity・floor系の定数は sim 側 (FluidGL) の
// 値と食い違わないよう引数で受け取り、シェーダ文字列へ焼き込む
// (WebGPU版が REST_DENSITY 等をテンプレートリテラルで焼き込むのと同じ流儀)。

import { FULLSCREEN_VS } from './gl-shaders.js';
import { urlNum } from './config.js';

// GLSL ES 300 は int↔float の暗黙変換を一切許さないため、JSの数値をテンプレート
// リテラルでGLSLソースへ焼き込むときは整数値でも必ず小数点を付けて float リテラル
// にする (`3` ではなく `3.0`) — さもないと `float / 3` のような式がコンパイルエラーになる。
function glFloat(n) { return Number.isInteger(n) ? `${n}.0` : `${n}`; }

// 粒子半径 (グリッド単位)。旧 particle-render.js の radius と同値。
export const PARTICLE_RADIUS = 0.45;
const NRF_SIGMA = 1.5 * PARTICLE_RADIUS;
const NRF_DELTA = 10.0 * PARTICLE_RADIUS;
const NRF_MU    = 1.0 * PARTICLE_RADIUS;
const BG_DEPTH  = -1.0;   // 背景センチネル (WebGPU版と同値)

// NRF H+V の反復回数。WebGPU版と同じ既定値2、?nrf= で診断用に上書き可能
// (WebGPU版 fluid-renderer.js の同名ノブと同じ流儀、config.js urlNum参照)。
export const NRF_ITERATIONS = urlNum('nrf', 2);

// オフスクリーンレンダリング解像度スケール。Pass1〜shadeまでをキャンバス解像度の
// この倍率で描き、最後のblitパスがバイリニアで実解像度へアップスケール (+FXAA)する。
// WebGPU版は FLUID_RES_SCALE (流体チェーン) と RENDER_SCALE (最終シーン) を独立に
// 持つが、Sim_WebGLはshadeがdepth/thicknessをtexelFetchで直接読む構造 (アップサンプル
// 無し) なので、両方を1つの解像度に統一したほうがシンプル — shadeまでの全パスが
// 同じ解像度なら、shade内で追加のバイリニアアップサンプルを実装する必要が無い。
// ?rs= で診断用に上書き可能 (WebGPU版の同名ノブと同じ流儀)。
export const RENDER_SCALE = urlNum('rs', 0.5);

const THICK_SMOOTH_SIGMA  = 3.0;
const THICK_SMOOTH_RADIUS = Math.ceil(2 * THICK_SMOOTH_SIGMA);

// FXAA (最終blitパス)。WebGPU版 fluid-renderer.js のFXAA定数と同値
// (EDGE_THRESHOLD/EDGE_THRESHOLD_MIN/FXAA_SPAN_MAX/FXAA_REDUCE_MUL/FXAA_REDUCE_MIN)。
const EDGE_THRESHOLD     = 1.0 / 8.0;
const EDGE_THRESHOLD_MIN = 1.0 / 16.0;
const FXAA_SPAN_MAX      = 8.0;
const FXAA_REDUCE_MUL    = 1.0 / 8.0;
const FXAA_REDUCE_MIN    = 1.0 / 128.0;

const THICKNESS_ABSORPTION = [0.05, 0.02, 0.005];   // per world-unit RGB absorption
const STRETCH_SENSITIVITY  = 0.15;
const STRETCH_MAX          = 1.5;
// 床際ストレッチフェード。STRETCH_FLOOR_FADE は gl-shaders.js G2P_VS の
// FLOOR_SPRING_BAND (=3.0) と一致させること (床バネ帯の幅と同じ想定)。
const STRETCH_FLOOR_FADE = 3.0;

// Pass1のクリア色 (背景センチネル) がNRFフィルタの背景判定と食い違わないよう、
// nrf-renderer.js から参照できるようにこれだけexportする。
export { BG_DEPTH };

// pos/vel から camera-facing なストレッチ済みビルボードを組む頂点シェーダ本体
// (Pass1深度・厚み蓄積で共有)。WebGPU版 particleShader の vs() を GLSL に移植したもの —
// gl_VertexID による頂点生成ではなく、3頂点分のコーナーを静的VBO (aCorner, divisor=0)
// として渡す。WebGL2 の「インスタンス描画には divisor=0 の属性が最低1つ必要」という
// 制約 (CLAUDE.md 踏んだ落とし穴参照) を、P2G散布の ineighbor と同じ要領で満たす。
function billboardVS(hardMin) {
    return `#version 300 es
precision highp float;

layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aVel;
layout(location = 2) in float aDensity;
layout(location = 3) in vec2  aCorner;

uniform mat4  uViewProj;
uniform mat4  uView;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uHalfSize;
uniform float uRestDensity;

out vec2 vUV;
out vec3 vCenterView;
out float vSize;
out vec2 vEx;
out vec2 vEy;

// G2P の床バネが常時上向き速度を注入するため、床付近の粒子は縦にストレッチして
// 見える。ストレッチ用速度のY成分だけ床から STRETCH_FLOOR_MIN(=HARD_MIN)〜
// +STRETCH_FLOOR_FADE の帯でフェードする (WebGPU版 WGSL_FLOOR_STRETCH_FADE と同じ)。
vec3 floorFadedVel(vec3 vel, float worldY) {
    vec3 v = vel;
    v.y *= clamp((worldY - ${glFloat(hardMin)}) / ${glFloat(STRETCH_FLOOR_FADE)}, 0.0, 1.0);
    return v;
}

void main() {
    float densityNorm = uRestDensity > 0.0 ? clamp(aDensity / uRestDensity, 0.0, 1.0) : 1.0;
    float sizeScale    = mix(0.6, 1.0, densityNorm);
    float effSize      = uHalfSize * sizeScale;

    vec2 vv = (uView * vec4(floorFadedVel(aVel, aPos.y), 0.0)).xy;
    float vl = length(vv);
    vec2 ex = vl > 1e-4 ? vv / vl : vec2(1.0, 0.0);
    vec2 ey = vec2(-ex.y, ex.x);

    // 面積を保ったままストレッチ (回転前のローカル座標でスケール)。
    float stretch = 1.0 + clamp(vl / effSize * ${glFloat(STRETCH_SENSITIVITY)}, 0.0, ${glFloat(STRETCH_MAX)});
    vec2 lc = aCorner;
    lc.x *= stretch; lc.y /= stretch;
    lc = ex * lc.x + ey * lc.y;

    vec3 world = aPos + (lc.x * uCamRight + lc.y * uCamUp) * effSize;
    gl_Position  = uViewProj * vec4(world, 1.0);
    vUV          = aCorner;
    vCenterView  = (uView * vec4(aPos, 1.0)).xyz;
    vSize        = effSize;
    vEx = ex; vEy = ey;
}
`;
}

// Pass1: 球インポスター → 線形深度 (R32F) + 実深度テスト (粒子同士のZオーダー)。
// 既存 particle-render.js の RENDER_FS と同じ「gl_FragDepthを球表面へ押し込む」技法。
const DEPTH_FS = `#version 300 es
precision highp float;
in vec2 vUV;
in vec3 vCenterView;
in float vSize;
in vec2 vEx;
in vec2 vEy;
uniform mat4 uProj;
layout(location = 0) out float oDepth;
void main() {
    float dd = dot(vUV, vUV);
    if (dd > 1.0) discard;
    float z = sqrt(1.0 - dd);
    vec2 xy = vEx * vUV.x + vEy * vUV.y;
    vec3 vp = vCenterView + vec3(xy * vSize, z * vSize);
    vec4 cl = uProj * vec4(vp, 1.0);
    gl_FragDepth = (cl.z / cl.w) * 0.5 + 0.5;
    oDepth = -vp.z;   // 正の線形視点深度
}
`;

// Pass T1: 厚み蓄積 (R16F、加算ブレンド、深度テストなし)。
const THICK_FS = `#version 300 es
precision highp float;
in vec2 vUV;
in float vSize;
layout(location = 0) out float oThick;
void main() {
    float dd = dot(vUV, vUV);
    if (dd > 1.0) discard;
    oThick = 2.0 * sqrt(1.0 - dd) * vSize;
}
`;

// NRF 1Dフィルタ (H/V)。深度のみで重みを決めるbilateralカーネルを、厚みにも同じ重みで
// 適用してMRT同時出力する (WebGPU版 mkNRF1DShader と同じアルゴリズム)。
// horiz: true=水平パス(投影行列のprojElem=proj[0][0]・幅を使う)、false=垂直パス。
function nrfFilterFS(horiz) {
    const projElem  = horiz ? 'uProj[0][0]' : 'uProj[1][1]';
    const screenDim = horiz ? 'float(sz.x)' : 'float(sz.y)';
    const stepDir   = horiz ? 'ivec2(i, 0)' : 'ivec2(0, i)';
    return `#version 300 es
precision highp float;
uniform mat4      uProj;
uniform sampler2D uDepthTex;
uniform sampler2D uThickTex;
layout(location = 0) out float oDepth;
layout(location = 1) out float oThick;

// sigma/delta/mu は実行中に変わらないので (URLノブ無しの固定値)、uniform化せず
// シェーダ生成時に定数として焼き込む。
const float SIGMA = ${glFloat(NRF_SIGMA)};
const float DELTA = ${glFloat(NRF_DELTA)};
const float MU    = ${glFloat(NRF_MU)};

float gaussian(float dist, float isigma2) { return exp(-0.5 * dist * dist * isigma2); }

void main() {
    ivec2 sz    = textureSize(uDepthTex, 0);
    ivec2 coord = ivec2(gl_FragCoord.xy);
    float depth = texelFetch(uDepthTex, coord, 0).r;
    if (depth < 0.0) { oDepth = ${glFloat(BG_DEPTH)}; oThick = 0.0; return; }

    float thickC = texelFetch(uThickTex, coord, 0).r;

    int rad = min(int(ceil(SIGMA * ${projElem} * ${screenDim} / (2.0 * depth))), 100);
    if (rad == 0) { oDepth = depth; oThick = thickC; return; }

    float sigma2  = float(rad) * float(rad) / 9.0;
    float isigma2 = 1.0 / sigma2;

    float dLow  = DELTA;
    float dHigh = DELTA;

    float sum  = depth;
    float tsum = thickC;
    float wsum = 1.0;

    for (int i = 1; i <= rad; i++) {
        ivec2 ncJ = clamp(coord - ${stepDir}, ivec2(0), sz - ivec2(1));
        ivec2 ncK = clamp(coord + ${stepDir}, ivec2(0), sz - ivec2(1));
        float zj = texelFetch(uDepthTex, ncJ, 0).r;
        float zk = texelFetch(uDepthTex, ncK, 0).r;

        bool jBg = zj < 0.0;
        bool kBg = zk < 0.0;

        bool jOk = !jBg && zj > depth - dLow && zj < depth + dHigh;
        bool kOk = !kBg && zk > depth - dLow && zk < depth + dHigh;
        if (jOk) { dLow = max(dLow, depth - zj + DELTA); dHigh = max(dHigh, zj - depth + DELTA); }
        if (kOk) { dLow = max(dLow, depth - zk + DELTA); dHigh = max(dHigh, zk - depth + DELTA); }

        bool outlier = (!jBg && zj > depth + dHigh) || (!kBg && zk > depth + dHigh);
        float w = outlier ? 0.0 : gaussian(float(i), isigma2);

        float fj = zj < depth - dLow ? depth - MU : zj;
        float fk = zk < depth - dLow ? depth - MU : zk;

        bool pairBg = jBg || kBg;
        float wj = pairBg ? 0.0 : w;
        float wk = pairBg ? 0.0 : w;
        sum  += fj * wj + fk * wk;
        wsum += wj + wk;

        float tj = texelFetch(uThickTex, ncJ, 0).r;
        float tk = texelFetch(uThickTex, ncK, 0).r;
        tsum += tj * wj + tk * wk;
    }

    oDepth = sum / wsum;
    oThick = tsum / wsum;
}
`;
}

// 厚み専用ガウシアンブラー (H/V、固定半径)。NRFのbilateral重みとは独立に、
// グリッド周波数の厚みムラを均す (WebGPU版 mkThickSmoothShader と同じ)。
function thickSmoothFS(horiz) {
    const stepDir = horiz ? 'ivec2(i, 0)' : 'ivec2(0, i)';
    return `#version 300 es
precision highp float;
uniform sampler2D uSrcTex;
layout(location = 0) out float oThick;
void main() {
    ivec2 sz    = textureSize(uSrcTex, 0);
    ivec2 coord = ivec2(gl_FragCoord.xy);
    float sum  = texelFetch(uSrcTex, coord, 0).r;
    float wsum = 1.0;
    float isigma2 = 1.0 / (${glFloat(THICK_SMOOTH_SIGMA)} * ${glFloat(THICK_SMOOTH_SIGMA)});
    for (int i = 1; i <= ${THICK_SMOOTH_RADIUS}; i++) {
        ivec2 ncJ = clamp(coord - ${stepDir}, ivec2(0), sz - ivec2(1));
        ivec2 ncK = clamp(coord + ${stepDir}, ivec2(0), sz - ivec2(1));
        float w = exp(-0.5 * float(i) * float(i) * isigma2);
        sum  += (texelFetch(uSrcTex, ncJ, 0).r + texelFetch(uSrcTex, ncK, 0).r) * w;
        wsum += 2.0 * w;
    }
    oThick = sum / wsum;
}
`;
}

// shade: 深度→法線再構成 + Schlick Fresnel + Beer-Lambert。WebGPU版 (../fluid-renderer.js
// shade パス) と大枠は同じだが、背景/反射/屈折は2026-07-30にsuburban_garden_4k.exrから
// 変換したequirectangularパノラマ(textures/suburban_garden_equirect.png)を使うよう
// WebGPU版から分岐した — キューブマップやスカイボックスジオメトリは使わず、方向ベクトル
// →equirect UV変換 (equirectUV()) を背景・反射・屈折の3箇所で使い回すだけ。
// Sim_WebGLは壁セル/床ジオメトリが恒久的に対象外なので、水が無いピクセルはカメラ視線
// 方向をそのまま環境マップへ投げて表示する (画面上=空、画面下=地面が写るだけで、
// 実際の地面ジオメトリとの整合は取れない — 見た目のための近似)。
//
// ネイティブ解像度のみ (FLUID_RES_SCALE無し) なので、depth/thicknessはtexelFetchで
// そのまま読む (バイリニアアップサンプル不要)。9×9インラインクリーンアップ
// (WebGPU版) は省略 — NRF+厚みブラーで実機目視して十分ならこのままでよい。
//
// gl_FragCoord は左下原点・Y上向き (WebGPUは左上原点・Y下向き) なので、近傍の
// 上下(dU/dD)取得方向とNDC Yの符号がWGSL版と逆になる点に注意 (詳細はコード内コメント)。
const SHADE_FS = `#version 300 es
precision highp float;
uniform sampler2D uDepthTex;
uniform sampler2D uThickTex;
uniform sampler2D uEnvTex;
uniform vec2      uScreenRes;
uniform float     uTanHalfFovY;
uniform float     uAspect;
uniform vec3      uCamRight;
uniform vec3      uCamUp;
uniform vec3      uCamForward;
layout(location = 0) out vec4 oColor;

const float PI = 3.14159265359;

// equirectangular環境マップの方向→UV変換。gl-utils.js loadEquirectTexture()が
// UNPACK_FLIP_Y_WEBGLなし(既定false)でアップロードしているため、v=0がパノラマの
// 先頭行=上端(+Y、空)、v=1が下端(-Y、地面)になる。
vec2 equirectUV(vec3 dir) {
    float u = 0.5 + atan(dir.z, dir.x) / (2.0 * PI);
    float v = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    return vec2(u, v);
}

// view空間の方向ベクトルをworld空間へ回転する。uCamRight/uCamUp/uCamForwardは
// nrf-renderer.js _setBillboardCamera() がmath.js cameraVectors()の出力(いずれも
// 正規直交)をそのままアップロードしたもの — フラグメントごとにcross()で作り直さない。
vec3 viewToWorldDir(vec3 v) {
    return normalize(uCamRight * v.x + uCamUp * v.y - uCamForward * v.z);
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    float rawDepth = texelFetch(uDepthTex, coord, 0).r;

    float ndcX = 2.0 * gl_FragCoord.x / uScreenRes.x - 1.0;
    float ndcY = 2.0 * gl_FragCoord.y / uScreenRes.y - 1.0;
    // depth=1のview空間レイ方向。water分岐の pos は depth 倍しただけ (pos = viewRay * depth)。
    vec3 viewRay = vec3(ndcX * uAspect * uTanHalfFovY, ndcY * uTanHalfFovY, -1.0);

    if (rawDepth < 0.0) {
        // 水が無いピクセル: カメラ視線方向をそのまま環境マップへ投げて背景として表示。
        oColor = vec4(texture(uEnvTex, equirectUV(viewToWorldDir(viewRay))).rgb, 1.0);
        return;
    }
    float depth = rawDepth;

    ivec2 sz = textureSize(uDepthTex, 0);
    float wpX = 2.0 * depth * uTanHalfFovY * uAspect / float(sz.x);
    float wpY = 2.0 * depth * uTanHalfFovY / float(sz.y);
    const float GRAD_SCALE = 50.0;

    // gl_FragCoord は Y が上向きなので、dU (視覚的に上) は +Y 方向
    // (WGSL版は Y が下向きで -texel.y が上だった — 符号が逆)。
    float dR = texelFetch(uDepthTex, clamp(coord + ivec2(1, 0), ivec2(0), sz - 1), 0).r;
    float dL = texelFetch(uDepthTex, clamp(coord - ivec2(1, 0), ivec2(0), sz - 1), 0).r;
    float dU = texelFetch(uDepthTex, clamp(coord + ivec2(0, 1), ivec2(0), sz - 1), 0).r;
    float dD = texelFetch(uDepthTex, clamp(coord - ivec2(0, 1), ivec2(0), sz - 1), 0).r;

    bool validR = dR > 0.0 && abs(dR - depth) < wpX * GRAD_SCALE;
    bool validL = dL > 0.0 && abs(dL - depth) < wpX * GRAD_SCALE;
    bool validU = dU > 0.0 && abs(dU - depth) < wpY * GRAD_SCALE;
    bool validD = dD > 0.0 && abs(dD - depth) < wpY * GRAD_SCALE;

    float dzdx = validR && validL ? (dR - dL) * 0.5 : (validR ? dR - depth : (validL ? depth - dL : 0.0));
    float dzdy = validU && validD ? (dU - dD) * 0.5 : (validU ? dU - depth : (validD ? depth - dD : 0.0));

    vec3 nRaw = vec3(dzdx * wpY, dzdy * wpX, wpX * wpY);
    float nLen = length(nRaw);
    vec3 n = nLen > 1e-8 ? nRaw / nLen : vec3(0.0, 0.0, 1.0);

    // viewRayは関数冒頭で計算済み (depth=1のレイ方向)。実位置はdepth倍するだけ。
    vec3 pos = viewRay * depth;

    vec3 viewDir = normalize(-pos);
    float NdotV  = max(dot(n, viewDir), 0.0);
    float fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);

    // 反射・屈折は同じequirect環境マップを reflect()/refract() の方向でサンプルする
    // (キューブマップ・スカイボックスジオメトリ無し、方向→UV変換1つを使い回すだけ)。
    vec3 incident = -viewDir; // カメラ→サーフェスの入射方向 (view空間)
    vec3 reflectedColor = texture(uEnvTex, equirectUV(viewToWorldDir(reflect(incident, n)))).rgb;

    // 空気(n=1.0)→水(n=1.333)へ入る屈折。eta<1なのでTIRは起こらず常に有効な方向が返る
    // (GLSL refract()の仕様上、k<0のケースはeta>1=媒質から出る側でしか発生しない)。
    const float IOR_AIR_OVER_WATER = 1.0 / 1.333;
    vec3 refractedColor = texture(uEnvTex, equirectUV(viewToWorldDir(refract(incident, n, IOR_AIR_OVER_WATER)))).rgb;

    float thickness  = texelFetch(uThickTex, coord, 0).r;
    vec3 absorption  = vec3(${THICKNESS_ABSORPTION.map(glFloat).join(', ')});
    vec3 transmittance = exp(-absorption * thickness);
    vec3 deepWater    = vec3(0.04, 0.25, 0.60);
    vec3 shallowWater = vec3(0.45, 0.72, 0.90);
    vec3 bodyColor    = mix(shallowWater, deepWater, 1.0 - transmittance.g);
    vec3 scattered    = bodyColor * (1.0 - transmittance.g);

    vec3 refracted  = refractedColor * transmittance;
    vec3 transmitted = refracted + scattered;
    oColor = vec4(mix(transmitted, reflectedColor, fresnel), 1.0);
}
`;

// 最終blit: shadeの出力 (sceneColorTex、RENDER_SCALE解像度) をバイリニアで実解像度へ
// アップスケールしつつ、FXAA (Timothy Lottes版FXAA3-console、WebGPU版 fluid-renderer.js
// と同じ11タップ・同じ定数) をかけてキャンバスへ書く。uv は実キャンバス解像度
// (uCanvasRes、WebGPU版の blitUniBuf.fullRes に相当) で正規化し、FXAAの近傍オフセット
// (inv) は sceneColorTex 自身のテクセル単位 (textureSize) を使う — 解像度が違う2つの
// 基準を混同しないこと。
const BLIT_FS = `#version 300 es
precision highp float;
uniform sampler2D uSceneTex;
uniform vec2      uCanvasRes;
layout(location = 0) out vec4 oColor;

void main() {
    ivec2 sz = textureSize(uSceneTex, 0);
    vec2 inv = 1.0 / vec2(sz);
    vec2 uv  = gl_FragCoord.xy / uCanvasRes;

    vec3 rgbNW = texture(uSceneTex, uv + vec2(-1.0, -1.0) * inv).rgb;
    vec3 rgbNE = texture(uSceneTex, uv + vec2( 1.0, -1.0) * inv).rgb;
    vec3 rgbSW = texture(uSceneTex, uv + vec2(-1.0,  1.0) * inv).rgb;
    vec3 rgbSE = texture(uSceneTex, uv + vec2( 1.0,  1.0) * inv).rgb;
    vec3 rgbM  = texture(uSceneTex, uv).rgb;

    const vec3 LUMA = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, LUMA);
    float lumaNE = dot(rgbNE, LUMA);
    float lumaSW = dot(rgbSW, LUMA);
    float lumaSE = dot(rgbSE, LUMA);
    float lumaM  = dot(rgbM,  LUMA);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    // 低コントラストな箇所はエッジ処理をスキップしてそのまま返す。
    float range = lumaMax - lumaMin;
    if (range < max(${glFloat(EDGE_THRESHOLD_MIN)}, lumaMax * ${glFloat(EDGE_THRESHOLD)})) {
        oColor = vec4(rgbM, 1.0);
        return;
    }

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * ${glFloat(FXAA_REDUCE_MUL)}), ${glFloat(FXAA_REDUCE_MIN)});
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-${glFloat(FXAA_SPAN_MAX)}), vec2(${glFloat(FXAA_SPAN_MAX)})) * inv;

    vec3 rgbA = 0.5 * (
        texture(uSceneTex, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
        texture(uSceneTex, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(uSceneTex, uv + dir * -0.5).rgb +
        texture(uSceneTex, uv + dir *  0.5).rgb);

    float lumaB = dot(rgbB, LUMA);
    oColor = vec4((lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB, 1.0);
}
`;

export function buildNRFShaders({ hardMin }) {
    return {
        FULLSCREEN_VS,
        BILLBOARD_VS: billboardVS(hardMin),
        DEPTH_FS,
        THICK_FS,
        NRF_H_FS: nrfFilterFS(true),
        NRF_V_FS: nrfFilterFS(false),
        THICK_SMOOTH_H_FS: thickSmoothFS(true),
        THICK_SMOOTH_V_FS: thickSmoothFS(false),
        SHADE_FS,
        BLIT_FS,
    };
}
