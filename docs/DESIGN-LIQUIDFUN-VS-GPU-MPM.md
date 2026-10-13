# LiquidFun（CPU 前提）と powder-play（GPU 前提）の設計比較

作成日: 2026-10-12

本書は、powder-play の GPU MLS-MPM を設計するときの参考書である。
LiquidFun は CPU の逐次処理を前提に作られた Box2D 系の粒子ソルバーであり、
powder-play は WebGPU の compute shader で数万から百万粒子を扱う。
両者の差は「どの計算機を前提にしたか」から生じる。本書はその前提から、
実装上の性質と、性能を得るための手法の違いを整理する。

個々の論点の詳細（ゼリーの材料表現、表面、接触の物理）は
[材料表現の調査](./RESEARCH-MLS-MPM-MATERIALS.md)、
性能の計測手順は [計測・検討指針](./PERFORMANCE-MLS-MPM.md) を参照する。

## 読み方

- **確認済み**: 読んだソースまたは計測結果に直接根拠がある。
- **導出**: 確認した式やコードからの推論。実機の計測値ではない。
- **未確認**: 抜粋やビルド設定から確認できなかった事項。

LiquidFun は `google/liquidfun` のコミット
`7f20402173fd143a3988c921bc384459c6a858f2` の抜粋を読んだ。
抜粋は `b2ParticleSystem.cpp/.h`、`b2World.cpp`、`Test.cpp`、`Main.cpp` である。
`b2Settings.h` など定数定義のファイルは含まれていないため、定数の具体値は本書で扱わない。
powder-play の設計記述は current source と reference tests を根拠とする。
GPU 実装で未確認の項目は後半の stage status と limitation に区別して記す。
数値の根拠は
[GPU kernel README](../src/lib/mpm/kernels/README.md)、
[CPU reference README](../src/lib/mpm/reference/README.md)、各ソースである。

## 1. 前提の違い

| 観点 | LiquidFun | powder-play の GPU 版 |
|---|---|---|
| 想定する計算機 | 汎用 CPU（スカラー処理中心） | 大量の軽量スレッドを持つ GPU（WebGPU compute） |
| 性能の主因 | 分岐、キャッシュ、逐次の依存関係 | メモリ帯域、同期回数、dispatch 数、atomic の競合 |
| 並列化の単位 | 既定では逐次ループ（並列化は未確認） | workgroup（64 invocation）単位の data parallel |
| 粒子数の扱い | 必要に応じて容量を倍々に拡張（`CreateParticle`） | 初期化時に固定容量を確保し、拡張しない |
| 数値精度 | `float32`（`b2Vec2`）を基本とする | GPU 側は `f32`、CPU の oracle は倍精度 |
| 結果の再現性 | 逐次処理のため、同一入力で同じ順序になりやすい | atomic の挿入順が実行ごとに変わり、加算順序は非決定的 |
| CPU↔GPU の往復 | 不要（すべて CPU 上） | readback は高コストのため、最小限に抑える必要がある |

**導出**: CPU では、データを読み直すコストは小さく、分岐や逐次の依存関係を
そのまま書けることが利点になる。GPU では、同じ数値処理でも
メモリ読み書きと同期の回数が支配的になりやすい。そのため、
「計算を減らす」「読み書きを減らす」「同期を減らす」という判断の優先順位が変わる。

## 2. 実装上の性質の違い

### 2.1 接触の求め方

**確認済み（LiquidFun）**: `UpdateContacts` は毎回、`UpdateProxies` で
粒子を `computeTag` によるタグへ写し、`std::sort` でタグ順に並べ替える。
その後、近傍の粒子ペアを `AddContact` で `m_contactBuffer` に追加する。
`Solve()` の substep ループ（`particleIterations` 回）の先頭で毎回
この処理を行う。ソートはソース中で「profile のホットスポット」と注記されている。
SIMD 経路（`FindContacts_Simd` など）は `LIQUIDFUN_SIMD_NEON` が定義された場合のみ使われる。
既定で有効かどうかは、抜粋からは確認できない（未確認）。

**確認済み（powder-play）**: 格子節点を基準にした linked list を使う。
`buildBuckets` は粒子ごとに 3×3 の 9 組の (粒子, 節点) を作り、
`atomicExchange` で節点の先頭に連結する。ここでの atomic は連結リストの挿入だけに使う。
`p2g` は節点ごとに 1 invocation を割り当て、その節点の連結リストを走査して
通常の `f32` 演算で加算する。粒子の近傍探索（ソートや格子の全走査）はない。

**導出**: LiquidFun は「粒子どうしの近さ」を毎 substep 探す。powder-play は
「粒子が触れる格子節点」を毎 step 登録する。どちらも近傍の列挙だが、
GPU では節点ごとの gather にすると書き込み競合を避けられる。
この違いが、以降の節で述べる atomic と scatter の扱いに直結する。

### 2.2 力と速度の書き込み

**確認済み（LiquidFun）**: `SolvePressure` は接触リストを順に読み、
ペアごとに `m_velocityBuffer.data[a] -= f` と `m_velocityBuffer.data[b] += f` を書く。
同じ粒子が複数の接触に現れるため、ペア単位の scatter 書き込みになる。
CPU では逐次実行なので競合は起きない。

**確認済み（powder-play）**: 粒子の応力と affine 運動量は `prepareParticles` で
粒子ごとに 1 回だけ計算し、16 byte/粒子の `preparedAffines` に保存する。
P2G はこの値を読むだけで、浮動小数点の atomic は使わない。
WGSL の `atomic<T>` は `i32`/`u32` に限られるため、整数 atomic は
linked list の挿入にだけ使い、格子への加算は節点側の gather で行う。

**導出**: GPU で同じ scatter を書くと、同じ格子値への加算が並列に衝突する。
powder-play は「書き込みを gather に変える」ことで、この衝突を避けている。
代償として、粒子ごとの近傍の一覧（9 組 × 粒子数の `bucketNext`）の記憶領域が必要になる。

### 2.3 材料と格子

**確認済み（LiquidFun）**: 粒子の属性は SoA（Structure of Arrays）で持つ。
`m_flagsBuffer`、`m_positionBuffer`、`m_velocityBuffer`、`m_forceBuffer`、
`m_weightBuffer`、`m_accumulationBuffer`、`m_depthBuffer`、`m_colorBuffer`、
`m_groupBuffer`、`m_userDataBuffer` などが並ぶ。
各 solver は必要な配列だけを読み書きする。
弾性・バネは `b2ParticlePair`/`b2ParticleTriad` の別配列で保持する。
材料ごとの連続体構成則ではなく、フラグと solver の組み合わせで振る舞いを変える。

**確認済み（powder-play）**: 粒子は 80 byte の AoS（Array of Structures）で、
`position`、`velocity`、`c`、`f`、`plasticVolume`、`materialModel`、
`lambda`、`mu`、`density`、`volume`、`frictionCoefficient`、
`bondedTriangleVertex` を 1 レコードに詰める。Neo-Hookean 粒子の damage は
`frictionCoefficient` の slot を再利用し、powder の摩擦係数と材料 model によって
意味を分けるため、stride は増やさない。構造体サイズは `layout.ts` の
`PARTICLE_STRIDE_BYTES`（80 byte）と一致する。格子節点は 80 byte で、
水・油ごとの質量、運動量、速度、体積を持つ。WGSL の alignment（`vec2` は 8 byte、
`vec4` は 16 byte）から算出した値とも一致する。

**導出**: AoS は 1 粒子の全属性を 1 回の連続読み出しで取れるため、
粒子ごとに複数の属性を使う solver（P2G、G2P、commit）では
キャッシュ線と読み出し回数の点で有利になりやすい。一方、
ある属性だけを大量に読むパスでは無駄な帯域が増える。
powder-play は、粒子を 1 本の storage buffer にまとめる代わりに、
dispatch の数を増やして役割を分けている（§4.2）。

### 2.4 容量と動的な粒子数

**確認済み（LiquidFun）**: `CreateParticle` は容量が尽きると `m_count` の 2 倍を確保し、
`ReallocateBuffer` で旧配列を `memcpy` して移す。`DestroyParticle` 後は
`SolveZombie` が生存粒子を先頭へ詰め直す。詰め直しは逐次の `for` ループで、
フラグ、位置、速度、グループなどの配列を同じ添字で一緒に移す。

**確認済み（powder-play）**: 容量は初期化時に確保し、拡張しない。`config.capacity` は
`particleCapacity` として固定され、クローンは `cloneParticles` へ GPU 側でコピーする。
emitter append は GPU buffer の容量内だけを受け入れ、`accepted` / `refused` /
`dropped` を返す。dispatch は容量ではなく予約済み particle slot 数で行う。
water/oil outlet は GPU 上で removed sentinel を設定するが、slot の圧縮・再利用はしない。
画面上の有効粒子数と予約済み slot 数は別々に報告する。固定容量時のメモリ量は複数の
particle/grid/readback buffer を含み、particle buffer 1本の byte 数だけでは表せない。
現在の実機での peak memory は計測していないため、本書では総量を断定しない。

**導出**: 固定容量は GPU バッファの再確保と、その途中のデータ移動を避けられる。
その代わり、確保量そのものが GPU メモリの上限になる。
既存の性能文書は、この確保量を「速度の改善と別」に扱うよう注意している。
1 step あたりの仕事を減らしたことと、メモリを減らしたことは別の成果である。

## 3. 時間積分と同期

### 3.1 substep と batch

**確認済み（LiquidFun）**: `Solve()` は `particleIterations` 回の substep を回す。
各 substep で `UpdateContacts`、`UpdateBodyContacts`、`ComputeWeight`、
`ComputeDepth`、`SolvePressure`、`SolveElastic`、`SolveSpring`、
`LimitVelocity`、`SolveCollision` など多数の solver を逐次に呼ぶ。
コードのコメントには、`SolveElastic` と `SolveSpring` は現在速度を参照するため
できるだけ後で呼ぶべきだ、とある。solver 同士の順序が結果に影響する。

**確認済み（powder-play）**: `stepBatch` は 1 回の `writeBuffer` と 1 つの
`beginComputePass` の中で、複数 step の dispatch を順番に encode する。
各 step は `clearGrid → prepareParticles → buildBuckets → p2g → updateGrid → g2p → commitParticles`
の順で、粒子バッファを ping-pong して入出力を入れ替える。接触・bond・CFL・fracture の pass は、対応する機能があるときだけ同じ step に挿入される（格子場の `buildTriangleField` は `p2g` と `updateGrid` の間、三角形接触と bond の solver は `g2p` の後）。
性能文書の A/B では、8 step の通常 compute pass が 48 から 1 に減った
（[性能文書 §5](./PERFORMANCE-MLS-MPM.md)）。

**導出**: LiquidFun の substep は CPU 上の関数呼び出しなので、各 solver の間に
同期のコストはない。GPU では dispatch の境界に暗黙の同期（バリア）が入るため、
step 内の処理を 1 pass に束ねることは「メモリ帯域の往復」を減らすことにつながる。
ただし、dispatch の間にはデータの依存を正しく保つ必要がある。
powder-play はこの依存を、dispatch の順序と ping-pong で保っている。

### 3.2 CFL と数値安全性

**確認済み（LiquidFun）**: `LimitVelocity` は速度の大きさを
`GetCriticalVelocity`（= `particleDiameter / dt`）で制限する。
`SolvePressure` の圧力尺度 `GetCriticalPressure` は `density × (criticalVelocity)^2`。
この上限は粒子径と時間刻みから決まる単純なものである。

**確認済み（powder-play）**: 音速の上限（`waveSpeedBound`）と速度の上限から、
`calculateMpmCflMicrosteps` が 1 マクロ step を複数の microstep に分割する。
実行時は `validateCflParticles` が GPU 上で各粒子の速度と波速の条件を確認し、
違反した microstep を rollback する。最大速度は `measureParticleSpeedPartials`
と `reduceParticleSpeedPartials` が GPU で reduction し、速度 scalar・成功済み
microstep 数・数値 status の計 40 byte を readback する。CFL 違反や診断可能な
変形失敗なら、適応 retry が最後に commit した状態から残り時間をより細かい
CFL step で再試行する。バッチ予算内に安全な継続分を収められない場合は、
commit 済みの進捗と未処理時間を返し、Gate 4 の accumulator が後続バッチへ繰り越す。
時刻を飛ばさず、CFL 判定を緩めず、1 step も進められない状態は明示的に停止する
（[GPU kernel README](../src/lib/mpm/kernels/README.md)）。

**導出**: LiquidFun の制限は粒子径と時間刻みだけで決まる。
powder-play は材料の波速と実際の速度を見るため、材料パラメータを変えても安全側に働く。
その代わり、検査そのものが GPU の仕事（と readback）になる。
readback の回数と量は、性能に直接影響する。

## 4. 性能を得るための手法の違い

### 4.1 CPU 側の手法（LiquidFun）

- **SIMD 化の余地**: 接触探索と代理粒子（proxy）の更新に NEON 経路がある。
  コメントには、ソートの高速化が将来の最適化対象として残されている。
- **ソートによる局所性**: タグで並べ替え、近傍を連続メモリに並べる。
  CPU のキャッシュを使う典型的な手法である。
- **逐次性の利用**: `SolveZombie` のように、順序を保った詰め直しを逐次で行う。
  GPU では同じ処理に並列性が不足する。
- **分岐の少ない処理**: 粒子ごとに `flags` で分岐する部分はあるが、
  「フラグがある solver だけを呼ぶ」ことで、不要な処理を全体から外している
  （`m_allParticleFlags & b2_...` の判定）。

### 4.2 GPU 側の手法（powder-play）

| 手法 | 目的 | 根拠 |
|---|---|---|
| 応力・affine の事前計算（prepare pass） | P2G の重複評価（最大 9 回/粒子）を 1 回へ | 性能文書 §5 の A/B、kernel README |
| 整数 atomic と linked list による gather | 浮動小数点 atomic を避け、節点ごとに 1 invocation | kernel README「no floating-point atomics」 |
| 1 batch の単一 compute pass | pass の開閉を減らす | 性能文書 §5（48 → 1） |
| 粒子 ping-pong | 入出力の依存を dispatch 順で保つ | `stepBatch`、`solver.ts` |
| 固定容量の事前確保 | 再確保と移動を避ける | `layout.ts`、kernel README |
| 最大速度の GPU reduction と小さな readback | 全粒子の CPU readback を避ける | kernel README（36 byte 程度） |
| 壁際の速度の減衰（cushion）と halo | 境界条件を格子上で処理する。cushion の幅はドメインに応じて縮む（最大 8 cell） | `WALL_CUSHION_CELLS`、`shaders.ts` |
| 三角形 contact field の全走査 | 少数の三角形collider向けprototype。全node×bodyの `O(G·B)` 走査で、AABB rejectionは詳細判定だけを省く | `buildTriangleField`。候補絞り込み案と規模ゲートは[性能チューニングロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md) |
| 小石同士のSAT候補探索 | 実際の接触がない物体も全組を調べる `O(B²)` prototype。AABB rejectionはSAT判定だけを省く | `solveBondedTriangles`。広域候補生成は未実装 |
| 水・油の相別 grid fields | phase ID を保ち、各 node で mass / momentum / volume を別に gather する | `GridNode`、`p2g`、`updateGrid` |
| node-owned water/oil contact | 混在 node で法線方向の impulse を計算し、相 field に等量・逆向きに適用する | `solveLiquidGridContact` と controlled reference test |
| surface occupancy | render-only の水/油 field。色・輪郭を物理 contact や surface tension に流用しない | `gatherSurfaceOccupancy`、`occupancy.ts` |
| 固体 fracture topology | 初期 bond graph を CSR adjacency と bond state に pack し、GPU で damage / broken を更新する。追加 particle は bond graph に入れない | `layout.ts`、`updateFractureBonds`、`applyFractureParticleDamage` |
| source / outlet | deterministic emitter を両 ping-pong particle buffer に append し、容量を超える分を数えて報告する。water/oil のみ outlet で sentinel 化し、slot は再利用しない | `scenes/emitter.ts`、`appendParticles`、`removeParticlesInOutlet` |

**導出**: GPU の性能は、個々の計算の速さより、
「dispatch の数」「buffer の往復」「readback の回数」で決まりやすい。
上の表の多くは、この 3 つを減らすための手法である。

### 4.3 性能の計測で分かったこと

**計測済み（powder-play）**: 静止暖機を入れた同条件の A/B で、
GPU の 1 step の中央値は 0.08634 ms から 0.07373 ms へ約 14.6% 短縮した
（[性能文書 §5](./PERFORMANCE-MLS-MPM.md)）。
ただし、p95 は一部の試行で悪化しており、tail latency の改善は主張していない。
メモリは 16 MB 増えた。確保量と速度を同じ成果として扱わないことを、
この計測の教訓として残す。

**計測しない限り不明な点**: GPU の時間は、`timestamp-query` が使える環境でのみ正確に測れる。
性能文書は、queue 完了待ちを GPU 単体の実行時間として扱わないよう注意している。
LiquidFun 側の性能は、この計算機・ビルドでは測っていない。
したがって、両者の速度を直接比較する数値はない。

## 5. 今後の設計で使う指針

1. **前提の計算機を先に決める**。CPU の知見（逐次の分岐、ソートによる局所性）を
   そのまま GPU に持ち込むと、dispatch の数と readback で性能を落とす。
2. **粒子の書き込みは gather に寄せる**。scatter が必要な場合は、
   整数 atomic や節点ごとの一覧を使い、浮動小数点 atomic に頼らない。
3. **dispatch と同期の数を先に数える**。1 step ごとの pass と readback を
   表にし、減らせるものから手を付ける。
4. **確保量と速度を別の指標で見る**。メモリ増加を速度改善として報告しない。
5. **数値の安全策は、読み出しを最小にして GPU 上で行う**。
   最大速度の reduction のように、スカラー 1 個だけを CPU に返す形を基本にする。
6. **LiquidFun の solver を移植する場合は、物理の前提を確認する**。
   LiquidFun の弾性と圧力は粒子系の係数で動き、連続体の構成則ではない
   （[材料表現の調査](./RESEARCH-MLS-MPM-MATERIALS.md) の §0 を参照）。

## 6. 未確認・制約

- LiquidFun の NEON 経路が既定で有効かは、ビルド設定を見ていないため未確認。
- LiquidFun の定数定義（`b2_particleStride` など）は、抜粋に含まれていないため扱わない。
- 両者の速度を同じ機器で比較した数値はない。本書の性能値は powder-play の A/B のみ。
- GPU の正確な時間は、`timestamp-query` が使える環境で、専用の計測手順を使って測る。
性能文書の A/B は Chrome 154 / Apple Metal 3 / `timestamp-query` 対応の環境で取った。
別の GPU や別のブラウザで同じ値が出るとは限らない。
- 本書は設計の比較であり、実装の変更や物理の正しさを保証するものではない。

## 7. インタラクティブ 2D sandbox の scope と architecture boundary

このページの目標は、材料の追加・操作・削除と、制御された水/油 interface および
局所 fracture を試せる **1つの testable vertical slice** である。一般用途の流体・固体
エンジン、物理 fidelity の保証、任意 topology の編集機能を目標にしない。

| boundary | 所有する責務 | 所有しない責務 |
|---|---|---|
| `src/routes/labs/mpm/kernels/+page.svelte` | 2D gravity UI、DeviceOrientation permission、source の入力/停止、outlet 操作、状態と件数の報告 | particle state の CPU mirror、画面回転、材料物理の実装 |
| `scenes/gate4.ts` / `scenes/emitter.ts` | 決定論的な初期 particle batch、seed/shape/material/velocity/domain の検証 | GPU 実行状態、source が容量を超えたときの暗黙の成功 |
| `kernels/solver.ts` と WGSL | upload 後の authoritative particle state、CFL、相 field/contact、bond damage、append/outlet、描画 | CPU 側の粒子状態を毎 frame 同期すること |
| `reference/solver.ts`、`reference/occupancy.ts`、tests | 小さな deterministic case の oracle と保存則/境界の検査 | 14k 粒子以上の runtime solver、GPU 結果の fidelity を証明すること |
| render occupancy | water/oil surface を可視化するための phase occupancy | 物理 contact、界面 tension、wetting |

初期 scene/emitter の CPU particle 配列は upload 前の入力データである。upload 後は
GPU state が正本となる。GPU から CPU へ戻す値は CFL/status や outlet の件数などの
小さな readback に限り、全粒子の読み戻しを通常更新に使わない。source append は
accepted prefix だけを両 ping-pong buffer に書く。容量超過分は `dropped`、検証失敗は
`refused` として報告する。outlet は water/oil の GPU particle を removed sentinel にし、
有効粒子数と予約 slot 数を分けて扱う。slot の compaction/reuse は行わない。

## 8. 六段階 roadmap と acceptance gates

ステージ 1–4 がこの vertical slice の実装範囲である。ステージ 5–6 は将来設計と
acceptance gate のみであり、現時点で solver API、phase field、thermal state を追加しない。
各ステージを別の実装単位として進めるため、実行タスクと完了状況は
[ステージ1–4の実装ロードマップ](./ROADMAP-MLS-MPM-STAGES-1-4.md) と
[ステージ5–6の将来ロードマップ](./ROADMAP-MLS-MPM-STAGES-5-6.md) に分ける。
本書の表は全体の境界と受け入れ条件の要約であり、個別ロードマップでは実装済み部分と
未通過ゲートを区別する。

| stage | scope | gate |
|---|---|---|
| 1 · 傾き重力 | DeviceOrientation の beta/gamma を現在の screen axes に投影し、重力 acceleration の 2D vector として使う。iOS の `requestPermission()` は user gesture handler 内で呼ぶ。`enabled` / `denied` / `unsupported` を UI に示し、常時使える manual X/Y vector に fallback する。画面 rotation/lock は行わない。 | 水平・各方向の傾き・screen angle・不正 sample の deterministic tests。実ブラウザーでは許可、拒否、非対応/HTTP、センサー未着を操作し、manual vector が常に sim を動かせることを確認する。 |
| 2 · source / outlet | seed 付き circle/rectangle emitter から water/oil/Neo-Hookean solid particle を repeatable に追加する。material、shape、count、position、spacing、初速を調整し、1回追加/連続追加を開始・停止できる。outlet は画面下部で water/oil のみを除去する。 | 同じ source input/sequence が同じ粒子位置・初速を作る。範囲外/不正 particle は全件拒否し、容量時は accepted/refused/dropped を実数で表示する。GPU write 範囲が予約 buffer 内で、両 ping-pong state に対して out-of-bounds がないこと。outlet の GPU readback 件数、active/allocated/capacity の差、および非再利用 slot を test する。 |
| 3 · water/oil phase | phase/material identity を particle に保持し、water/oil の P2G mass/momentum/volume と node velocity を分離する。混在 node では phase field を owner とする法線 impulse を計算し、等量・逆向きに返す。surface occupancy は可視化専用のままにする。 | 小さな controlled interface case で水/油の identity と相ごとの mass が変化せず、normal contact 後に界面を横切らず、合計 momentum が許容誤差内で保存されること。current-area occupancy の phase 分離/総量テストも維持する。これは選択した grid・初期状態の test であり、任意流れの immiscibility、surface tension、wetting の証明ではない。 |
| 4 · connected solid fracture | Gate 4 に 12×8 node / 249 bond の connected Neo-Hookean body を置く。初期 bond topology を GPU に固定して tensile strain damage/broken state を更新し、localized damage を粒子 stress/render に反映する。破断 body は選択・力ベースで編集できるが、clone/rebond で broken graph を復活させない。 | deterministic reference tests で failure threshold、broken state の永続性、damage localization、particle count/mass/momentum、fragment の独立移動、grid bounds を確認する。GPU pass ordering、bond CSR/buffer packing、clone refusal を test する。threshold はこの resolution 用の探索値であり、material strength や解像度独立性を主張しない。 |
| 5 · gas / cavitation / bubbles（future design only） | 液体圧力からの cavitation、液相から気相への mass transfer、bubble motion/merge/split を扱う設計検討。まだ実装しない。 | 液相だけを消す実装は gate 不合格とする。liquid+gas の総 mass、transfer 時の momentum/energy accounting、非負 phase mass、pressure の boundedness、再現性、bubble lifecycle を GPU-controlled case で示してから実装を検討する。 |
| 6 · thermal / phase change（future design only） | enthalpy による温度・latent heat・phase fraction、温度/相依存 volume change と熱輸送を扱う設計検討。まだ実装しない。 | 閉じた系で total mass を保ち、境界 heat flux を含む total energy accounting、latent-heat plateau、加熱/冷却の phase fraction、density/volume response を test してから実装を検討する。 |

### 8.1 Stage 3: 非混合 interface の証拠と境界

water/oil は palette 色だけで区別しているわけではない。particle material ID、別々の
node mass/momentum/volume/velocity、node-owned normal contact を持つ。CPU reference の
controlled two-particle case は接近する water/oil の法線速度を抑え、相 identity、相ごとの
particle mass、全 momentum と位置順序を確認する。`occupancy.ts` は変形後の phase area
field が独立して集計できることを boundary halo 込みで検査し、GPU surface occupancy は
描画用にだけ使う。したがって、合格している主張は「この controlled discrete case の
interface/contact」であり、完全な流体分離、長時間安定性、任意解像度への一般化ではない。

### 8.2 Stage 4: fracture data と resolution limit

fracture bonds は初期 solid object の global particle index を参照する固定 graph であり、
GPU buffer に endpoint、rest length、failure strain、damage、broken を保持する。particle ごとの
CSR incidence が incident bond damage の最大値を particle へ伝える。破断 edge は topology から
消さず、`broken` を永続させる。append 粒子はこの初期 CSR 範囲外として明示的に扱い、新しい
bond topology を作らない。fracture bond を持つ object の clone は false を返す（bonded 三角形だけの object は容量内で clone できる）。

この vertical slice は 10 world-unit grid cell に対して 9 world-unit node spacing の2D discrete
body である。failure strain `0.12`、damage による stress fade、粒子色は研究・検証用の制御値で
あり、ひび割れ幅、fracture energy、stress intensity、連続体の破壊強度を測定した値ではない。
分解能・spacing・積分 step・material coefficients を変えた同等性は示していない。

### 8.3 実装範囲と検証状況

この変更では stage 1–4 の UI/runtime と deterministic/reference tests を追加した。
`npm test` は 20 files / 170 tests、`npm run check` は 0 errors / 0 warnings、
`npm run build` は成功した。dev server の SSR smoke check では対象 route と
gravity/source/outlet/fracture controls が返ることを確認した。

共有ブラウザーの WebGPU 対応環境では route の起動と初回CPU/GPU probeを確認した。
さらに水64粒子の1回追加を操作し、受入64・拒否0・容量超過0、活動粒子数14,696から
14,760への更新を確認した。これはsolverの実機実行とemitter appendのsmoke testであり、
全機能のbrowser/E2E受け入れではない。

2026-10-13に専用benchmark tabを用いたApple Metal 3 / Headless Chrome 154の実GPU計測を追加した。
現行の14,696粒子sceneでshader compile・dispatch・数値statusが通過し、batch profileを5回計測した。
GPU step medianの中央値は0.40109 ms、p95中央値は0.43035 ms（描画なし）。
別途45粒子・5材料・8 stepのCPU/GPU parityも最大絶対差1.3955e-5で通過した。
同じ第5回benchmarkの制御GPU fixtureでは、水・油2粒子/8 stepでCPU/GPU最大差3.8653e-5、
相別mass各0.25、最大運動量残差1.1921e-7、全step非交差を確認した。4粒子・3 bondの破断fixtureでは、
閾値下/上/無負荷strainを比較しbond index 1だけが破断、damage差は7.63e-6以下。
rest lengthの初期broken bondも2 step保持した。これらは小規模fixtureの確認であり、
現行scene全粒子のreadbackやscene-scaleの相別mass/momentum、界面貫入、破断pattern、
長時間形状保持を確認したものではない。
この性能値・fixture結果は後から作業treeに加わったadaptive CFL microstep/recovery変更前の
source snapshotに対する記録である。変更後sourceの検証結果ではないため、M2確認後に再計測する。
DeviceOrientation の実機許可・拒否、画面向きごとのsensor axes、iOS/Android実機動作、
outletのGPU除去、連続追加停止、固体emitterは未確認である。
fake GPU tests はcommand order・buffer bounds・容量報告を検査するが、すべてのGPU実行の
OOB safetyやinterface/fractureの数値挙動を証明しない。したがって、Stage 1–4は実装prototypeと
一部smoke testがあるが、実機acceptance gate全体を通過したとは扱わない。

Stage 5–6 はこの文書の設計と acceptance gate のみで、runtime solver/API/UI は
追加していない。GPU 実測値や物理 fidelity は主張しない。buffer memory の値は
layout-derived estimate、fracture threshold と resolution は §8.2 の検証範囲に限る。

## 9. Stage 5–6 用の将来 data contract（設計のみ）

以下は後から phase と熱力学を追加するときに既存 particle slot を暗黙に流用しないための
**設計契約案であり、runtime TypeScript interface / GPU buffer / shader ではない**。

```text
MaterialDefinition
  materialId, constitutiveModel, phaseId
  referenceDensity, mechanical/EOS parameters
  thermal parameters (future): heatCapacity, conductivity, latentHeat, transitionRange
  volume law / state schema version

ParticleState (future)
  mass, position, velocity, deformation, affineVelocity
  materialId, phaseId, damage/bond references
  specificEnthalpy, temperature/phaseFraction (derived or explicitly versioned)

NodePhaseState (future, per phase)
  mass, momentum, currentVolume, pressure/EOS state
  enthalpyMass and conservative heat-flux accumulator

PhaseTransfer (future, explicit event/flux)
  sourcePhase, destinationPhase, transferredMass
  transferredMomentum, transferredEnthalpy, stable transfer key
```

mass は phase transfer event で source から destination へ一度だけ移し、particle を
delete/duplicate して相変化を表さない。total mass は全 phase の particle/node mass と
transfer ledger で照合する。enthalpy は specific enthalpy×mass で積算し、latent heat を
含む `h(T, phaseFraction)` の単調な関係を明示する。温度による体積変化は density/volume law
から導き、mass を変更して体積変化を偽装しない。GPU の parallel flux は同じ face/node
で equal-and-opposite に計算し、mass/energy diagnostics は GPU reduction と小さな readback
を使う。今の 80-byte particle slot の `frictionCoefficient` や `damage` slot に thermal state
を上書きしてはならない。

### Stage 5 の solver choices（未決定）

- **弱圧縮性 gas EOS**: phase ごとの gas mass/volume を node gather し、pressure を EOS から
  求める explicit MLS-MPM。既存 GPU pass と揃う一方、sound speed/CFL と圧縮率の調整が必要。
- **低 Mach / incompressible gas**: node pressure Poisson solve を Jacobi または GPU 向けの
  preconditioned iterative solve で行う案。反復数、収束・境界条件、GPU dispatch cost の測定が必要。
- **cavitation transfer**: 閾値 pressure で liquid mass の一部を gas phase へ移す conservative
  nodal flux。nucleation hysteresis と min/max transfer limiter を持たせ、負 mass と発散を防ぐ。
- **bubble representation**: gas field を連続相として解く案と、GPU cell/bin に所属する bubble
  marker を使う案を比較する。merge/split は deterministic prefix-sum/sort または node-owned
  gather で行い、float atomic scatter を前提にしない。

圧力 solve、gas EOS、bubble marker を混在させる最小構成を決める前に、mass closure と CFL/
pressure stability の小規模 test を通す。gas field・cavitation API は未実装である。

### Stage 6 の solver choices（未決定）

- enthalpy を保存変数にして、temperature と phase fraction を `h(T, phaseFraction)` から
  求める。融解/蒸発の latent heat は enthalpy interval として扱い、温度だけを積分して
  相変化エネルギーを失う設計にしない。
- conduction は grid-face の equal-and-opposite heat flux を使う explicit stable substep
  と、GPU Jacobi/iterative implicit diffusion を比較する。どちらも境界熱流束を明示し、
  閉じた断熱ケースで energy を監査する。
- 固体/liquid/gas の density law を material/phase ごとに定義し、`volume = mass / density`
  に沿った膨張・収縮を反映する。相変化時に液体を削除して gas を新規生成するのではなく、
  mass・momentum・enthalpy を同一 transfer で移す。

これらは設計案にとどまり、stage 5/6 の solver code、data schema、UI を今回の実装に含めない。
