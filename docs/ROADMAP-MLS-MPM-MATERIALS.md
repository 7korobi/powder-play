# GPU MLS-MPM 材料表現の実装ロードマップ

作成日: 2026-10-12

状態: **設計・実装計画。以下の M0〜M7 は未着手**。
根拠・式・現行ソースのハッシュは
[４論点の調査資料](./RESEARCH-MLS-MPM-MATERIALS.md) に保存する。
本書は [既存の Gate 0〜5](./ROADMAP-MLS-MPM.md) を置き換えず、
Gate 1・3・4 の数値／材料機能を深め、Gate 5 に渡すためのサブロードマップである。

追記: M1 の粒子別応力precomputeに相当する性能変更は
[性能ロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md) で実装・検証した。
現行cacheは必要なアフィン運動量のみの16 byte/粒子で、batchを１compute passに集約する。
以下の32 byte cacheやpass構成は追加診断・材料拡張時の設計候補であり、
M0〜M7全体の物理ゲートが合格したことを意味しない。

## 1. 到達点と設計境界

最初の到達点は、GPU 上で「柔らかいゼリーの復元、自由表面、
２体の滑り・離脱」を独立に確認した後、水・油・ゼリー／粉体を少数場で共存させること。
最初から実材料の校正、濡れ、浸透、破断、自己接触、明示的気体まで含めない。

| 論点 | 最初の実装 | 後段の分岐 |
|---|---|---|
| 圧力・体積 | 構成則の計測、物理 step 全体の細分化、パラメータ校正 | 等容・体積分離、必要なら圧力 projection |
| 自由表面 | 現在面積の場、等値線描画、境界・閾値校正 | 曲率、表面張力、非混和界面 |
| 材料間接触 | 少数の格子速度場、接近時の法線拘束、滑り・離脱 | 材料対摩擦、多点接触、流体–固体荷重 |
| 変形・復元 | 現行 Neo-Hookean、回転・圧縮解放の基準試験 | 粘性、別構成則の比較 |

維持する契約:

- 粒子状態と物理計算の正本は GPU。CPU は入力、時間予算、少数 metadata、
  command encoding、試験用の小規模 reference を扱う。
- 毎 frame の粒子全量 readback、CPU 接触探索、CPU 三角形構築は導入しない。
- 整数 atomic の bucket 構築と節点 gather を出発点にする。
  float atomic、固定小数点への無検証の置換を前提にしない。
- 異常時の永続 status と step 全体の rollback を維持する。
  無効な `F`、容量不足、非収束を clamp や成功扱いで隠さない。
- 実験は `/labs/mpm/kernels` 周辺に隔離する。通常画面 `/` と既存 β は維持する。
- 新しい物理経路はラボの明示的な実験設定として選び、旧経路と A/B 比較できるようにする。
  旧モデルを無告知で置き換えない。

## 2. 現行実装からの差分

出発点の詳細は [調査資料 §0](./RESEARCH-MLS-MPM-MATERIALS.md#0-出発点-liquidfun-のゼリーと現行-mls-mpm)。
特に次を前提にする。

1. 粒子 stride は **80 byte**、格子は単一速度場、物理 step は６ compute pass。
2. P2G リスト内で構成則を再計算している。粒子ごとに最大９回である。
3. 1/240 秒固定の batch 送信はあるが、CFL に基づく全物理 step の細分化はない。
4. 液体の `J = 0.5〜2`、粉体の `J >= 0.5` は安定化制限であり、
   体積保存・非貫入・塑性圧密の代用にはしない。
5. 密度差加速度は G2P 後の近似補正。新しい圧力／接触との整合性を検証せず併用しない。
6. 20,000 粒子しか動かさなくても、現在は 1,000,000 粒子分を確保する。
   新機能より先に capacity と総メモリ予算を切り分ける。
7. 既存の３ probe の初回 CPU/GPU 比較だけでは、４論点の合格を宣言しない。

## 3. 推奨 GPU データ・pass 構成

### 3.1 粒子別の prepare cache

M1 の候補として、32 byte/粒子の scratch を用意する。

```text
PreparedParticle (暫定 32 byte)
  affineMomentum: vec4<f32>  // A = m*C - 4*dt*V0/h^2*tau
  mass: f32
  J: f32
  waveSpeedBound: f32
  flags: u32
```

１粒子１ invocation で応力・分解・質量を計算し、P2G は位置・重み・cache から gather。
Neo-Hookean では `tau = mu*(F*F^T-I) + lambda*log(J)*I` の等価式を使える。
Snow の硬化と粉体の分解もここへ移す。有限値検査とモデル別の異常原因を維持する。

`A` は `dt` と現在の `F/C` に依存するため、**毎 microstep 更新**する。
batch 内で古い cache を再利用しない。複製後も active count を使って更新する。
scratch は step 間で使い回せるが、次の P2G より前に GPU pass 境界で確定させる。

実時間の改善は未確認。液体だけの軽い EOS では cache 帯域が不利にもなり得る。
最終採否は M1 の測定で決める。

### 3.2 Contact field

`contactFieldId` を粒子の構成則・UI の `objectId` と別に定義する。
最初は２場、次に最大４場。水・油は相ごと、独立ゼリーは物体ごとに場を割り当てる。
場の descriptor に `phaseId`、材料対の接触係数、境界方針を持つ。

現在の粒子 layout は word 19 が padding なので、ここへ `u32 contactFieldId` を
正式に追加し、80 byte stride を維持する案を先に検証する。
「空いているから書く」のではなく、TS packing、compute/render WGSL、
CPU 型、readback、clone、テストを同じ契約へ変更する。
別 metadata buffer が必要なら `4 * capacity` byte と binding を追加予算に入れる。

暫定 channel node は 48 byte/場/節点を上限目安とする。
mass、momentum、velocity、現在面積、面積勾配・幾何補助量を保持し、
最終の WGSL alignment と buffer tests で確定する。
必要な量だけを残し、現在の全量 mixture 集計を無条件に並存させない。

P2G は１節点１ invocation で bucket list を**１回**走査して各場へ振り分ける。
節点×場ごとに同じリストを再走査する `9*N*M` 型の実装を避ける。
接触も１節点内の場を所有して計算し、他節点の速度を同時に読み書きしない。

### 3.3 数値 pass と表示 pass

M1 採用後の単一場:

```text
clearGrid
  -> prepareParticles
  -> buildBuckets
  -> p2gGather
  -> updateGrid
  -> g2p
  -> commitParticles
```

M4 の複数場:

```text
clearFields
  -> prepareParticles
  -> buildBuckets
  -> p2gGatherFields
  -> deriveInterfaceGeometry
  -> updateGridFields (重力・局所入力)
  -> solveNodeContact (全場を同一 invocation が所有)
  -> g2pFromOwnField
  -> commitParticles
```

`deriveInterfaceGeometry` が近傍を読む場合は、P2G 完了後の面積場を入力にする。
壁と接触の適用順を CPU/GPU で固定し、corner では両拘束を同時に満たすか検証する。
M5 の粘性・表面張力は対応する力・係数場が確定してから grid update に加える。
隣接速度を読む粘性 stencil は入出力を分離し、in-place の競合を作らない。

frame 終了時の表面描画は、必要なら現在の粒子状態から
`buildBuckets -> gatherCurrentArea -> renderSurface` を別に行う。
物理に使った G2P 前の格子を最終粒子位置の表面として無条件に流用しない。
表示場の平滑化はまず表示専用とし、物理の面積・接触場を変えない。

どの新 pass でも数値 status を共有する。
G2P で各粒子を部分的に確定せず、全 pass 後に commit/rollback を決める。

### 3.4 WebGPU limits と概算メモリ

現在の P2G は５ storage buffer を使う。prepare cache の追加で６になる可能性がある。
[layout の制約検査](../src/lib/mpm/kernels/layout.ts) の「５本」判定を流用せず、
entry point ごとの binding 数を数え直す。場の小さな設定表は uniform を第一候補にする。
buffer size、storage binding size、dispatch 上限、shader compile と allocation error scope を
初期化時に検証し、要求機能を使えない場合は理由を UI に返す。

`N = capacity`、`G = 34,119`、`M = 4` とする概算。MB は `10^6 byte`。

| 主な GPU buffer | 現行 | M1 | M4 候補 |
|---|---:|---:|---:|
| 粒子 A/B | `160*N` | 同左 | 80 byte stride を維持できれば同左 |
| object ID | `4*N` | 同左 | 同左 |
| bucket links | `36*N` | 同左 | 同左。場ごとには複製しない |
| bucket heads | `4*G` | 同左 | 同左 |
| grid | `32*G` | 同左 | `48*M*G` に置換 |
| prepare scratch | なし | `32*N` | `32*N` |

| capacity | 現行主要 buffer | M1 追加後 | M4 の４場案 |
|---|---:|---:|---:|
| 20,000 | 約 5.23 MB | 約 5.87 MB | 約 11.33 MB |
| 50,000 | 約 11.23 MB | 約 12.83 MB | 約 18.29 MB |
| 1,000,000 | 約 201.23 MB | 約 233.23 MB | 約 238.69 MB |

M4 の増分格子は約 5.46 MB。contact ID を別 buffer にするならさらに `4*N` byte。
４相の表示用 scalar 場２枚なら `8*M*G ≈ 1.09 MB` を追加する。
表は小さな uniform/status、texture、driver overhead、診断 reduction scratch を含まない。
全粒子 readback は別途最大約 84 MB。通常処理では確保しない。
device limits を満たしても総メモリが確保できる保証はない。

**提案する試験 profile**: mobile は capacity 50,000 以下・20,000 active を起点、
PC の増殖性能試験だけ 1,000,000 capacity を明示選択する。
mobile の総 GPU 予算は初期案 128 MiB とし、実機結果で調整する。
勝手に粒子を捨てたり CPU 物理へ切り替えたりせず、profile と機能を表示する。

## 4. 実装段階と判定ゲート

以下の数値閾値は**提案する PoC 合格条件**であり、確認済みの実績ではない。
試験 scene、単位、機種、設定を固定してから使用する。

<a id="m0"></a>
### M0 — 現状の再現・計測・メモリ予算

**前提**: なし。

- 調査資料のソース状態を基準に、現行 Gate 4 を改めて検証する。
  古い条件の 60 simulated seconds を今回の実績にしない。
- capacity 別の確保 byte を表示し、PC と mobile の profile を分ける。
- GPU reduction で材料別の質量、面積 `sum(V0*J)`、`J` の範囲、
  射影回数、最大速度、運動量などを集計する。数秒ごとの小さな summary を読む。
- 既存 28 byte の numerical status と、診断 summary を区別する。
  成功した microstep 数と submit 換算を混同しない。
- `timestamp-query` が使える場合は pass 時間を測る。
  非対応なら CPU encode と queue 完了待ちを別表示し、GPU 単体時間と呼ばない。
- 粒子間隔・基準面積・質量・密度の意味を固定する。
  現在の占有率約 0.309 を前提に、表面閾値や「密度」を校正する。

**合格条件**:

- small scene の CPU/reference と GPU 初回比較を全粒子で行い、差分の項目を記録する。
- 壁なし・外力なしの transfer の質量／運動量を確認する。
- 現行条件で数値異常・projection 頻度・メモリ・描画 FPS を記録できる。
- 通常動作で粒子全量 readback がない。計測追加のメモリを含めて容量上限を説明できる。

**担当面**: [layout](../src/lib/mpm/kernels/layout.ts)、
[GPU solver](../src/lib/mpm/kernels/solver.ts)、
[lab](../src/routes/labs/mpm/kernels/+page.svelte)、[scene](../src/lib/mpm/scenes/gate4.ts)。

<a id="m1"></a>
### M1 — 粒子別応力 precompute

**前提**: M0。後続物理機能の必須条件ではなく、測定して採否を決める最適化。

- §3.1 の prepare/cache を実装し、P2G 内の構成則評価を移す。
- Neo-Hookean の等価 `tau` 式を導入。CPU の旧式と代数対応をテストする。
- layout、両 ping-pong bind group、容量検査、dispose、clone 後の更新、
  persistent status と error detail を接続する。
- 単一液体、ゼリー、Snow／粉体、20,000 粒子の混合を A/B 測定する。
  dense node の linked-list 走査も別に測る。

**合格条件**:

- 全モデルの constitutive stress と１ step 状態が旧経路と許容誤差内。
  純回転・大せん断・小さい正の `J`・不正入力も含める。
- 応力／分解計算は各 active 粒子で１回。GPU 全体時間と追加メモリを報告する。
- 主要混合 scene で end-to-end step 時間が 5% を超えて悪化する場合、
  無条件採用せず baseline を維持する。timestamp のない環境ではこの判定を保留する。

**担当面**: [WGSL](../src/lib/mpm/kernels/shaders.ts)、
[GPU solver](../src/lib/mpm/kernels/solver.ts)、
[layout tests](../src/lib/mpm/kernels/layout.spec.ts)、
[CPU tests](../src/lib/mpm/reference/solver.spec.ts)。

<a id="m2"></a>
### M2 — 全物理 substep と圧力・復元の基準化

**前提**: M0。M1 が採用されれば cache を再利用する。

- 1/240 秒を UI の macrostep として維持する場合、内部で
  `k = ceil(macroDt / safeDt)`、`microDt = macroDt/k` に分割する。
  各 microstep で全数値 pass を実行し、macrostep と microstep を別表示する。
- 初期案は `safeDt <= 0.3*h/(c_bound+v_bound)`。
  現行液体の静止条件でも `k >= 2` が必要となる。
  既定 60 FPS・2×は概ね８ macrostep/frame なので、２分割なら16 microstep/frame。
- 最初は材料・想定変形・速度の保守的上限から一定 `k` を決め、
  prepare/grid/commit 前の GPU 検査で上限逸脱を検出する。
  逸脱時は rollback と通知を行い、上限や刻みを再設定する。
  直前 batch の非同期最大速度だけを次 batch の保証として使わない。
- Neo-Hookean の大変形、Snow の硬化を含む接線剛性 bound を CPU で検証する。
  `sqrt((lambda+2*mu)/rho0)` は微小ひずみの目安であり、全状態の bound としない。
- [scheduler](../src/lib/mpm/kernels/step-scheduler.ts) の仕事上限は microstep 数で制限する。
  超過時は `dt` を増やさず、実効倍率・破棄時間を表示する。
  揺動は実際のシミュレーション区間平均と整合させる。
- ゼリーは現行 Neo-Hookean を維持し、`mu` と体積応答を別々に校正する。
  等容・体積分離の別モデルは必要性が確認された場合だけ追加する。

**合格条件**:

- `F = R`、一様圧縮、単純せん断の解析応力と CPU/GPU が一致する。
- 小振幅ゼリーの振動周期は細かい `dt` の reference と 5% 以内、
  解放後の面積 drift は基準面積の 2% 以内を初期目標とする。
  減衰時間と拘束条件を固定し、形状は並進・回転を除いて比較する。
- `dt` を半分にしたときの圧力・体積・回転誤差の収束傾向を記録する。
- 60 simulated seconds の基準 scene で数値 rollback がなく、
  clamp 発生数を明示する。clamp に依存する試験は物理合格としない。

F-bar 型安定化は全文確認と locking の定量的検出後の別実験。
単に `J` を平滑化してこの段階の合格にしない。

<a id="m3"></a>
### M3 — 自由表面の幾何・表示

**前提**: M0。物理挙動の判定には M2。表示だけの試作は並行可能。

- 現在面積から材料別の占有場を gather し、境界／halo を含む校正を行う。
- 粒子表示、占有場、法線、surface 表示を切り替えられるようにする。
- 物理用と表示平滑化用を分け、最終粒子位置と同じ時刻の表示場を作る。
- 現行の共通描画半径 `0.22*h` は粒子位置を示す marker であり、物理的な粒子径ではない。
  これを大きくして物理密度を変えたように見せず、現在の `liquidVolume` が水・油を合算している点も踏まえて、
  表示用の占有場は水・油別に作る。marker は数値確認用 overlay として残す。
- 占有閾値は任意の `0.5` ではなく `sum(V0*J)` に対する表示面積で校正する。
  Gate 4 の初期占有率目安は `V0 / spacing^2 = 25 / 9^2 ≈ 0.309` であり、
  描画半径の変更や LiquidFun の粒子半径・圧力係数の移植で物理密度を調整しない。
- 空気圧ゼロの自由表面と、壁面条件を区別する。
  pressure projection をまだ使わない経路に不要な Poisson solve を導入しない。

**合格条件**:

- 静置・揺動で材料別粒子質量が不変。
- 表示面積と `sum(V0*J)` の差を校正後 5% 以内にする。
  １種類の密度だけに合わせた mass 閾値で合格しない。
- 液柱の圧力を EOS と重力による reference と比較する。
  近非圧縮条件では `p ≈ rho*g*depth`。境界・壁 cushion の領域は別に評価する。
- `h` と粒子間隔を細かくした場合の表面位置・面積誤差を記録する。

<a id="m4"></a>
### M4 — 少数場の接触・滑り・離脱

**前提**: M2 と、M3 の校正された幾何場。M1 の採否に合わせて P2G を拡張する。

1. 同じ Neo-Hookean のゼリー２個を別 field に置く。
2. 法線 impulse、離脱、摩擦ゼロの滑りを実装する。
3. 材料対摩擦、多点接触、壁 corner を追加する。
4. 水–油の２相、次に現在の水–油–粉または水–油–ゼリーの３場を比較する。
5. 最大４場まで検証する。全組接触の `O(M^2*G)` を計測する。

#### Gate 4 の三粒子小石は幾何 collider として扱う

Gate 4 の各小石は３個の Neo-Hookean 粒子で表す変形三角形であり、
`objectId` の分離や描画上の輪郭だけでは内部を満たす collider にならない。
また、200個の小石それぞれに `contactFieldId` を割り当てる設計は、
この段階の最大４場という予算に合わない。同じ field に全小石をまとめると、
P2G support が重なる近接小石の速度が平均化され、別物体が接着するおそれがある。

小石については material contact field と分け、各物体の３頂点から現在の三角形境界を毎 substep 再構成する。
三角形の AABB を接触 skin 分だけ広げて grid cell の候補 list に登録し、
水・油粒子は近傍三角形だけを厳密判定する。全液体粒子と全小石を総当たりする
`14,000 × 200 = 2.8M` narrow-phase pair/step は避ける。
粒子の移動区間も検査して tunneling を防ぎ、三角形内部または許容 gap 未満なら、
最近接辺へ押し戻し、辺上の barycentric 速度に対する inward normal velocity を除く。
初期条件は反発係数 0・接線摩擦 0 とし、液体の接線 slip を保つ。

接触半径は描画半径と別に扱い、`sqrt(V0*J/pi)` を初期候補として解像度と体積保存で校正する。
最終的には液体と三角形頂点へ等大反対向きの impulse を与え、
頂点側は contact-pair buffer から gather/reduce して float atomic scatter を避ける。
変形三角形の退化、候補 list overflow、衝突未解決は数値 status として報告し、無接触扱いで隠さない。
LiquidFun から採るのは圧力・形状復元・接触を分ける設計原則であり、
`SolveSolid` や共有粒子半径の係数を MLS-MPM の三角形 collider として直接移植しない。

接触判定は support 共有だけにせず、法線・gap・面積・接近速度を使う。
低質量閾値と法線退化の扱いを CPU/reference と共有し、未解決件数も表示する。
G2P は自分の field の速度を読む。旧密度差加速度は新経路では無効にし、
圧力荷重による浮力を独立に確認する。

clone の規則:

- ゼリーは新しい `objectId` と field を確保する。field 上限では明示的に拒否する。
- 液体は新 `objectId` を持っても、同じ phase の field に合流する。
- 粒子容量と field 容量を両方検査し、失敗時に ID や粒子数を部分更新しない。
- 選択・局所力の対象 field を明示する。現在の全材料向け drag を無告知で変えない。

**合格条件**:

- ２場の impulse 単体試験で運動量が保存され、離脱方向に力を与えない。
- gap が十分ある物体は support が重なっても反発しない。
- ２体衝突後の幾何学的貫入は初期目標 `0.25*h` 以下とし、解像度依存も記録する。
- 摩擦ゼロでは接線相対運動を保ち、摩擦ありは `|j_t| <= mu_contact*j_n` を満たす。
- 三粒子小石では液体粒子が三角形内部へ抜けず、衝突後の運動量残差が scene scale の許容値以内である。
- ３場以上の順序変更による差を測る。水–油の界面混合を単なる材料色で判定しない。
- clone/readback/render に field と材料状態が保持され、容量不足が UI に現れる。

格子未解像の薄片、自己接触、液体–粉体浸透はこの段階の対応外として表示する。

<a id="m5"></a>
### M5 — 粘性・表面張力・界面保持

**前提**: M3・M4。

- 先に粘性を加え、水と油の rate dependence、ゼリーの減衰を校正する。
  表面張力を同じ変更で同時に増やさず、原因を追えるようにする。
- CSF 原論文を全文確認し、balanced な力の離散化と曲率計算を比較する。
  水–空気、油–空気、水–油の係数を分ける。
- 表面張力・粘性の時間制約を M2 に追加する。
- 界面を横切る法線流量と混合領域を測り、必要なら転送 compatibility や
  界面輸送の改良を追加する。接触投影だけで完全な非混和性を宣言しない。

**合格条件**:

- 静止 2D 液滴の圧力差が `gamma/R` と 10% 以内。
  半径は十分な格子解像度を持つ試験に固定する。
- 静止液滴の spurious velocity は毛管速度 `sqrt(gamma/(rho*R))` の 1% 以下を初期目標。
- 粘性を変えた振動減衰の再現性と、外力なしの運動量・エネルギー挙動を確認する。
- 水–油界面の質量は保存され、混合帯が時間とともに無制限に拡大しない。
  許容帯幅は初期案 `2*h`。解像度変更で比較する。

<a id="m6"></a>
### M6 — 条件付き分岐: 圧力 projection／陰的体積応答

**着手条件**: M2 の測定で、要求する体積誤差と持続性能を
明示的 microstep の範囲では同時に満たせないことが確認された場合。
近非圧縮性の見た目を得るだけのために、最初から導入しない。

- [調査資料の S3](./RESEARCH-MLS-MPM-MATERIALS.md#sources) を出発点に、
  deviatoric/volumetric splitting と離散化を設計する。
- まず単一液体の MAC 格子、外気 `p=0`、壁の法線速度条件を実装する。
  既存 collocated MPM と同一の転送・演算子であるとは仮定しない。
- 圧力 solve は GPU の行列作用・reduction・PCG 等を候補とし、
  勾配と発散の整合性、nullspace、低占有セル、収束停止を設計する。
- 水–油へ広げる場合は密度差を持つ界面で圧力／法線流量を結合する。
  相ごとの独立 solve で界面条件が満たされるとはしない。
- まずゼリーは圧縮性のまま残す。固体の近非圧縮体積応答への拡張は別レビューを行う。
- explicit EOS の体積力と projection を二重適用しない。
  粘性、移流、せん断弾性、毛管波の時間制約は残る。

**合格条件**:

- 相対 residual `<= 1e-4` と正規化 divergence `<= 1e-3` を初期基準とし、
  境界を含めて測る。分母がゼロに近い場合の絶対基準も定義する。
- 非収束は専用 status と rollback/通知で扱う。固定回数 Jacobi の終了を成功としない。
- 静置・自由表面試験で面積 drift 2% 以内を目標とし、
  M2 経路との総 GPU 時間、メモリ、粒子挙動を比較する。

<a id="m7"></a>
### M7 — 操作回帰・モバイル持続性能・採用判断

**前提**: 採用する M1〜M5 の数値ゲート。M6 は必要な場合だけ。

- pause/resume、FPS／速度変更、左右揺れ、選択、drag、clone、
  capacity/field 上限、device loss、background 復帰を通して確認する。
- Android Chrome と WebGPU 対応 iOS Safari で５分以上の実時間試験。
  20,000 active・1×をまず基準にし、2×以上と粒子増殖は別の性能条件として記録する。
- 初期目標は描画 30 FPS、p95 frame interval `<= 40 ms`、
  実効時間倍率 `>= 0.95×`。発熱後の後半でも比較し、達成しない場合は profile を明示して見直す。
- GPU byte、pass／step 時間、CPU encode、queue 待ち、microstep 数、
  discarded time、射影・rollback、界面／体積誤差を一緒に残す。
- 粒子全量 readback が通常操作にないことを再確認する。
  診断の小さな定期 readback と、ユーザー要求時の full inspection を区別する。
- 合格後にのみ [既存ロードマップの Gate 5](./ROADMAP-MLS-MPM.md) へ結果を反映し、
  通常画面への段階導入を判断する。

## 5. 依存関係と最初の実装単位

```text
M0: 現行再現・計測・容量
  ├─ M1: prepare 最適化 ──────── 採否を決めて後続に使用
  └─ M2: 全物理 substep・圧力／復元
       ├─ M3: 自由表面幾何 ── M4: 少数場接触 ── M5: 粘性・界面
       └─ M6: 圧力 projection（必要な場合のみ。複数相は M3/M4 に依存）
                                      └──────── M7: 実機・採用判断
```

最初の実装単位は **M0 → M1 の A/B → M2**。
圧力・復元・安定性を測れるまで接触・表面張力を一度に増やさない。
次に M3 とゼリー２体の M4 を進め、その後で水・油・粉を共存させる。
M3 の表示試作や小規模 reference 試験は独立に進められるが、
共通 layout の変更は１つの契約としてまとめて統合する。

## 6. 検証の実施契約

| 層 | 試験 | 初期の判定方法 |
|---|---|---|
| 数式 | 応力、回転、圧縮、せん断、energy の有限差分微分 | CPU double。解析値と比較。energy/stress の整合性を先に確認 |
| layout | byte/offset、モデル・field ID、device limits、clone/readback | 純粋な既存 Vitest に追加。padding 再利用も契約テストを作る |
| GPU 対応 | small scene の１ step 全粒子、複数 step、材料混在 | `abs(error) <= 1e-5 + 1e-4*abs(reference)` を stress の開始基準とする。状態量は単位ごとに正規化した誤差を記録 |
| 保存量 | 外力・壁なしの mass、momentum、場間 impulse | CPU 基準テストの既存許容差を維持。GPU は scene scale で相対誤差を測り、初期目標 mass `1e-5`・momentum `1e-4` |
| 物理挙動 | M2〜M6 の個別 scene | 各ゲートの体積・形状・gap・圧力・界面閾値。NaN 検査だけで代用しない |
| 運用 | M7 の操作・５分以上の実機 | profile、機種、設定、後半の性能を含む結果 |

GPU の summation 順序は不定なので、bitwise 一致は要求しない。
運動量がゼロに近い場合は入力運動の代表量で正規化し、ゼロで割らない。
状態量の正規化には `h`、代表速度、`C` の代表逆時間、`F` の無次元値を使い、
１つの raw absolute tolerance を位置・速度・応力すべてに流用しない。

現行の局所３ probe は短い support に基づく初回比較用である。
新しい幾何 pass に近傍依存が増えれば、必要な全 support の閉包をサンプルに含める。
圧力 solve は非局所なので、局所サンプルだけの CPU oracle に置き換えず、
小さい**完全な領域**で検証する。

実装時は変更に対応する既存テストを最小範囲で実行する。
型・WGSL layout・UI へまたがる変更では `npm run check`、
GPU compile と挙動は実機ラボで確認する。全文 readback を性能試験の各 frame に入れない。

記録する結果の最小セット:

```text
source revision + dirty-state identifier
browser / OS / adapter / device limits / timestamp availability
active particles / capacity / grid / contact fields / GPU bytes
model parameters / particle mass-area convention / macroDt / microDt
render target / requested speed / effective speed / discarded time
CPU-GPU error by quantity / mass / momentum / area / J / clamp count
contact gap / interface width / shape error / pressure error
CPU encode / queue wait / GPU pass time if available / test duration
failures / device loss / unresolved normals / solver residual
```

この資料を追加した時点で、上記の新機能・閾値・性能が確認済みになったわけではない。
実装と合格記録は段階ごとに本書と既存ロードマップへ反映する。
