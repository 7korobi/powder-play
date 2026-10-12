# MLS-MPM 材料表現の調査: LiquidFun から学ぶ４論点

調査日: 2026-10-12

対象: ゼリーと水・油・粉体を、powder-play の GPU MLS-MPM 上で共存させる設計。
本書のコード状態・20,000粒子数値は調査時点の粉体版に対するスナップショットである。
現在の初期シーンは粉体を正三角形のNeo-Hookean弾性小石200個に置き換えた。
小石は各3粒子（計600粒子）で、水8,000・油6,000と合わせ14,600粒である。
小石の各辺は66 world units（枠幅の3%）で、直前の33-unit版から倍増した。
200個は14列×15行（最終行4個）・中心間隔68 world unitsに配置している。
小石置換後の検証状況は [MLS-MPM ロードマップ](./ROADMAP-MLS-MPM.md#現在の実装状況) を参照する。
実装の段階・依存関係・合格条件は
[GPU 材料表現の実装ロードマップ](./ROADMAP-MLS-MPM-MATERIALS.md) に分離する。
製品全体の採用判断は [既存 MLS-MPM ロードマップ](./ROADMAP-MLS-MPM.md) を維持する。

## 読み方と調査基準

- **確認済み**: 読んだソースコード・原論文に直接根拠がある内容。
- **導出**: 確認した式からの代数計算・概算。GPU 実測値ではない。
- **提案**: 今後の実装候補。現在の機能として扱わない。
- **未検証**: 実機挙動、物理校正、全文を取得できなかった文献など。

本書の現状・ハッシュはチューニング前の調査記録。
その後に実装した粒子別応力cacheとbatch pass集約、および性能比較は
[パフォーマンス計測資料](./PERFORMANCE-MLS-MPM.md) に追記する。
調査時の「P2Gで最大９回の応力評価」は、チューニング後の実装には当てはまらない。

powder-play の基準コミットは `3d21ca58eb5a9e7430db4a4c34f2cc0c0ea1f412`。
ただし、以下は**調査時の未コミット変更を含む作業ツリー**を読んだ結果であり、
このコミットだけでは再現できない。末尾に対象ソースの SHA-256 を記録する。
この文書の作成ではシミュレーションコードを変更せず、新しい GPU 試験も実施していない。

LiquidFun は `google/liquidfun` の
`7f20402173fd143a3988c921bc384459c6a858f2` に固定して確認した。
文献の確認範囲は [根拠一覧](#sources) に記す。

## 0. 出発点: LiquidFun のゼリーと現行 MLS-MPM

### LiquidFun の Elastic Particles は何か

**確認済み**: [Testbed の ElasticParticles][LF1] は粒子半径を `0.035` とし、
次の３グループを同じ particle system に生成する。

| 表示 | 粒子フラグ | 形・初期状態 | グループフラグ |
|---|---|---|---|
| 赤 | `b2_springParticle` | 半径 0.5 の円、位置 `(0, 3)` | `b2_solidParticleGroup` |
| 緑 | `b2_elasticParticle` | 半径 0.5 の円、位置 `(-1, 3)` | 同上 |
| 青 | `b2_elasticParticle` | 半幅 1・半高 0.5 の箱、位置 `(1, 4)`、角度 -0.5、角速度 2 | 同上 |

地面・左右の壁と、半径 0.5・fixture 密度 0.5 の動的 Box2D 球もある。
球の密度は粒子系の密度とは別物である。このサンプル自体は水との混合シーンではない。

粒子系の既定値は密度 1、`pressureStrength = 0.05`、
`elasticStrength = springStrength = viscousStrength = 0.25`、
`dampingStrength = 1`。表面張力用の２係数はそれぞれ 0.2、
`repulsiveStrength = 1`、`powderStrength = ejectionStrength = 0.5`。
グループの `strength` は既定 1。サンプルは弾性・バネ係数を上書きしない。
係数が存在しても、対応フラグのない処理がすべて働くわけではない。[LF1][LF2][LF4]

内部では、通常の位置・速度・フラグ・グループ所属に加えて、次を保持する。

- **バネの pair**: 粒子２個の index、フラグ、strength、初期距離。
- **弾性の triad**: 粒子３個の index、フラグ、strength、初期重心からの相対位置
  `pa/pb/pc` と補助量 `ka/kb/kc/s`。Voronoi 図由来の近傍三角形を使い、
  粒子間距離や接続条件で選別する。
- 接触リストと rest 接続は別物。pair/triad は生成・グループ操作・reactive 更新などで
  管理される。毎 substep の接触探索で rest shape を作り直すモデルではない。[LF2][LF3]

`SolveElastic` は、現在速度で予測した三角形と rest 三角形との最適回転 `R` を求め、
各粒子の速度に概ね次の補正を加える。[LF3]

```text
v_a += elasticStrength * triad.strength / dt
       * (R * restOffset_a - predictedOffset_a)
```

これは局所 shape matching 型の復元であり、変形勾配 `F`、応力テンソル、
Lamé 定数による連続体弾性ではない。１ triad の補正に着目すると、
位置誤差の一定割合を substep ごとに戻す。したがって `elasticStrength = 0.25` を
物理的なヤング率や `mu` にそのまま換算できない。

体積感・粒子の詰まりへの抵抗は、別の `SolvePressure` が接触 weight から作る。
その圧力尺度は `density * (particleDiameter / dt)^2`。
`SolveSolid` は異なるグループ間に depth ベースの追加反発を与える。
`solidParticleGroup` は `rigidParticleGroup` と異なり、形状を剛体に固定する指定ではない。
弾性復元だけでゼリーの全挙動が成立しているわけではない。[LF2][LF3][LF4]

接触フラグは２粒子のフラグの OR で、各 solver が対応フラグを判定する。
水・粉・粘性・弾性粒子は同じエンジン内に共存できるが、
粒子系の半径・密度などは共有される。材料ごとの連続体構成則とは設計が違う。[LF2][LF3]

**持ち込む知見**は「形状復元、圧力、接触、減衰を分けて設計すること」。
pair/triad の重複 scatter 書き込みは GPU 化で競合するが、GPU で不可能という意味ではない。
gather 化や coloring も可能である。ただし現行 MLS-MPM に対しては、
LiquidFun の CPU solver を移植するより、粒子の `F` と格子処理を拡張する方が自然である。

### 現行コードで確認した状態

| 項目 | 調査時の状態 | 根拠 |
|---|---|---|
| 構成則 | Neo-Hookean、Snow、水、油、簡易粉体を粒子ごとに選択 | [CPU solver](../src/lib/mpm/reference/solver.ts)、[WGSL](../src/lib/mpm/kernels/shaders.ts) |
| 粒子 layout | 20 word、80 byte。`F`、`C`、材料・密度・基準面積・塑性履歴など | [layout](../src/lib/mpm/kernels/layout.ts) |
| 格子 layout | １節点 32 byte、**単一速度場**。液体質量・現在面積も集計 | [WGSL](../src/lib/mpm/kernels/shaders.ts) |
| 数値 pass | clear → bucket 構築 → P2G → grid update → G2P → commit の６ pass | [GPU solver](../src/lib/mpm/kernels/solver.ts) |
| P2G | 整数 atomic の linked list を節点ごとに gather。構成則はリスト走査内で評価され、同じ粒子で最大９回 | [p2g](../src/lib/mpm/kernels/shaders.ts) |
| Gate 4 | 220×150 セル、`h = 10`、halo 込み 34,119 節点。水 8,000・油 6,000・粉 6,000 | [scene](../src/lib/mpm/scenes/gate4.ts) |
| 容量 | 初期粒子 20,000、確保容量 1,000,000。既存文書の GPU 確保量は約 201 MB | [scene](../src/lib/mpm/scenes/gate4.ts)、[kernel README](../src/lib/mpm/kernels/README.md) |
| 時間制御 | 固定 `dt = 1/240` 秒。複数 step を batch 送信し、上限超過時間を破棄 | [lab](../src/routes/labs/mpm/kernels/+page.svelte)、[scheduler](../src/lib/mpm/kernels/step-scheduler.ts) |
| 変形安定化 | `F` の積分だけを最大 64 分割。液体の `det(F)` を 0.5〜2、粉を 0.5 以上に制限 | [CPU solver](../src/lib/mpm/reference/solver.ts)、[WGSL](../src/lib/mpm/kernels/shaders.ts) |
| 水・油の分離近似 | G2P 後に局所密度差に基づく相対加速度を加算。界面・接触モデルではない | 同上 |
| 異常処理 | 永続 status、step 全体の粒子状態 rollback、小さな status の定期 readback | [GPU solver](../src/lib/mpm/kernels/solver.ts) |

既存ロードマップには、**変更前の条件**で 60 simulated seconds、240 step/s、
約 4.8M particle-steps/s、初回比較の最大絶対誤差 `2.44e-5` の確認記録がある。
新しい格子・密度・揺動・相対加速度を含む現在の条件の検証結果に流用しない。
queue 完了待ちも pass 単体の GPU 時間とは異なる。

<a id="pressure-volume"></a>
## 1. 圧力・体積応答

### 現行 Neo-Hookean の意味

**確認済み**: 現行モデルは圧縮性 2D Neo-Hookean で、`J = det(F) > 0` として次を使う。
`V0` は基準**面積**、`rho0 = m / V0` は面密度である。
単位は [CPU reference README](../src/lib/mpm/reference/README.md) に従う。

```text
P = mu * (F - F^(-T)) + lambda * log(J) * F^(-T)
A = m * C - 4 * dt * V0 / h^2 * (P * F^T)
```

**導出**: この応力に対応するエネルギーと Kirchhoff 応力は以下。
`B = F * F^T`、Cauchy 応力を `sigma`、圧縮を正とする平均圧力を `p` とする。

```text
Psi(F) = mu/2 * (tr(B) - 2) - mu * log(J)
         + lambda/2 * log(J)^2
tau = P * F^T = mu * (B - I) + lambda * log(J) * I
sigma = tau / J
p = -tr(sigma) / 2
```

重要なのは、`lambda * log(J)` だけが物体の圧力ではないこと。
例えば `F = s * I`、`J = s^2` では、

```text
p = -(mu * (J - 1) + lambda * log(J)) / J
K_2D = lambda + mu  (微小な面内等方圧縮の体積弾性率)
```

`mu` も体積応答に寄与する。ゼリーでは正の引張応力も必要なので、
水用の tension-free EOS をそのまま使わない。

Poisson 比は次元・拘束条件を明示する。通常の**3D** Lamé 定数から
平面ひずみの材料を考える対応は `nu_3D = lambda / (2 * (lambda + mu))`。
この場合 `nu_3D = 0.4/0.45/0.49` は `lambda/mu = 4/9/49` になる。
一方、純粋な 2D 材料を面内単軸・横応力ゼロで定義すると
`nu_2D = lambda / (lambda + 2 * mu)`。両者を「2D の Poisson 比」として混同しない。

### 水・油の圧力と現在の安定性

**確認済み**: 液体は弾性せん断を持たず、
`tau = K * min(J - 1, 0) * I` を使う。コード上の名前は `lambda` だが、
ここでは液体の圧縮係数を `K` と呼ぶ。

**導出**: `rho = rho0 / J` なので、圧縮側の Cauchy 圧力は
`p = K * (rho/rho0 - 1)`、膨張側はゼロ。
現在の係数から求めた基準音速 `c = sqrt(K/rho0)` と、移流を除いた Courant 数は次のとおり。

| 液体 | `K` | `rho0` | `c` | `dt * c / h` |
|---|---:|---:|---:|---:|
| 水 | 40,000 | 0.02 | 約 1,414.2 | 約 0.589 |
| 油 | 30,000 | 0.016 | 約 1,369.3 | 約 0.571 |

これは計算によるリスク指標であり、発散したという実測ではない。
弾性体の微小ひずみでは `c_L ≈ sqrt((lambda + 2*mu)/rho0)`。
大変形・Snow の硬化では接線剛性が変わるため、この式だけで全状態を保証できない。

**提案**:

1. `dt <= alpha * h / (c_bound + v_bound)` を基準に、
   **P2G・grid update・G2P 全体**を細分化する。`F` だけの分割とは区別する。
   初期比較用に `alpha = 0.3` を仮置きすると、静止状態の現在の液体でも
   1/240 秒を少なくとも２分割する。実際の係数は検証で決める。
2. まず弾性・体積応答を別々に測り、`mu`、`K`、密度、時間刻みを一緒に校正する。
   密度半減は音速を `sqrt(2)` 倍にし、安定性に対して無料の変更ではない。
3. 近非圧縮性を高い `lambda` だけで追う前に、過度な硬化・圧力振動・解像度依存を調べる。
   volumetric locking は候補リスクであり、現行コードで確認した不具合ではない。

圧力と偏差変形をより明確に分ける**別構成則の比較候補**は、
2D で `F_bar = J^(-1/2) * F` を用いた次のエネルギーである。

```text
Psi_iso = mu/2 * (tr(B)/J - 2)
Psi_vol = K/2 * log(J)^2
tau_iso = mu/J * (B - tr(B)/2 * I)
tau_vol = K * log(J) * I
K = lambda + mu  (現行式と微小ひずみ応答を合わせる場合)
```

これは**導出した設計候補**で、現行 Neo-Hookean と大変形応答が異なる。
従来モデルを無告知で置き換えず、エネルギーの微分と CPU/GPU の両方を検証する。
isochoric 分離自体が locking や CFL 制約を取り除くわけではない。
上の `F_bar` は等容部分であり、空間平均化を行う **F-bar 安定化法**とは別物である。

F-bar 型の体積ひずみ平均化は候補だが、文献 [S4] は今回書誌のみ確認した。
原定式化を確認するまでは、粒子の `J` を近傍平均に差し替える実装を採用しない。
強い体積保存が必要なら、体積成分の陰的処理・圧力 projection が次の分岐となる。
[S3] はその一次根拠だが、MAC 格子・圧力未知量を含む別設計であり、
現在の collocated 格子に Jacobi pass を足すだけでは再現できない。

<a id="free-surface"></a>
## 2. 自由表面

### 自由表面の運動と見た目は別

**確認済み**: MPM は粒子が占有する材料領域を積分し、空間を空気粒子で埋めなくても
自由表面を動かせる。弱形式の境界 traction は [S1] §3.3 に現れる。
外気圧を gauge ゼロとし、表面張力なしなら自然境界条件は `sigma * n = 0`。
弾性体では「境界の圧力スカラーをゼロにすること」と同義ではない。

現行の弱圧縮性 MPM は、自由表面のために必ず圧力 Poisson solve が必要な方式ではない。
一方、空節点の速度をゼロにするだけで、精密な圧力境界や液面が得られるわけでもない。
粒子円の描画半径を広げても、物理的な面積・界面位置・体積保存は変わらない。

### GPU 上の表面再構成

**提案**: 物理状態から材料別の占有場を作り、粒子表示と併用する。

```text
phi_i^a = sum_{p in phase a} w_ip * V0_p * J_p / h^2
```

これは nodal dual area を `h^2` とする内部節点の概算。
壁・halo の境界補正、サンプリング密度、等値線閾値を校正する。
全材料の mass を同じ閾値で切ると密度の違いで表面位置がずれるので、
幾何用には現在面積を使う。

現行配置は粒子間隔 9、基準面積 25 なので、一様な内部の占有率目安は
`25/9^2 ≈ 0.309`。無条件に `phi = 0.5` を物理的表面としない。
基準面積を `spacing^2` に合わせる案もあるが、その場合は液体質量
`rho0 * V0` まで変わるため、見た目だけの調整として行わない。

- 最初は fragment shader で占有場の輪郭・法線を可視化し、粒子表示と切り替える。
- 表示用の平滑化場と、接触・圧力に使う場を区別する。
  見た目を滑らかにしただけで、接触精度が向上したとは判定しない。
- 最後の P2G の場は G2P 前の粒子位置に対応する。
  最終粒子位置に一致する液面を描くなら、frame 終了時に GPU 内で再集計する。
- 空気は当面、外圧条件を与える周囲として扱う。気体粒子の全面導入は対象外。

### 表面張力・非混和性

表面張力は描画後処理ではなく力である。CSF 型の
`f_surface = gamma * curvature * grad(colorField)` は候補 [S5]。
ただし [S5] は今回書誌のみ確認したため、離散化の採用前に全文確認が必要である。
符号規約、曲率ノイズ、静止液滴の spurious current、運動量整合性を検証する。
2D 円の静止液滴では圧力差の大きさは `gamma / R`。
3D 球の `2*gamma/R` をそのままテストに使わない。

水–空気、油–空気、水–油の係数は別に持つ。
明示的表面張力には `dt` が概ね `sqrt(rho * h^3 / gamma)` に依存する制約も加わる。
法線・曲率用の GPU pass と、この時間制約を予算に含める。

**材料ラベルを維持すること、界面を表示すること、非混和性を維持することは別条件**。
単一速度場に色を付けても界面の数値混合は解決しない。次節の速度場分離と組み合わせる。

<a id="material-contact"></a>
## 3. 材料間の接触

### 現在できることと不足すること

単一格子速度への mass-weighted 集約は、同じ節点を使う材料の相対運動を平均化する。
格子解像度では no-slip 的な結合が起きるが、滑り・離脱・鋭い界面・粒子レベルの
非貫入まで保証しない。現在の密度差加速度も接触力ではなく、
等大反対向きの反作用や界面境界条件を保証する構成ではない。

**確認済み**: [S2] §3 は物体別の格子速度と共通重心速度を比較し、
接近時の法線拘束と Coulomb 摩擦を扱う。
**提案**: その考え方を踏まえ、最初は少数の contact field に速度を分離する。
以下の２場間 impulse は本資料の設計案であり、[S2] のアルゴリズムのそのままの再現ではない。

### ID を分ける

| ID | 意味 |
|---|---|
| `materialModel` | Neo-Hookean、水、油などの構成則 |
| `phaseId` | 水相、油相などの界面・物性分類。場の descriptor に保持する案 |
| `objectId` | 選択・複製などの操作対象 |
| `contactFieldId` | 独立した格子速度を持つ単位。今後追加する案 |

同じ構成則のゼリー２個でも、独立接触には別の contact field が必要。
逆に、水の複製ごとに全格子を１枚追加する必要はなく、同じ水相に合流できる。
同一物体の自己接触・切断はこれだけでは解決しない。CPIC [S1] は参考になるが、
colored distance field と転送の compatibility を含む別機能として扱う。

### 節点所有の接触投影

各節点で場 `a/b` の質量 `m_a/m_b` と速度 `v_a/v_b` を取得する。
単位法線 `n` は a から b を向く。占有場勾配・重心・gap を使って接触を判定し、
近接して接近中の組にだけ、反発係数ゼロの法線 impulse を与える。

```text
g = v_b - v_a
g_n = dot(g, n)
inverseMass = 1/m_a + 1/m_b
j_n = max(0, -g_n / inverseMass)

g_t = g - g_n * n
j_t_free = -g_t / inverseMass
|j_t| <= mu_contact * j_n

impulse = j_n * n + j_t
v_a -= impulse / m_a
v_b += impulse / m_b
```

接触が離脱方向なら引き戻さない。２場に等大反対向きの impulse を与えるため、
外力のない節点の運動量は保存される。ゼロ質量の場は接触組にしない。

重要な制約:

- **共通節点に質量があるだけでは接触としない**。
  3×3 support の重なりによる離れた物体の早期反発を検証する。
- 法線が退化する場合を計測し、無条件の sticky 結合やランダムな法線で隠さない。
- ３場以上では組の処理順・反復が結果を変える。
  固定順序と少数反復を CPU/GPU で合わせ、多点接触と順序依存を測る。
- 粉体の内部摩擦パラメータと、材料対の `mu_contact` は別。
  液体対には乾燥固体の Coulomb 摩擦をそのまま当てはめず、まず法線結合と slip を扱う。
- 圧力・接触を整合させても、浸透、濡れ、粉体への吸水までは含まれない。
- 接触後の G2P で旧密度差加速度を足すと拘束を再び変え得る。
  新経路では旧近似を明示的に無効化して比較し、浮力を圧力・界面荷重で検証する。

GPU では１ invocation が１節点の全場を所有し、少数場の impulse を局所計算する。
既存 bucket list を１回走査して各場へ gather すれば float atomic は不要。
場数 `M` の密な格子はメモリが `O(M * nodes)`、
全組比較は `O(M^2 * nodes)`。最初は `M <= 4` の PoC に限定し、
無制限の object 数と同一視しない。

<a id="elastic-recovery"></a>
## 4. 変形・弾性復元

### ゼリーはすでにある数値状態から作れる

`F` は rest から現在への変形、`C` は局所アフィン速度であり、
構成則の応力が P2G を通じて復元力になる。[S1][S6]
LiquidFun の triad を追加しなくても、Neo-Hookean 粒子群をゼリーとして使える。

- `mu` はせん断・形状復元、体積係数は圧縮への抵抗、密度は慣性に影響する。
- `F = R` の純粋回転では Neo-Hookean 応力はゼロ。
  解放後は `F = I` に強制するのではなく、回転を除いて rest shape に戻ることを測る。
- 液体はせん断記憶を捨てるが、ゼリーの `F` を液体型射影で消してはいけない。
- Snow は塑性射影・履歴・硬化を持つ。ゼリーの完全な復元と同一視しない。
  粉体の `J >= 0.5` も塑性圧密の実装ではない。

`F` の積分は現在 `F' = (I + dt*C') * F` を必要に応じて細分化する。
有限時間の純粋回転を厳密に積分する式ではないので、
回転中の偽ひずみ・長時間の形状 drift は独立したテスト対象になる。
APIC は transfer の散逸を抑える方法 [S6] であり、全シミュレーションの
エネルギー保存・壁衝突・減衰の正しさを自動保証しない。

### 復元と減衰を別々に調整する

**提案**: 最初は圧縮性 Neo-Hookean を維持し、落下・せん断・引張・回転を独立に測る。
Fixed Corotated は比較候補だが、最初の実装で構成則を増やす必須条件とはしない。

ゼリーの rate dependence を加えるなら、物理的粘性の候補は Kelvin–Voigt 型である。
空間速度勾配の近似を `L ≈ C` とし、2D で以下を比較する。

```text
D = (L + L^T) / 2
D_dev = D - tr(D)/2 * I
sigma_visc = 2 * eta * D_dev + zeta * tr(D) * I
tau_visc = J * sigma_visc
```

これも追加モデルの提案であり、現在の drag damping と別物。
単なる全粒子の速度減衰は並進運動まで失わせる。
明示的粘性には `dt` が概ね `rho * h^2 / eta` に依存する制約があり、
粘性を増やせば無条件に安定するわけではない。
油に水と違う「粘り」を持たせるにも粘性モデルが必要で、
現在の密度・EOS 係数の違いだけでは粘度差を表していない。

### GPU に合う最初の改良

現行 P2G が最大９回評価する `tau`、粒子質量、`A` を
粒子ごとの prepare pass で１回計算し、節点 gather が読む案を優先評価する。
Neo-Hookean の `tau = mu*(B-I) + lambda*log(J)*I` は代数的に同じなので、
`F^(-T)` とその後の積を明示的に作る必要もない。
Snow・粉体の分解計算にも再利用効果がある。[S1] §6 の粒子別 affine momentum が参考になる。

ただし、追加 pass・cache の読み書き・GPU メモリも増える。
単純な液体 EOS では計算削減より帯域増が不利な可能性があり、
「９回を１回にしたから９倍速い」とは結論しない。

## ４論点を横断する判定

| 論点 | 小さく確認する場面 | 測るもの |
|---|---|---|
| 圧力・体積 | 一様圧縮、単純せん断、静置液柱 | 応力曲線、`J`、面積、圧力、CFL、射影発生数 |
| 自由表面 | 平坦液面、揺動、静止円形液滴 | 面積誤差、表面位置、解像度依存、曲率、spurious current |
| 接触 | ゼリー２体、滑り、離脱、水–油 | 非貫入、相対速度、運動量、gap、界面混合、処理順依存 |
| 復元 | 回転、圧縮して解放、落下後の振動 | 偽応力、回転除去後の形状、周期、減衰、エネルギー drift |

１ step の CPU/GPU 一致は実装対応の証拠であり、物理モデルの妥当性の証明ではない。
NaN がないことや `J` clamp が効くことも、体積保存・接触・界面維持の合格には代用しない。
詳しい実施順と暫定数値基準は [実装ロードマップ](./ROADMAP-MLS-MPM-MATERIALS.md) に記す。

<a id="sources"></a>
## 調査根拠と未確認事項

### 外部の一次資料

| ID | 資料 | 確認した範囲・根拠として使う内容 |
|---|---|---|
| [LF1] | LiquidFun `ElasticParticles.h` | 固定コミットの全文。３グループ、半径、剛体球、初期状態 |
| [LF2] | LiquidFun `b2ParticleSystem.h` | pair/triad、既定値、critical velocity/pressure、粒子質量 |
| [LF3] | LiquidFun `b2ParticleSystem.cpp` | 接続生成、接触フラグ、Solve 順、Elastic/Spring/Pressure/Solid |
| [LF4] | LiquidFun `b2ParticleGroup.h` | solid と rigid の区別、グループパラメータと既定 strength |
| [S1] | Hu et al. (2018), *A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling* | 著者公開 PDF の §3、§5、§6。MLS transfer・弱形式・CPIC・affine momentum。論文の CPU ベンチマークを WebGPU の実測として扱わない |
| [S2] | Bardenhagen, Brackbill, Sulsky (2000), *The material-point method for granular materials* | 著者公開 PDF の §3。物体別速度場、接近時の法線拘束、摩擦。DOI: `10.1016/S0045-7825(99)00338-2` |
| [S3] | Stomakhin et al. (2014), *Augmented MPM for phase-change and varied materials* | 著者公開 PDF の §4.2〜4.5 と §5 の導入。等容・体積分離、圧力未知量、MAC 格子、陰的体積処理 |
| [S4] | Coombs et al. (2018), *Overcoming volumetric locking in material point methods* | Crossref で著者・題名・DOI 確認のみ。公開本文の取得は不成功。具体的安定化式の採用根拠にはまだ使わない |
| [S5] | Brackbill, Kothe, Zemach (1992), *A continuum method for modeling surface tension* | Crossref で題名・著者・DOI 確認のみ。CSF 離散化は全文確認を次の作業に残す |
| [S6] | Jiang et al. (2015), *The Affine Particle-In-Cell Method* | 著者の publication page の要旨。局所 affine 速度、PIC 散逸低減、transfer の角運動量保存 |
| [W1] | WGSL specification, Atomic Types | 通常の `atomic<T>` は `i32/u32`。現行 gather と整数 atomic を維持する根拠。`f32` atomic に依存しない |

LiquidFun の参照ソースのヘッダには zlib 型の許諾・出自表示・変更版表示・
notice 保持の条件がある。repository の marker 名だけから全ファイルを BSD と断定しない。
この資料はソース移植ではなく設計知見の記録である。
将来コードを取り込む場合は対象ファイルと同梱物の条件を確認して notice を保持する。

### 現行ソースの記録

以下は調査時の SHA-256。将来値が変わった場合、この文書の現状説明を再確認する。
未コミットコードのコピーや秘密情報はこの資料に保存しない。

| ソース | SHA-256 |
|---|---|
| [GPU layout](../src/lib/mpm/kernels/layout.ts) | `5eae313afadfbf5c86c385ca3ed674d7850b3e86bbdfd9b8cc48968d4d45f9be` |
| [WGSL](../src/lib/mpm/kernels/shaders.ts) | `7a882d46ac4712f944f838369e23437ef3ac1bff45090bb54c2ade69bab39be9` |
| [GPU solver](../src/lib/mpm/kernels/solver.ts) | `b99eca4addfd520cbfd986bfc9bfbeac23ef1817180cea339de9bd7bb9a57ad1` |
| [CPU solver](../src/lib/mpm/reference/solver.ts) | `c254ea112115d754e8cacb088273b8804fc8de2257546c094ebef85081541cad` |
| [Gate 4 scene](../src/lib/mpm/scenes/gate4.ts) | `2f1fc7688cc58c0b8f8d1590158b399b1a0ece11e0d731794d65ba79e0f28b6f` |
| [Step scheduler](../src/lib/mpm/kernels/step-scheduler.ts) | `942a9b4f1f92f31c3a30ad29d928ca142bebb5628ba24401f76040748b5d0f8f` |
| [Kernel lab](../src/routes/labs/mpm/kernels/+page.svelte) | `51d1fd1763c79f783dd1ab8c942aba026af1e87cb4bbae2ed342f55e13cc8ddd` |

残る確認: 現行条件の GPU 長時間安定性と性能、モバイル持続性能、
材料パラメータの校正、接触場・自由表面・表面張力の実装結果、
F-bar/CSF の原定式化。この文書でそれらの完了を宣言しない。

[LF1]: https://github.com/google/liquidfun/blob/7f20402173fd143a3988c921bc384459c6a858f2/liquidfun/Box2D/Testbed/Tests/ElasticParticles.h
[LF2]: https://github.com/google/liquidfun/blob/7f20402173fd143a3988c921bc384459c6a858f2/liquidfun/Box2D/Box2D/Particle/b2ParticleSystem.h
[LF3]: https://github.com/google/liquidfun/blob/7f20402173fd143a3988c921bc384459c6a858f2/liquidfun/Box2D/Box2D/Particle/b2ParticleSystem.cpp
[LF4]: https://github.com/google/liquidfun/blob/7f20402173fd143a3988c921bc384459c6a858f2/liquidfun/Box2D/Box2D/Particle/b2ParticleGroup.h
[S1]: https://yuanming.taichi.graphics/publication/2018-mlsmpm/mls-mpm-cpic.pdf
[S2]: https://www.math.unm.edu/~sulsky/papers/CMAME.pdf
[S3]: https://media.disneyanimation.com/uploads/production/publication_asset/108/asset/siggraph2014_melt.pdf
[S4]: https://doi.org/10.1016/j.cma.2018.01.010
[S5]: https://doi.org/10.1016/0021-9991(92)90240-Y
[S6]: https://disneyanimation.com/publications/the-affine-particle-in-cell-method/
[W1]: https://www.w3.org/TR/WGSL/#atomic-types
