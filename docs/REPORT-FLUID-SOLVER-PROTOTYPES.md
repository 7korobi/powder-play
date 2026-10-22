# 流体ソルバー試作候補の調査・比較レポート

作成日: 2026-10-21

更新: 同日夜にP2Gの試作・実GPU比較を実施した。§1〜6は方式選定の出発点、
最新の実装・実測・採否は[§7](#7-2026-10-21-実装と実測)を参照する。

## 要約

**現行 MLS-MPM をただちに置き換える根拠はまだない。** 一方で、1,600粒子の単相水試験では、描画なしでも実時間に追いつかない条件があり、現行の陽的 Tait 圧力応答と microstep 数は有力な改善対象である。

比較候補は、解く問題が異なるため分けて評価する。

1. **固定小数点 atomic scatter P2G** は、P2G のデータ構造を変える性能試験。microstep 数や物理モデルは変えない。
2. **PB-MPM** は、粒子・格子 MPM を保ちながら陽的 CFL 制約を緩和できるかを試す候補。現在の水の密度・圧力応答を保てるかは未確認。
3. **MAC 格子 pressure projection + FLIP/APIC** は、液体の非圧縮性を音速ではなく圧力 solve で課す別方式。液体の microstep を減らす可能性があるが、圧力 solve と多相・固体結合の費用は未計測。
4. **PBF / DFSPH** は大きめの時間刻みで液体らしい挙動を作る別候補だが、気体の熱力学や真空膨張を本来の目的として解くものではないため、初回試作の優先度を下げる。

調査時の推奨順は、**受け入れ基準の確定と既存 CFL 候補の品質確認 → P2G の限定 A/B → PB-MPM の水単相試作 → 必要なら MAC の単相試作**。P2G試作は§7の計測まで進んだが、PB-MPM・MAC・PBF/DFSPHの実装・WebGPU性能比較はまだ行っていない。

## 1. 何を比較するか

この問題には、少なくとも２種類のコストがある。

- **１ microstep の費用**: P2G、格子更新、G2P などの GPU 時間。
- **１ simulation second を進めるための費用**: 必要な microstep 数、pressure solve の反復数、batch の未処理時間も含む。

P2G を速くしても CFL が要求する microstep 数は減らない。逆に projection や PB-MPM で時間刻みを大きくできても、１ step の費用が高ければ総時間が短縮するとは限らない。各試作では両方を測り、P2G 単体の時間や論文の粒子数だけで採否を決めない。

## 2. 現行の比較基準と注意点

### 2.1 SI 水試験で分かっていること

[SI 水設定](../src/lib/mpm/scenes/si-fluid.ts)は、20 cm 級の箱、40×40 格子、5 mm 格子間隔、2.5 mm 粒子間隔、1,600 水粒子、1/120 s macrostep を使う。液体の有効音速 `20 m/s` は**実水の音速ではなく、計算負荷を制限するための人工的なモデル値**である。これを実物性として扱ったり、速度を上げるためだけにさらに下げたりしない。

[2026-10-21 の計測](./PERFORMANCE-MLS-MPM.md)では、Apple Metal 3 上の無描画試験で、12.014 秒の simulation を16.055秒で処理し、実効時間倍率は0.748倍だった。波速上限回復後は１ macrostep に約141 microstep が必要で、512 microstep/batch の上限により30 Hz入力の要求４ macrostepのうち３つしか進められない区間が生じた。これは描画込み性能やモバイル性能の測定ではない。

同じ資料の短い sloshing 試験では CFL 0.3 / 0.4 / 0.5 の実効時間倍率がそれぞれ0.89 / 1.17 / 1.46倍だった。一方、CFL 0.3 に対する粒子位置差は、0.4でRMS 3.18 mm・最大15.53 mm、0.5でRMS 5.37 mm・最大30.10 mmだった。これらは軌道差であり、体積誤差・圧力精度・視覚品質の合格を示さない。**まず許容する密度・体積誤差を決めて、この速度改善が物理的に受け入れ可能かを確認する。**

### 2.2 現行 P2G と過去の計測の扱い

現行 GPU 経路は、粒子が９個の二次 B-spline support を linked-list bucket に登録し、節点側がリストを gather する。応力・affine momentum の重複計算は prepare cache で１粒子１回に整理済みで、通常 batch の compute pass も集約済みである。したがって候補は古い最適化の再実装ではなく、**gather/list を固定小数点 atomic scatter に置き換える価値があるか**の比較である。[kernel 実装説明](../src/lib/mpm/kernels/README.md)

旧20,000粒子シーンの記録では P2G が GPU step median の約49%だったが、これは現行の1,600粒子 SI 水試験や retune 後の3,674粒子シーンの計測ではない。また、Apple GPU の短い per-dispatch timestamp は粗く、現在の SI workload で P2G 単独が主因とは確認できていない。P2G 候補は**低リスクで価値のある反証可能な性能試験**であり、先に高速化できると決めつけない。

### 2.3 120 FPS目標と既存profileの区別

ユーザーの再指定により、**SIラボでは120 FPSを目指す**。既存の通常画面の30 FPS目標、Gate 4 の15〜120 FPS選択・1/240 s物理刻みは別profileの記述で、SIラボの目標を30 FPSへ下げる理由にはしない。[SPEC](./SPEC.md)、[README](../README.md)、[材料ロードマップ M7](./ROADMAP-MLS-MPM-MATERIALS.md)。描画120 FPSは約8.33 ms/frameであり、物理刻み1/120秒と同じ意味ではない。以下を別々に判定する。

- 実効 simulation time が実時間に追いつくか（時間倍率・未処理時間）。
- 描画込みの frame interval が目標を満たすか（30 FPS基準と120 FPS目標を分けて報告）。
- 目標粒子数を発熱後も維持できるか。

## 3. 試作候補

### A. P2G: 固定小数点 integer atomic scatter

**仮説**: 粒子側から９個の support node へ整数 atomic で寄与を scatter すると、linked-list 構築と節点リスト走査が減り、１ microstep の P2G 費用が下がる。

**効く範囲**: P2G の GPU kernel だけ。CFL 条件、人工有効音速、microstep 数は変わらない。P2G が全 batch の有意な割合を占めなければ、simulation second あたりの改善も小さい。

**外部根拠と限界**:

- [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) は、同プロジェクトの README で、統合 GPU 上の SPH が約3万粒子、MLS-MPM が約10万粒子、離散 GPU の MLS-MPM が約30万粒子という目安を報告し、MLS-MPM の P2G に整数 fixed-point atomic を使う。これは WebGPU 上の実例として参考になるが、3D・異なる場面・GPU・目標条件の結果であり、今回の2D・千粒子規模・多材料 workload の性能予測には使えない。SPH と MLS-MPM の物理・実装条件も同一ではない。
- [Gao et al., GPU Optimization of Material Point Methods](https://pages.cs.wisc.edu/~sifakis/papers/GPU_MPM.pdf) は GPU P2G の設計候補と粒子密度を考える資料になるが、報告規模は今回より大きく、その速度値は移植できない。

**主な実装・数値リスク**:

- WebGPU の整数 atomic を使うため、実装前に mass、momentum、affine/stress 寄与など全 accumulator の単位・符号・最大合計を洗い出し、`i32` の範囲内に収まる scale を決める。浮動小数点値を整数へ写す量子化誤差を測り、overflow を無視できる設計にしない。
- mass・momentum・材料別の体積など複数 field がある。水単相で始め、既存値と必要な accumulator を明確にしてから複数材料へ広げる。
- node への atomic 競合が増える。リスト memory をなくした分が accumulator buffer・clear・変換・競合で相殺される可能性がある。
- scatter 順・atomic 加算は決定的ではない。現在の linked-list insertion order も非決定的なので、bitwise一致ではなく、短時間の状態誤差と長時間の保存量・挙動で比較する。

**採用判断**: 同じ物理・同じ microstep 列で whole-batch GPU median が既存の性能採用基準（10%以上）を上回り、p95・メモリ・数値検査が悪化しないときだけ採用候補にする。P2Gだけの測定値で決めない。詳細は[計測ロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md)を参照。

### B. PB-MPM: compliant constraint による MPM の安定化

**仮説**: MPM の粒子・格子転送を使い続けながら、陽的な剛い体積応答の時間刻み制約を制約ベースの半陰的処理で緩和し、必要な microstep 数を減らせる。

**外部根拠と限界**:

- [Lewin, A Position Based Material Point Method, SIGGRAPH 2024 Talks](https://doi.org/10.1145/3641233.3664323) と[公式実装](https://github.com/electronicarts/pbmpm)は、この方向の具体的な実装例である。発表資料は定性的に安定性向上を主張するが、今回と比較できる粒子数・fps・ハードウェア・圧力誤差の公開値は確認できていない。短い SIGGRAPH Talk の説明を「任意の大時間刻みで精度と性能が保証される」と解釈しない。
- 現行の MLS-MPM 自体も、[Hu et al. の論文](https://doi.org/10.1145/3197517.3201293)にある MPM 系である。二方向剛体連成はこの系統の能力だが、PB-MPM を導入して現在の固体・接触・bond pass がそのまま維持されることは別途確認が要る。
- [Becker & Teschner の weakly-compressible SPH](https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf)が示す通り、陽的 Tait EOS の人工音速を使う限り、SPHへ移るだけでは音響 CFL の性質は残る。PB-MPM を試す理由は「粒子法だから」ではなく、圧力／体積応答の時間積分を変えることにある。

**主な実装・数値リスク**:

- 制約の compliance と現在の Tait `p(rho)` / 体積応答をどう対応づけるかが未確定。Tait 圧力と新しい体積制約を二重に適用しない。`c_eff=20 m/s` の結果を実水の物性と誤認しない。
- timestep を大きくしたときの安定性だけでなく、密度・面積変化、静水圧、sloshing の位相、エネルギー減衰、粒子の軌道を確認する。
- 発表時の条件から現在の WebGPU shader・複数材料・接触試作への統合費用は推定できない。
- 液体向け圧力制約は、真空へ膨張する気体、膜の中の気体、熱・相変化を自動的に解決しない。

**試作範囲**: 単相水のみ。粒子配置・質量・domain・境界・重力入力は SI 水 baseline と揃え、同じ候補 macrostep で比較する。まず dt を増やしながら、保存量と体積誤差の限界で破綻する場所を計測する。公開情報に互換性が示されていないため、現行 solver への統合を前提にしない独立 prototype とする。

**採用判断**: 現在要求する体積／密度誤差を守りつつ、CFL microstep と全 batch 時間を減らせる場合に、次段の MPM 方式候補とする。閾値は計測前に固定し、実装後に速度に合わせて緩めない。

### C. MAC 格子 pressure projection + FLIP/APIC

**仮説**: 液体の近非圧縮性を、人工音速を用いる陽的 EOS ではなく、staggered MAC grid 上の圧力 Poisson solve によって課す。安定刻みを音速ではなく主に移流速度に結びつけ、SI 水の microstep 数を減らせる可能性がある。

**外部根拠と限界**:

- [Zhu & Bridson, Animating Sand as a Fluid](https://doi.org/10.1145/1073204.1073298) は、粒子と非圧縮性流体格子を組み合わせる FLIP 系を理解する基礎資料である。
- [Jiang et al., The Affine Particle-In-Cell Method](https://doi.org/10.1145/2766996) の APIC は粒子–格子転送方式である。**APIC 自体が pressure projection や CFL 緩和を行うわけではない。** 比較対象は MAC 圧力 solve と FLIP/APIC 転送を組み合わせた solver 全体である。
- 多相へ進む際には [Multiple Interacting Liquids](https://doi.org/10.1145/1141911.1141960) や [MultiFLIP](https://doi.org/10.1145/2159516.2159522) が追加論点となる。単相の projection prototype から水・油の界面が自動で正しく扱えるとはしない。

**主な実装・数値リスク**:

- 毎 step の圧力 solve が支配的になり得る。GPU 上の行列作用・反復・reduction・収束判定を含む総時間を測る。固定回数の反復を実行しただけで収束成功と判定しない。
- MAC の境界条件、free-surface `p=0`、低占有 cell、pressure nullspace、gradient/divergence の整合性が必要。変数密度の水–油界面では単相の Poisson operator のままでは足りない。
- Tait EOS を同じ相へ重ねて使うと圧縮応答を二重にする。projection の意味に合わせて液体モデルを設計し直す。
- 標準の単相 liquid projection は、固体・剛体・膜との二方向結合、熱、気体の真空膨張を含まない。全体を１つの solver で置換する案ではなく、液体 field と MPM 固体を連成する hybrid 候補である。
- 有効音速由来の制約を減らしても、粘性・表面張力・弾性・境界の時間制約が残る。速度改善は保証されない。

**試作範囲**: まず水単相、固定箱、自由表面、同じ重力・初期体積で小さな独立 prototype を作る。既存材料ロードマップ M6 の初期条件と基準（relative residual `<= 1e-4`、正規化 divergence `<= 1e-3`、面積 drift `<= 2%`、非収束を明示）を使い、総 GPU 時間と wall-time ratio を測る。[M6 の詳細](./ROADMAP-MLS-MPM-MATERIALS.md#m6)

**採用判断**: PB-MPM または現行 Tait 経路が圧力精度と目標性能を両立できず、単相液体で projection が数値ゲートと全時間性能を満たす場合に、液体専用 field の候補として継続する。水–油・剛体連成は別の採用ゲートにする。

### D. PBF / DFSPH（初回 prototype では後順位）

[Position Based Fluids](https://doi.org/10.1145/2461912.2461984) は近傍密度を目標値へ位置投影する方式、[DFSPH](https://doi.org/10.1109/TVCG.2016.2578335) は発散・密度条件を圧力反復で扱う方式である。大きめの時間刻みで液体らしい動きを作る用途には候補となるが、基準密度を保つ制約は Tait EOS の圧縮性圧力波や熱力学的な気体 EOS と同じではない。気体の真空膨張・風船内圧・熱・相変化には別モデルが必要である。

近傍探索も継続的な GPU コストになる。[WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) の通常 SPH と MLS-MPM の粒子数比較は、PBF/DFSPH の性能を直接示さない。見た目重視の液体を優先し、気体圧力や材料全体の連成を別系統で扱うと決めた場合に限り、追加候補として再検討する。

## 4. 横断比較

| 候補 | 主に改善を狙うもの | CFL microstep 削減 | 現行の粒子・固体との連続性 | 最も大きな未知 | 初回優先度 |
|---|---|---|---|---|---|
| 現行 explicit Tait MLS-MPM（対照） | 現在の多材料・粒子/格子 solver | なし。音響CFLに制約される | 既存基盤そのもの | 物理許容誤差でどのCFLまで上げられるか | baseline 維持 |
| fixed-point atomic P2G | １ microstep の P2G 時間 | なし | 高いが、量子化・scatterを検証 | 競合と整数精度が list gather より得か | 低コスト A/B |
| PB-MPM | MPM の体積・圧力安定化 | 期待されるが未実証 | 中〜高。現行 solver への接触等の再利用は未確認 | Tait 圧力との整合と、誤差を揃えた実速度 | 第二候補 |
| MAC + projection + FLIP/APIC | 液体の非圧縮性と音響CFL | 仕組み上期待できる | 液体粒子は保持可能、固体は hybrid 連成が必要 | pressure solve を含む総費用と多相境界 | 条件付き・高コスト |
| PBF / DFSPH | 見た目重視の液体・制約安定性 | 大きい刻みが期待される | 低〜中。物理モデルを置換する | EOS/気体要件との適合、neighbor search | 後順位 |

## 5. 比較実験と意思決定の順序

### Gate 0 — 条件と合否基準を固定

- 現行コードの commit / worktree、ブラウザー、GPU adapter、feature、粒子数、格子、物理係数、render の有無を記録する。
- まず単相 SI 水の1,600粒子で比較し、通過後に Gate 4 の3,674粒子混合シーンへ進む。異なる scene の結果を同じ baseline と呼ばない。
- 同じ seed・初期状態・重力入力列・領域・質量・壁条件を使う。異なる solver 間では軌道の完全一致を要求せず、密度/面積、運動量、自由表面、静水圧、発散/残差などの物理指標を合わせる。
- whole-batch GPU median/p95、CPU encode、queue wait、simulation/wall ratio、要求・完了・未処理時間、render frame interval を別々に測る。GPU の短い dispatch 分割値だけで順位をつけない。
- 120 FPSを判定目標にする場合は、8.33 ms/frame以下を別途測る。文書にある30 FPS/0.95x gateも併記し、描画周期と1/120 s物理macrostepを混同しない。

### Gate 1 — 現行方式の許容 CFL を品質込みで確認

既に速度差が観測された CFL 0.4 / 0.5 を、複数の重力方向・長めの sloshing・静置/自由表面で評価する。比較のため有効音速20 m/s、密度、EOS、格子、粒子数は固定する。密度/面積 drift、圧力の安定、rollback/retry、描画込み性能を確認する。これに合格すれば、方式変更より小さな変更で目標に届く可能性がある。現時点の短い軌道差だけでは合格にしない。

### Gate 2 — P2G 構造 A/B

water-only SI workload で list gather と固定小数点 scatter を切替可能にし、同じ microstep 列で whole batch を比較する。整数 scale と全 accumulator の最大値を事前に根拠づけ、差分精度、overflow/status、buffer byte、p95、他 material の回帰を記録する。性能差が採用基準に届かなければ、方式変更へ持ち込まない。

### Gate 3 — PB-MPM water-only proof

同じ初期状態と目標体積/密度誤差で、許される timestep を増やしたときの microstep 数、rollback、GPU total time、圧力/面積 drift、数値異常を測る。試作の段階では既存の固体・破断・接触 pass を統合しない。小さな solver が基準を満たした後で初めて統合費用を評価する。

### Gate 4 — 条件付き MAC projection prototype

Gate 1/3 で要求物性と性能を両立できないときに限定し、単相の2D MAC grid pressure solveを実装する。M6 の残差・発散・面積基準と、全時間費用を通過条件にする。通過後も水–油可変密度界面、剛体連成、ガス、膜は別々の試作・gate とする。

## 6. 現時点での判断

1. **現行方式を捨てる判断は保留する。** ただし0.748倍の実時間倍率は、描画なしの SI 単相試験でも scheduler の上限に達することを示しており、性能課題は将来機能を盛り込んだ結果ではない。
2. **有効音速20 m/sは性能比較の仮定値であって実水の物性値ではない。** P2G比較では固定し、CFL比較では密度・面積精度を同時に審査する。
3. **atomic P2G は総時間の短縮を確認する小さな A/B として妥当だが、CFL原因の解決策ではない。**
4. **PB-MPM は、MPMを維持したままmicrostepを減らせるかを調べる最初のsolver-level prototypeに適する。** 公開性能値がないため、発表資料から速度向上を見積もらない。
5. **MAC + FLIP/APIC は、液体の圧力方式を変える強い比較候補だが、最初の置換案ではない。** pressure solve の実測と PB-MPM/現行物性の品質結果を見てから、単相 prototype に進む。
6. **複数材料・気体・熱を一つのsolverで処理できるという前提は置かない。** projection/PBFは液体向け、密閉気体・真空膨張・膜・熱は別の状態方程式・連成方式が要る。

## 7. 2026-10-21 実装と実測

### 実装したもの

- 通常経路: prepareとbucket挿入を融合し、P2G自身が節点とbucket headを再初期化する。
  全格子clearはbatch冒頭に１回。剛体接触がない場合はCFL検査をG2Pへ融合した。
  接触後のCFL検査、全粒子rollbackのcommit barrierは残す。
  SIの１ batchは従来`8*k+2`から`5*k+3` dispatchになった（`k`はmicrostep数）。
- 実験経路: `createExperimentalMpmMlsAtomicGpuSolver`を追加。
  prepare後、９ support nodeへ質量・運動量・相別面積を整数atomic scatterする。
  shaderの`round`で`2^24`倍した値を格納し、個別寄与と加算途中のoverflowを検査する。
  overflowはstatus `65536`で全粒子を巻き戻し、飽和・黙ったfallbackは行わない。
- mass/momentumは設定された粒子質量、areaは設定された粒子基準面積で正規化する。
  signed accumulatorの範囲は約±128基準単位。速度・局所粒子密度の任意の上限を保証しない。
  追加scratchは44 byte/halo node、SIでは81,356 byte。
  surface描画用bucketは残るため、この試作はbucketメモリを削減しない。
- [SIラボ](../src/routes/labs/mpm/si/+page.svelte)に明示的な方式切替リンクを追加。
  標準はgather、`?p2g=atomic`は実験経路。切替時には初期状態へ戻る。
  有効音速20 m/s、CFL 0.3、格子・粒子間隔・EOS・壁摩擦の既定値は変えていない。
- SIの計算pumpをrAFから切り離した。１macrostepの完了後、残り時間があれば次のrAFを
  待たずに続行する。描画要求はrAFごとに高々１つ保持し、検証済みbatch完了後にだけ描く。
  GPU batchは同時に１つ、時間は捨てず、pause/dispose/errorで追加投入を止める。
  描画提出数に加えて実効時間倍率・未処理時間（投入済み未完了分を含む）を表示する。

### 環境と手順

baselineはcommit `824a18c97e232546120218e108270b52273116f5`の変更前solverを
`git archive`で固定し、別のlocalhost originから読み込んだ。
ViteのHMRで保存したconstructorが失われた試行はdispatch数が一致していたため棄却した。
以下はbaselineと新経路のdispatch数が異なることを確認した試行だけを使う。

計測した未commitソースのSHA-256:

- solver: `88580b11166ea27168cb37780547778b217f8c76551ca682f858ece76f123cca`
- shaders: `1197a40e62d4a5f19ef2e435092bdf37d8d44946f9e2a0e18b48ed13c946c288`
- SI計測module: `803b6a84345ca125ccf3460ffa305a1f7de9791a921fdd6d3378118074fc1409`

macOS / Headless Chrome 155 / `apple / metal-3` / fallback false、
`timestamp-query`有効、724×724 canvas、40×40格子、1,600粒子。
１batchは1/120秒、重力はsimulation timeで0.75秒ずつ下・右・左・下。
12 macrostep暖機後に３秒計測、baseline→融合→atomicを２組実施した。
GPU timestampは通常のwhole compute pass、各run全体に分散した36 batchを採取した。
wall値は360 batchのencode・adaptive readback・surface描画・queue完了待ちを含む。
timestamp mapと粒子snapshotの費用はbatch値の外側で、全loop時間の倍率も別に返す。
split-pass計測や無理な高FPS入力で値を作っていない。

### 同じ物理条件でのA/B

各値は２runの範囲。p95も少数試行・単一GPUの値である。

| 経路 | GPU median / batch | GPU p95 | readback・描画込み p95 | batch合計からの実効時間倍率 |
|---|---:|---:|---:|---:|
| 変更前 baseline | 8.45–8.52 ms | 9.11–9.31 ms | 11.50–11.60 ms | 0.798–0.800× |
| dispatch融合・gather | 7.80–7.86 ms | 8.32–8.39 ms | 10.70 ms | 0.885–0.888× |
| 融合＋atomic | 4.72–4.85 ms | 5.18–5.24 ms | 7.50–7.60 ms | 1.311–1.325× |

atomicはbaseline比でGPU medianを約43–44%、batchのwall medianを約40%短縮した。
融合単独はGPU median約8%で、旧P4の「GPU 10%短縮」条件には届かない。
**強い改善は融合単独ではなくatomicを含む試作全体の結果**として扱う。
microstep数は約140〜149のままで、人工音速由来のCFL費用は取り除いていない。
計測ループ全体（診断map/snapshot込み）の倍率もbaseline約0.796–0.797×、
atomic約1.304–1.317×であり、診断費用を外すことだけで実時間を達成した結果ではない。

120 FPSの8.33 ms予算にはp95で近づいたが、atomicのbatch最大は8.7 ms。
これは**unpacedな１macrostepの処理予算試験で、画面のpresented 120 FPSの実証ではない**。

### 長めの試験と粒子数の拡大

同じ1,600粒子の12秒試験では、atomic/CFL 0.3のGPU median/p95は4.85/5.37 ms、
readback・描画込みmedian/p95/maxは6.3/7.9/10.3 ms、実効倍率1.294×だった。
方向切替等で20 retryを要したが、要求した時間を完了した。
１秒ごとのsnapshotと最終値での最大絶対面積driftは0.041%、
粒子密度誤差のp95の最大は1.264%。これは全時刻・全粒子の最大誤差や静水圧の検証ではない。

atomic/CFL 0.4も同じ12秒を完了し、GPU median/p95 3.87/4.13 ms、
readback・描画込みp95 6.2 ms、実効倍率1.649×、21 retryだった。
ただし軌道は変わり、圧力・減衰・長時間安定性の合格ではないため**既定CFLは0.3のまま**。

粒子拡大ではdomain・5 mm格子・総質量・総基準面積を保ち、
粒子間隔を1/2・1/4、１粒子の質量・基準面積を1/4・1/16にした。
これは粒子積分の解像度を変えた別workloadであり、1,600粒子の物理的完全同等条件とはしない。
各条件１秒・120 batchの単一run（長時間/発熱試験ではない）。

| 粒子数 | 融合gather GPU median | atomic GPU median | atomic readback・描画込み p95 | atomic実効倍率 |
|---:|---:|---:|---:|---:|
| 6,400 | 19.40 ms | 6.09 ms | 9.0 ms | 1.072× |
| 25,600 | 81.26 ms | 15.66 ms | 19.2 ms | 0.464× |

両粒子数でretryはなく、両方式の面積・密度診断値は近かった。
ただしatomicでも6,400粒子で120 FPS予算を超え、25,600粒子は実時間を下回る。
**P2G構造の改善は有効だが、多粒子120 FPSにはstep数を減らす別の改善が必要**と判断する。

### 実画面・回帰

旧rAF依存の実画面では、atomicでも重力切替後に30 Hz寄りのbatch投入へ陥り、
５秒で未処理時間が約1秒へ増える試行があった。
計算pump変更後、約60 Hzの同ブラウザーで下・右・左・下を各３秒入力し、
各区間でsimulation timeが約３秒進み、描画提出60/s・実効1.00×、
観測した未処理時間最大0.025秒を確認した。pause中１秒は時刻不変、
resume後１秒は約0.99秒進んだ。これは短い操作試験で、120 Hz画面での確認ではない。
1200×2029と390×844のCSS viewportで表示を確認し、狭幅の横overflowはなかった。
方式切替による再起動、キーボードでの重力変更、pause/resumeが動作し、
最終ページのconsole warning/error・hydration error・画面alertはなかった。
390pxは同じデスクトップGPU上のviewport emulationで、スマートフォンの性能試験ではない。

- `npm run check`: エラー・警告０。
- `npm run test:unit -- --run src/lib/mpm`: 24 file / 226 test成功。
- `npm run build`: 成功。
- 既存５材料45粒子・８stepのCPU/GPU最大絶対差はgather/atomicとも`1.396e-5`、
  batch対単発step差は０（既存の非SI回帰fixture）。
- GPU overflow試験は１粒子の個別寄与・512粒子の加算総和の両方でstatus `65536`、
  永続error、および全粒子のbitwise rollbackを確認した。
- 既存のGPU pick/drag/clone/capacity、129粒子rollback、水–油接触、三角形接触、
  破断・broken状態の保持のfixtureは成功した。
- 既存`benchmark:mpm`全体の固定dt Gate 4性能preflightは、変更前baselineと新gatherの
  両方でparticle 1170の変形更新上限に失敗した。その性能部は成功扱いせず、
  後半の回帰fixtureだけを分離して実行した。
- SI単位へ全５材料を無校正で並べた別の予備fixtureでも、変更前・新gatherの両方に
  SnowのCPU/GPU差（`C.m11`約0.593）が出た。既存fixtureの成功をSI多材料の合格と
  読み替えず、この未校正条件の問題は今回修正していない。

### 再実行

[計測module](../src/lib/mpm/experiments/si-performance.ts)と
[GPU回帰module](../src/lib/mpm/experiments/transfer-regression.ts)はdev専用。
既存のdev serverを起動し、別のlocalhost tabを作ってから実行する。
計測中はsourceを変更せず、check/buildによるHMR reloadや他のGPUラボとの同時実行を避ける。

```sh
npm run dev -- --host 127.0.0.1 --port 4181
chrome-devtools new_page http://127.0.0.1:4181/@vite/client
# 以下の <page-id> は上で作った自分のtabのIDに置換する
chrome-devtools evaluate_script 'async () => {
  const {runSiPerformanceExperiment} = await import("/src/lib/mpm/experiments/si-performance.ts");
  const {particleState, ...result} = await runSiPerformanceExperiment({
    transfer: "atomic", simulationSeconds: 12, render: true
  });
  return result;
}' --pageId <page-id> --filePath /tmp/si-atomic-result.json
chrome-devtools evaluate_script 'async () => {
  const t = await import("/src/lib/mpm/experiments/transfer-regression.ts");
  return {
    parity: await t.runMpmTransferRegression(),
    overflow: await t.runMpmAtomicOverflowRegression()
  };
}' --pageId <page-id>
```

`transfer: "gather"`、`courantLimit: 0.4`、`particleRefinement: 2 | 4`で比較できる。
返す`particleState`は計測外の最終全粒子readbackで、位置・速度・C/Fなどの追加比較用。
通常画面では全粒子を毎frame読み戻さない。

### 採否と次の試作

atomicはSIで比較を継続する価値が十分あるが、他GPU・長時間・高速度/局所高密度・
実120 Hz画面の受け入れ前なので、標準経路へ自動採用しない。
120 FPSは引き続き目標であり、60 Hz画面の60提出/sやsynthetic budgetを達成扱いしない。

次は**PB-MPMのwater-only prototype**を優先する。公開実装の液体部分は、
体積を保つ変形変位のtrace制約と近似的な体積更新を使い、任意のTait `p(rho)`をそのまま
陰的に解くコードではない。[参照した実装](https://github.com/electronicarts/pbmpm/blob/main/shaders/g2p2g.wgsl)。
したがって20 m/sのEOSと等価と宣言せず、同じ箱・総質量・水位で
面積drift、圧力/静水位、減衰、iteration数対GPU総時間を先に比較する。
PB-MPM/MACの実装と速度測定は今回まだ行っていない。

## 参照

- [Hu et al. (2018), MLS-MPM with displacement discontinuity and two-way rigid-body coupling](https://doi.org/10.1145/3197517.3201293)
- [Becker & Teschner (2007), Weakly Compressible SPH for Free Surface Flows](https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf)
- [Gao et al. (2018), GPU Optimization of Material Point Methods](https://pages.cs.wisc.edu/~sifakis/papers/GPU_MPM.pdf)
- [Lewin (2024), A Position Based Material Point Method](https://doi.org/10.1145/3641233.3664323), [公式リポジトリ](https://github.com/electronicarts/pbmpm)
- [Zhu & Bridson (2005), Animating Sand as a Fluid](https://doi.org/10.1145/1073204.1073298)
- [Jiang et al. (2015), The Affine Particle-In-Cell Method](https://doi.org/10.1145/2766996)
- [Losasso et al. (2006), Multiple Interacting Liquids](https://doi.org/10.1145/1141911.1141960)
- [Boyd & Bridson (2012), MultiFLIP for Energetic Two-Phase Fluid Simulation](https://doi.org/10.1145/2159516.2159522)
- [Macklin & Müller (2013), Position Based Fluids](https://doi.org/10.1145/2461912.2461984)
- [Bender & Koschier (2017), Divergence-Free SPH](https://doi.org/10.1109/TVCG.2016.2578335)
- [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean)
- [powder-play 性能計測資料](./PERFORMANCE-MLS-MPM.md)、[MLS-MPM 性能ロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md)、[材料ロードマップ](./ROADMAP-MLS-MPM-MATERIALS.md)

公開論文・README の粒子数や安定性主張は、その原資料の workload に対する情報であり、powder-play の2D・対象端末・120 FPS・多材料条件での性能保証ではない。代替方式の採否は、上記 gate の同条件計測後に決定する。
