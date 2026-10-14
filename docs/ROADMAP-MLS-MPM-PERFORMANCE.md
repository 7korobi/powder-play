# MLS-MPM パフォーマンスチューニングのロードマップ

作成日: 2026-10-12

**計画の確定**: この文書と [計測・検討指針](./PERFORMANCE-MLS-MPM.md) を先に作成し、
その後に下記 P1〜P3 のソース変更をまとめて行う。
物理拡張の [材料ロードマップ](./ROADMAP-MLS-MPM-MATERIALS.md) とは分け、
`dt`、材料係数、密度、格子、粒子数、壁、操作意味を維持する。

結果: P1〜P3を実装し、暖機後の旧→新３組でGPU step medianを11.8〜16.1%短縮、
中央値ベースで約14.6%短縮した。P4のmedian改善基準を満たし、採用する。
追加16 MBのcache、p95の変動、CPU時計分解能の制限、回帰結果は
[性能資料の結果欄](./PERFORMANCE-MLS-MPM.md#5-実装後に残す比較) に記録した。
この測定は後に三粒子弾性小石へ置換された旧20,000粒子・粉体シーンを対象とする。
2026-10-12の14,600粒子・200小石sceneは、後に加わった96粒子の破断solidを含まない。
現行14,696粒子sceneで同じA/B条件を再測定した結果ではない。
計画を先に確定した履歴が分かるよう、以下の変更内容・採否基準は維持する。

## P0 — 計測と候補の選定

状態: baseline 取得済み。シミュレーションソース未変更。

- 条件と指標を計測文書へ記録する。
- GPU timestamp で P2G の約49%という寄与を確認。
- CPU が８ step 当たり48 pass を開閉していることを確認。
- ５構成則の小規模 CPU/GPU 比較と既存39テストで出発点を確認。

今回は **応力の粒子別 prepare** と **compute pass encoding の集約** を選ぶ。
容量変更・CFL変更・物理モデル変更・active-grid・sort・subgroup はこの変更に含めない。

## P1 — 応力を１粒子１回計算する

状態: 実装・検証済み。P0完了後に着手した。

### 変更内容

1. `prepareParticles` dispatch を追加し、モデル検証、応力、affine momentum
   `A = m*C - 4*dt*V0/h^2*tau` を active 粒子ごとに１回評価する。
2. scratch は `array<vec4<f32>>` とし、**16 byte/capacity** で `A` を保持する。
   波速診断等を含む材料ロードマップの32 byte案ではなく、今回必要な量に限定する。
3. P2G は cache を gather。軽い質量・液体現在面積の計算は残す。
4. 初版は構成則の式・演算順をできるだけ保持する。
   Neo-Hookean の代数簡約や新構成則を同時に導入しない。
5. scratch は毎 step 更新。batch 内で使い回した古い `F/C` の応力を読まない。
6. setup、bind group、layout byte、device limits、capacity、dispose を更新する。
   P2G の storage buffer requirement は５から６へ増えるので明示的に検査する。

### 維持すること

- error flags・粒子 index・非有限値検査・全 step rollback。
- 粒子80 byte layout、object/material state、clone/readback/render の互換性。
- 入力・出力粒子 buffer を scratch として流用する隠れた alias は作らない。
- 容量を黙って減らさない。1,000,000 capacity の scratch は追加16 MB、
  総主要 buffer は約201 MBから約217 MBとなる。

### 合格条件

- 実 GPU shader compile、５モデルの１ step／複数 step parity。
- invalid input/更新失敗時に異常を通知し、rollback と永続 status が保たれる。
- prepare+P2G の合計と step 全体を測る。P2G 単体だけの改善で採用しない。

## P2 — dispatch 順序を維持して batch の pass を集約

状態: 実装・検証済み。P1と同じソース変更batchで行った。

### 変更内容

１ `stepBatch` の command encoder 内で１ compute pass を開き、
各 step の dispatch を順番どおり encode する。

```text
clear -> prepare -> buckets -> P2G -> grid -> G2P -> commit
  -> 次 step の clear -> ... （ping-pong を更新）
```

compute dispatch ごとの usage scope と実行順を維持する。
workgroup 内 barrier を異なる dispatch の同期の代用品にしない。
既存の pick/clone の独立 command と、小さな queue 完了待ちの制御は維持する。

### 合格条件

- 通常 batch は１ compute pass、７ dispatch/step。
  空粒子の dispatch は既存と同様にゼロ workgroup を送らない。
- `stepBatch(k)` と同じ回数の `step()` の parity。
- command encoding の median/p95、GPU batch 全体の差を測る。
- buffer binding/usage error がなく、異常 step 後の rollback が全粒子へ作用する。

## P3 — 再現可能な benchmark と回帰試験

状態: 実装・検証済み。P1/P2の検証を常設した。

- Node の標準 WebSocket/CDP と既存の Vite dev server を使う
  `scripts/benchmark-mpm.mjs` を追加する。新依存パッケージは入れない。
- localhost の自分の benchmark tab だけを作り、終了時に閉じる。
  他の共有・非共有 tab は読み取らず操作しない。
- hardware adapter/fallback、feature、設定、warmup、median/p95 を JSON で返す。
- batch 全体の timestamp と、kernel attribution を分ける。
  attribution は dispatch ごとに pass を分ける診断経路で、通常実行時間と混同しない。
- render と全量 particle readback は kernel timing の外側。
  数値試験では小規模全粒子 readback を使い、５モデルと操作 state を検証する。
- timestamp 非対応・数値異常・CDP 未接続は明示エラーとする。
  fallback adapter の結果をモバイル／実 GPU の速度保証に使わない。
- layout・device limit・step encoding のテストを追加し、
  関連 test、type check、実 GPU ラボを確認する。

## P4 — 採否・結果の記録

状態: P1〜P3後に実施済み。median基準を満たし採用。実機持続試験は後続に残す。

### 採用基準

- 固定 Gate 4 条件で GPU step median **10%以上改善**、
  または高 step 数 batch の CPU encode が **20%以上改善**。
  時計分解能で CPU 判定ができない場合は GPU 基準を使う。
- GPU total/queue wait の p95 を併記し、ノイズが大きい場合は再測定する。
- ５モデルの８ step small-scene 最大絶対誤差は初期基準 `<= 1e-3`。
  項目別の誤差も確認し、これは物理校正の証明ではない。
- rollback、clone、pick、drag、pause、数値 status、型検査が退行しない。
- 追加メモリと新しい６ storage binding requirement を記録する。

採用基準を満たさない候補は残さず、この作業で加えた変更だけを修正する。
ユーザーの既存変更を戻したり、設定を有利に変えて合格にしたりしない。

### 記録先

- [計測指針の結果欄](./PERFORMANCE-MLS-MPM.md#5-実装後に残す比較):
  before/after、環境、方法、数値結果、操作回帰、未確認事項。
- [kernel README](../src/lib/mpm/kernels/README.md):
  新しい pass/cache/メモリ／端末制約。
- [既存ロードマップ](./ROADMAP-MLS-MPM.md):
  現行実装の変更と合格範囲。材料機能の gate を勝手に完了扱いにしない。

## Gate 4 接触prototype追加後の補足計測（14,600粒子、破断solidなし）

M4 の接触dispatchを含む当時の水・油・小石シーンを、P0〜P4で使った再現可能benchmarkで追加計測した。
14,600 active、1,000,000 capacity、200小石、固定 seed、Apple Metal 3 / Chrome 154。
batch profileは0.37094 ms/step median・0.37371 p95、GPU buffer確保量は221,564,392 byte。
stages profileは80 compute pass/8-step batchで診断し、`buildTriangleField` 0.13918、
`solveBondedTriangles` 0.10010 ms/step medianが上位だった。測定条件・全stage値は
[性能資料](./PERFORMANCE-MLS-MPM.md) に記録する。

計測で分かった候補のうち、三角形 SAT の AABB reject、contact impulse gather 範囲の縮小、
field broad-phase の平方距離化を実装した。変更前後は各１回の参考計測で、制御された反復A/Bではない。
P4のM1採否を変更する結果ではなく、接触追加後の性能合格も意味しない。

## 2026-10-13計測snapshot: 14,696粒子scene（adaptive CFL変更前）

初期ラボと同じ水8,000・油6,000・小石600粒・破断solid96粒を含むsceneを、
Mac / Headless Chrome 154 / Apple Metal 3で実GPU計測した。5回のbatch runのGPU step medianは
0.41254、0.40143、0.40109、0.36103、0.39817 ms、median-of-mediansは0.40109 ms/step。
runごとのp95の中央値は0.43035 ms/step。第4runは速い側の値で、600 msのwall-clock warmupにより
runごとのwarmup step数・scene状態が一致しないため、単一runの値だけで評価しない。
GPU buffer確保量は221,626,976 byte。stage診断では `buildTriangleField` 0.13882、
`solveBondedTriangles` 0.09796 ms/stepが上位だった。

上記のelapsed-timeとS3/S4 fixture結果は、作業treeにあるadaptive CFL microstep/recovery変更前の
sourceに対するbaselineである。CFL変更後はsubstep数・retry・時間会計が変わるため、
以前のms/stepを現行runtimeの性能とみなさず、M2検証後に同一条件・新source hashで再計測する。

2×で必要な480 step/sに対し、GPU timestampから導くcompute時間は約0.193秒/実時間1秒。
32×の7,680 step/sは約3.080 GPU秒/実時間1秒となり、この端末・sceneでは計算だけでも
現状のまま持続できない。これは描画なし・短時間のGPU計算からの導出であり、
画面上の実効時間倍率や発熱後の性能を測ったものではない。
第5回実行では別途、小規模GPU受け入れfixtureも通過した。水・油2粒子/8 stepで相別質量
各0.25・最大運動量残差1.1921e-7・CPU/GPU最大差3.8653e-5、4粒子/3 bondの破断fixtureで
閾値下damage 0.499992・閾値上のbond index 1のみ破断・CPU/GPU最大粒子状態差7.63e-7を確認。
rest lengthの初期破断bondも2 step保持した。scene-scale接触・運動量・実sceneの破断patternは未検証で、
物理gateは未通過。詳細な条件・誤差範囲・全stage値は
[性能資料](./PERFORMANCE-MLS-MPM.md)を参照する。

## 後続: 今回実装しない調査

- mobile で capacity と機能 profile を見直し、５分以上の持続性能を測る。
- canonical pair cacheはSAT再計算を減らすが、`O(G·B)` triangle-field探索と`O(B²)` pair列挙は残る。
  triangle-field用の16×16-node tile candidate bitsetは512 body以下で実装済みだが、
  brute-force oracle、dense/boundary case、mobileでのscene-scale受け入れは未完了。
  body-pairの広域候補索引は下記P5.3で引き続き計画する。
- dense node の list 長・偏りを収集して sort/tile gather を比較する。
- active node/tile と full-grid の crossover を測る。
- physics と render の decoupling、queue gate、pick の大粒子数 search を比較する。
- SoA、workgroup size、subgroup、`f16` は精度と対応端末を確認するまで採用しない。
- CFL と体積保存・界面機能は材料ロードマップで別に進める。

## 2026-10-13 canonical triangle-pair cache A/B

状態: 実装・実GPU検証済み。P4の10% median改善基準を満たし採用する。

同じ14,696粒子sceneで、cache導入前5 runのGPU median中央値は0.679936 ms/step、
導入後5 runは0.557056 ms/stepで **18.1%短縮**。run p95の中央値は0.696320から
0.565248 ms/stepへ18.8%短縮した。通常batchの1 compute passは維持し、GPU bufferは
4 MiB増えた。環境・warmup・source hashes・stage profileは
[性能資料のA/B記録](./PERFORMANCE-MLS-MPM.md#2026-10-13-canonical-triangle-pair-cache-ab)を参照。

最大512 active bonded bodiesでは、`gatherTrianglePairContacts`がunordered pairごとに
SATを一度だけ計算してbounded cacheへ格納する。contact countは両bodyへatomic加算し、
後段のimpulse passが同じrecordを再利用する。512 bodiesを超える場合は従来のdirect pair
scanへfallbackするため、全体の二次candidate enumerationは解消しない。追加dispatchと
4 MiB bufferを含めて測定し、小規模三角形衝突fixtureは4 stepのCPU/GPU差0.000381以内、
運動量残差0.000019以内だった。大規模dense pile、scene全体の貫入、長時間物理品質は未検証。

## P5 — 接触候補の空間索引（field部分実装、pair未実装）

状態: `buildTriangleField` は512 body以下で16×16-node tile candidate bitsetを使用する。
512 body超では従来の全走査にfallbackする。body-pairのspatial broad-phaseは未実装であり、
P5全体の採用ゲートも未通過である。測定条件と現状のA/B値は
[性能資料](./PERFORMANCE-MLS-MPM.md#2026-10-14-tile-field-candidates-and-pair-traversal)を参照。

**問題**: 現行sceneの `G=34,119` halo込みgrid nodes、`B=200` 三角形colliderでは、
full scan時およびdense overlapのworst caseでは `buildTriangleField` が最大
`G·B ≈ 6.82M` node/body候補をsubstepごとに調べる。tile bitsetはsparse caseの候補を絞るが、
実際のnode/body candidate countはまだ計測していない。
`gatherTrianglePairContacts` は最大 `B·(B−1)/2 = 19,900` unordered SAT候補を評価し、
`applyTrianglePairContacts` はactive contact bitsetを使い、sparse caseでは接触候補のみを処理する。
全pairが接触するdense caseの上限は `B·(B−1) = 39,800` directed contactsである。
1,000 colliderならfull-scan fieldは約34.12M、unordered SATは499.5k、dense directed contactsは999k。
pair cacheはSAT再計算を省き、active-contact bitsetはsparse apply走査を減らすが、
pair-gatherの候補列挙は二次のまま残る。

### P5.1 — 同条件の計測・小規模oracleを固定

- Gate 4の200-body測定は完了。candidate数を記録するinstrumentationと小規模brute-force oracleは未完了。
- collider数を400 / 800 / 1,000以上へ増やす。grid resolution、材料係数、
  particle count、dt、seedを固定し、uniform sparse配置とdense pileを別々に測る。
- 大きなsolidは粒子数 `P` と疎なbond数 `E` を独立に変え、既存の小石collider数 `B`
  の試験と混同しない。
- candidate body-node checks、unique body-pair数、exact triangle/SAT narrow-phase数、
  cell listの平均・p95・最大長、overflow数、`buildTriangleField` / SAT時間、
  CFL substep数、total batch p50/p95、GPU memoryを記録する。
- 小さいsceneでは全件走査のbrute-force結果をoracleとして保存する。
  spatial broad-phaseはfalse positiveを許容するが、false negativeを許容しない。

### P5.2 — triangle field tile bitset（部分実装）

- `buildTriangleFieldCandidates` が16×16 node tileごとに、field marginへ届き得るbodyを
  512-bit bitsetへ記録する。各nodeでは従来のAABBとtriangle-distance判定を再実行し、
  body ID昇順を保つ。固定候補listの上限や候補切り捨てはない。
- 512 body超ではcandidate passをdispatchせず、`buildTriangleField` が全bodyを走査する。
- Gate 4ではcandidate生成stage 0.028062 ms/step、field stage 0.022609 ms/stepのmedian。
  索引前のfield stage 0.120710 ms/stepと比べ、両stage合計medianは約58%低下した。
  `profile=batch` medianは0.290955から0.231175 ms/stepへ約20.5%低下した。
- 受け入れ未完了: GPU field全node出力のbrute-force differential oracle、
  tile境界・退化形状・dense overlap・512超fallbackのshader実行確認、candidate数counter、
  target mobile GPUでの計測。現時点では固定Gate 4のbenchmarkと既存物理acceptanceまで。

### P5.3 — body pairのbroad phase

- body AABBを空間cellへ登録し、同一/隣接cell内のpair候補だけを作る。
- 同じ2 bodyが複数cellで出会っても一度だけSATへ渡す。canonical cell ownerか、
  `min(id), max(id)` keyのsort/unique等、再現可能な重複除去を比較する。
- pairごとに狭域SATを一度だけ実行し、両bodyへのequal-and-opposite impulseは既存の
  owner/gather経路で反映する。floating-point atomic scatterを追加しない。
- 小石数が少ない場合はbrute-force経路の方が安い可能性があるため、切替閾値は実測で決める。

uniform gridは同程度サイズの小石の第一候補とする。非常に大きな構造物と小石が混在し、
大AABBが多数cellへ展開される場合は、hierarchical grid、BVH/LBVH、粗いobject proxyを
比較する。fracture objectは内部bondを疎な局所graph (`O(P + E)`) として維持し、
外部contactはsurface/粗いSDF等のobject-level proxyで扱う案を優先する。破断後のfragment
proxy更新は別途測定する。

### P5.4 — 採用ゲート

- 乱数seed付きのsparse/dense、境界、退化三角形、高速移動caseで、candidate listが
  brute-forceが発見する全接触を含む。overflowが0、またはoverflowを明示的に処理する。
- 接触後のpenetration、momentum residual、order dependenceが固定baselineの許容範囲を
  超えない。候補重複でdouble impulseを与えない。
- collider数と大型solidの `P/E` を別々に増やした実測を公開し、target mobile GPUで
  選んだscene budgetに対するp95 frame/physics budget、memory、CFL costを満たす。
- sparse caseで候補数が総 `G·B` / `B²` から局所参照数に縮小することをカウンタで確認する。
  dense caseで二乗項が残るなら適用範囲と上限を明示し、無理にlinear scalingを主張しない。
- indexed solverを小規模brute-force oracleと比較する。物理係数、接触skin、描画radiusを
  変えて候補数増大を見えにくくしない。

## ソース訂正前チェック

- [x] 計測する指標・観点・固定条件を記録。
- [x] 現行コード・実 GPU baseline と数値基準を確認。
- [x] 実装内容、対象、メモリ／limits、回帰条件、採否基準を確定。
- [x] この計画を起点に P1/P2/P3 のソース変更をまとめて実施。
- [x] 結果に基づいて P4 の採否と文書更新を実施。
