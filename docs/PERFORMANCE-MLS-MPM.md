# MLS-MPM パフォーマンス計測・検討指針

作成日: 2026-10-12

対象は `/labs/mpm/kernels` の**定常シミュレーション性能**。
ページロードの Lighthouse score や Core Web Vitals を GPU solver の速度の代理指標にしない。
変更手順と採否基準は [チューニングのロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md)、
材料・物理の論点は [４論点の調査](./RESEARCH-MLS-MPM-MATERIALS.md) を参照する。
2026-10-14にsceneを3,674 active粒子へretuneした。以下の数値は各節の計測日・旧14,600〜20,000粒子sceneに対する
歴史的baselineであり、現行sceneの性能を表さない。retuned sceneの単一`stages`診断結果は末尾に記録するが、
通常batchの反復性能・持続性能・旧sceneとの比較を示すものではない。

## 1. 比較時に固定する条件

| 条件 | 必ず記録するもの |
|---|---|
| ソース | commit、未コミット状態、kernel/layout/solver のハッシュ |
| 実行環境 | OS、ブラウザー、adapter vendor/architecture、fallback の有無、device limits |
| 能力 | `timestamp-query` の有無、実際に要求した feature、GPU 計測精度 |
| 仕事量 | active 粒子数、capacity、格子・halo 節点数、材料比率、１ batch の step 数 |
| 物理 | `dt`、重力・揺動、質量・基準面積・材料係数、初期 `F/C`、初期配置 |
| 描画 | 有無、canvas の pixel size/DPR、描画目標、画面リフレッシュレート |
| 時間 | warmup、サンプル数、simulated time、実時間、試験後半の発熱状態 |
| 干渉 | 他ページ／プロセスの GPU 使用、dev/prod build、profiler の overhead |

同じ初期 seed と同じ step 列で比較する。描画を外した kernel benchmark と、
描画・操作を含めた実時間試験を別に行う。
解像度、粒子数、EOS、密度、`dt`、実効速度の変更を「同じ物理の高速化」と呼ばない。
要求されたsimulation durationを処理できたか、未処理分が明示されているかを確認する。
時間を黙って破棄・clampしてFPSだけを高く保つことはsolverの成功ではない。

## 2. 計測する指標

### 時間・スループット

| 指標 | 意味・測定方法 | 誤読しないこと |
|---|---|---|
| GPU batch interval | timestamp で数値 dispatch 列の開始・終了を測る | queue 完了の JS callback 時間とは別 |
| GPU ms/step | GPU batch interval / 実行 step 数 | timestep や workload を同じにする |
| GPU pass 別時間 | clear、prepare、bucket、P2G、grid、G2P、commit | pass を分ける計測は通常実行と異なる overhead を持つ |
| particle-steps/s | active 粒子数 × 完了した step 数 / 実時間 | submit 換算、rollback、容量を混同しない |
| CPU encode/submit | `stepBatch` 呼び出しの JS 時間、median/p95 | GPU 時間を含まない。`performance.now()` の丸めに注意 |
| queue completion wait | submit 後の `onSubmittedWorkDone` 待ち | GPU 単体の kernel 時間ではなく IPC・他仕事も含む |
| frame interval/FPS | rAF・実描画の median/p95、目標に達したか | 60 FPS ceiling では小さな solver 改善を比較できない |
| input latency | pick/drag/clone の要求から応答・描画まで | 平均 FPS だけでは操作の遅延を説明できない |
| 実効時間倍率 | 完了 step × `dt` / 実時間 | requested speed と分ける |
| simulation-time accounting | requested / committed / remaining simulated seconds | 未完了durationを成功扱い・黙って破棄しない。通常はremainingが0、明示停止時は残量を記録する |

サンプルは median と p95、min/max、数を残す。少数試験の p95 は推定が粗いので、
１回の最高値や細かい小数だけで確信を持たない。
GPU 時間の reciprocal は「描画なしの kernel throughput 上限」であり、
ブラウザー上の実効 particle-steps/s と区別する。

### 構造・メモリ

| 指標 | 検討する観点 |
|---|---|
| dispatch 数、compute pass 数、submit 数 | CPU encoding、pass 切替、GPU barrier。必要な dispatch 順序は維持する |
| active/capacity と GPU byte | unused capacity は計算粒子数ではないがメモリを占有する |
| P2G 構成則評価回数 | 粒子×９ support で重い分解を繰り返していないか |
| stencil entries | 現行は `9 * activeCount`。capacity 分を毎 step 走査しない |
| bucket length の分布・最大 | dense node が１ invocation の長い直列走査になっていないか |
| 空節点率 | full-grid pass に対する active tile/list の追加コストを比較する |
| cache byte と bandwidth | 計算を減らしても読み書き・allocation の増加が得か |
| workgroup size/占有率 | 64 を変える前に shader レジスタ圧・分岐・端末差を測る |
| readback byte/frequency | 粒子全量を通常 frame に入れない。小さい status と区別する |
| startup/clone peak memory | device limits を満たすことと、総 allocation が成功することは別 |

linked-list ソート化、active-grid 化、SoA 化、subgroup、workgroup サイズ変更は比較候補。
今回の結果だけで同時に採用しない。精度・移植性・追加 pass のコストを測ってから進める。

### 数値・挙動の回帰

- CPU/GPU 誤差を位置・速度・`C/F`・塑性履歴・材料情報ごとに測る。
- 小さな完全 scene の全粒子を比較する。既存の３ probe だけで全材料の回帰を判定しない。
- 粒子数・object/material ID・基準面積・質量が step/readback/clone で維持されるか。
- rollback、永続 error status、異常原因の記録、device loss と dispose。
- 複数 step batch と同じ数の１ step 呼び出しが許容誤差内で一致するか。
- clamp 回数、`J` 範囲、体積、運動量、壁・入力の挙動。
  NaN がないことと物理の妥当性を区別する。

P2G の atomic list 順は不定なので bitwise 一致は要求しない。
長い chaotic scene の粒子位置差だけで、短い step の式の正否を判断しない。

## 3. 調査時点のボトルネックと現在の優先測定

1. **GPU: P2G の重複構成則評価（M1 で解消）**。
   [WGSL](../src/lib/mpm/kernels/shaders.ts) の node list 内で同じ粒子の応力を最大９回評価していたが、
   16 byte/粒子の prepare cache により現在は１回/粒子/step。
2. **CPU: batch 内の pass encoding（集約済み）**。
   batch を１ compute pass にまとめた。M1 benchmark では８ step の dispatch が48から56へ増えても、
   pass open/close は１回だった。後続の contact dispatch 数は下記 M4 実装状況を参照。
3. **メモリ: active と capacity の差（継続課題）**。
   旧 Gate 4 scene は20,000 active に対して1,000,000 capacity、M1 後の確保量は約217 MB。
   capacity の削減は今回の計算高速化とは別の検証が必要。
4. **GPU: 三角形 contact field（tile candidate bitsetを実装）**。
   full scanとdense overlapの上限は引き続き `G·B`。512 body以下では16×16 node tileごとの
   conservative bitsetを使い、sparse caseのbody走査を減らす。Gate 4では索引前のfield stage
   0.120710からcandidate generation 0.028062 + field 0.022609 ms/stepへ低下した。
   candidate数counter、brute-force field oracle、dense/mobile受け入れは未完了。
5. **GPU: 小石同士のSATとpair apply（計測済み）**。
   canonical cacheでunordered pairのSAT結果を共有し、active-contact bitsetでsparse applyを短縮する。
   Gate 4の `applyTrianglePairContacts` は0.142011から0.013213 ms/step medianへ低下した。
   512 body超では従来pair scanへfallbackし、pair-gatherの全候補列挙は `O(B²)` のまま。

contact cache passは計測済み。密なnode bucket、実機のrAF/queue gate、操作latency、
scene全体の接触・長時間安定性はまだ定量測定していない。仮説と計測済みbottleneckを混ぜない。

## 4. 実装前の baseline

計測日: 2026-10-12。基準は [調査時の作業ツリー](./RESEARCH-MLS-MPM-MATERIALS.md#sources)。
シミュレーションソースを変更する前に、別の localhost tab で計測した。
既存の共有ページは操作していない。

- Headless Chrome 154、macOS、adapter `apple / metal-3`、fallback **false**。
- `timestamp-query` 対応。adapter の storage buffer 上限は10。
- Gate 4旧粉体シーン、20,000 active / 1,000,000 capacity、220×150セル、`h=10`、
  `dt=1/240`、重力 `(0,90)`、揺動なし、描画なし。
- 64 step warmup、８ step/batch、12 batch サンプル。
- timestamp resolve/map と数値 readback は timed dispatch 列の外側。

| 指標 | median | p95 |
|---|---:|---:|
| GPU step interval | 0.08648 ms/step | 0.15085 ms/step |
| CPU encode/submit | 0.100 ms/batch | 0.100 ms/batch |
| queue 完了待ち | 1.200 ms/batch | 1.700 ms/batch |
| clear | 0.00457 ms/step | 0.00508 ms/step |
| bucket 構築 | 0.00981 ms/step | 0.02574 ms/step |
| P2G | 0.04242 ms/step | 0.04971 ms/step |
| grid update | 0.00797 ms/step | 0.01107 ms/step |
| G2P | 0.01508 ms/step | 0.01701 ms/step |
| commit | 0.00464 ms/step | 0.01136 ms/step |

P2G median は GPU step median の約49%。個々の median は同一サンプルでないので、
合計を厳密な割合にしない。CPU の 0.1 ms はブラウザーの時計分解能の影響を受ける。
短い試験・単一デスクトップ GPU であり、モバイルの性能保証ではない。

数値 baseline:

- Neo-Hookean/Snow/水/油/粉体、45粒子、８ step の CPU/GPU 最大絶対差
  `1.3956e-5`。material/object identity は一致。
- 既存 layout・CPU reference・WebGPU session の39 unit test が成功。
- 現行 Gate 4 の数値 status が正常であることを timing 後に確認。

この性能baselineと以下のA/Bは、水8,000・油6,000・粉体6,000の旧20,000粒子シーンで取得した。
粉体置換直後の途中版は三粒子Neo-Hookean小石1個・14,003粒子であり、
その測定値も現行の50個版との速度比較として解釈しない。

## 5. 実装後に残す比較

### 採用した変更（M1 の A/B。M4 接触追加前）

ロードマップを完成させてから、粒子別の16 byteアフィン運動量cacheと
batchの１compute pass集約を実装した。`dt`・係数・格子・粒子数・容量・壁は変えていない。
新依存なしの [benchmark](../scripts/benchmark-mpm.mjs) と layout／batch／clone テストを追加した。

| 項目 | 変更前 | 変更後 |
|---|---|---|
| 応力・分解の評価 | 最大９回/粒子/step | １回/粒子/step |
| ８ step の通常 compute pass | 48 | １ |
| ８ step の dispatch | 48 | 56（prepareを追加） |
| GPU buffer 確保量 | 201,228,520 byte | 217,228,520 byte |
| P2G storage binding requirement | ５ | ６ |

メモリは**16 MB増える**。capacityを減らして速く見せる変更はしていない。
GPU／CPUの仕事を減らすことと、メモリを減らすことを同じ成果にしない。

### 同条件の暖機後 A/B

最初の短い計測ではGPU step medianが0.074〜0.135 msに変動したため、
単発の最高値を採用せず再計測した。
比較用の旧layout/shader/solverは§4で記録したSHA-256と一致する状態を使用し、
ユーザーの作業ツリーを戻さず、localの比較用moduleとしてロードした。

同じtab内で旧→新を３組、逐次実行した。各試行は600 msの**重力ゼロの静止暖機**を加え、
その後に同じ64 stepの重力90暖機と、12 batch×８ stepの計測を行った。
静止暖機は初期の粒子位置・`F/C`を変えず、GPU時計の立上りの影響を抑える。
実シミュレーションの速度倍率とは無関係な、benchmarkだけの準備である。

| 組 | 旧 GPU median ms/step | 新 GPU median ms/step | 短縮 | 旧 p95 | 新 p95 |
|---|---:|---:|---:|---:|---:|
| １ | 0.08790 | 0.07373 | 16.1% | 0.09675 | 0.07740 |
| ２ | 0.08634 | 0.07302 | 15.4% | 0.12550 | 0.11420 |
| ３ | 0.08542 | 0.07537 | 11.8% | 0.08702 | 0.09079 |

３組のmedianの中央値は **0.08634 → 0.07373 ms/step、約14.6%短縮**。
各組で計画の10% median改善基準を満たした。p95は組３で悪化しており、
tail latencyの普遍的改善は主張しない。CPUの通常８ step encodeは時計の丸めで
medianが0〜0.1 msになり、CPU改善率は算定しない。

kernel attributionの初回再測定では、P2Gは0.04242→0.03123 ms/step、
prepareは0.00659 ms/stepだった。prepareを含む合計は約0.03782 msであり、
P2G単体の短縮率だけで採否を決めていない。
attributionは７ dispatchを分割passで測るので、通常１passのbatch値とは区別する。

### 数値・操作回帰

- 実GPUで全shaderをcompileし、５モデル45粒子・８ stepの最大絶対差は
  `1.3956e-5`で変更前と同水準。
- 同じ８ stepのbatchと単発呼び出しの差は、この試験ではゼロ。
- GPU pick、局所力drag、cloneの全状態保持、容量超過拒否、render submitを確認。
- 過大な変形増分の失敗でerror flag 32を検出し、
  粒子A/B bufferの全wordが最後の正常状態へ戻ることを確認。
- 関連44 unit test、Svelte/TypeScript check（error/warningゼロ）が成功。

### 再計測の実行方法

sourceを配信するdev serverと、専用のlocal Chrome CDP endpointを用意する。
preview serverはTypeScript moduleを配信しないため、このscriptには使わない。

```sh
npm run dev -- --host 127.0.0.1 --port 4181 --strictPort
```

別terminalで、専用profileのChromeを起動する。macOSの例:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/powder-play-mpm-benchmark-profile \
  about:blank
```

CDPを外部公開しない。普段のChrome profileを使わず、起動したprocessだけを終了する。
すでに専用CDP環境がある場合は、新しいChromeを重複起動しない。

```sh
npm run benchmark:mpm -- --profile=batch --output=/tmp/mpm-batch.json
npm run benchmark:mpm -- --profile=stages --output=/tmp/mpm-stages.json
```

`--origin=http://127.0.0.1:PORT`、`--cdp=http://127.0.0.1:PORT`で接続先を指定できる。
scriptは自分のlocal benchmark tabだけを作り、終了時に閉じる。
source hash、device features/limits、暖機、median/p95、数値・操作回帰をJSONで記録する。
sourceが試験中に変わった場合は比較を拒否する。
通常アプリにはtimestamp／benchmark用全量readbackを追加していない。

### 途中版: 小石1個シーン計測（2026-10-12）

粉体版を三粒子の弾性小石1個へ変更した直後、14,003粒子・1,000,000 capacityを
同じ格子 `220×150`, `h=10`, `dt=1/240`, 重力 `(0,90)` で再計測した。
Chrome 154 / Apple Metal 3（fallbackなし）/ `timestamp-query`、暖機64 step、
静止暖機600 ms、12 batch×8 step、描画なし。GPU確保量は217,228,520 byte。
この測定は旧20,000粒子粉体シーンとのA/Bではなく、速度改善率を主張しない。

| profile | GPU median / p95 (ms/step) | CPU encode median / p95 (ms/batch) | queue wait median / p95 (ms/batch) | measured pass/batch |
|---|---:|---:|---:|---:|
| batch | 0.07137 / 0.08673 | 0 / 0.10 | 1.10 / 1.40 | 1 |
| stages | 0.07464 / 0.11440 | 0 / 0.10 | 1.30 / 1.60 | 56 |

stages profileのdispatch attribution medianはclear 0.00471、prepare 0.00549、
bucket構築 0.00813、P2G 0.02768、grid update 0.00834、G2P 0.01268、
commit 0.00401 ms/step。attributionはbatchと異なるpass構成であり、合計をbatch値と同一視しない。
Chrome CPU時計の分解能のためCPU medianの0 msは実処理時間ゼロを意味しない。

この途中版ページの実GPU初回比較は各材料の中央粒子を含む局所サンプルで最大絶対誤差
`1.28e-5`、許容範囲内。benchmarkの別個の45粒子・５モデル／８ step回帰では
最大絶対誤差 `1.3956e-5`、batch/単発差0、pick・drag・clone・render・rollback・
永続数値statusを確認した。これらは小石の長時間弾性復元、衝突、全14,003粒子の
全量CPU parityを検証するものではない。

### ベースライン: 小石200個シーン計測（11-unit版、2026-10-12）

水8,000・油6,000・三粒子小石200個の14,600 active / 1,000,000 capacityを計測した。
各三角形は辺長11 world units、20×10配置・中心間隔20 world units、個別 `objectId` 3〜202。
この測定後に辺長33 / spacing40の1.5%-width版へ変更し、さらに現行の辺長66 /
spacing68・14列×15行版へ拡大・再配置した。以下の結果はどちらの後続版の性能や弾性挙動も表さない。
格子 `220×150`, `h=10`, `dt=1/240`, 重力 `(0,90)`。Chrome 154 / Apple Metal 3
（fallbackなし）/ `timestamp-query`、暖機64 step、静止暖機600 ms、12 batch×8 step、描画なし。
GPU確保量は217,228,520 byte。この単発profileは旧20,000粒 powder版や14,003粒途中版との
paired A/Bではなく、差分の性能改善率は主張しない。

| profile | GPU median / p95 (ms/step) | CPU encode median / p95 (ms/batch) | queue wait median / p95 (ms/batch) | measured pass/batch |
|---|---:|---:|---:|---:|
| batch | 0.06726 / 0.08440 | 0 / 0 | 1.00 / 1.30 | 1 |
| stages | 0.07526 / 0.07767 | 0 / 0 | 1.10 / 1.50 | 56 |

stages profileのdispatch attribution medianはclear 0.00467、prepare 0.00563、
bucket構築 0.00838、P2G 0.02819、grid update 0.00884、G2P 0.01261、
commit 0.00400 ms/step。attributionはbatchと異なるpass構成であり、通常batchのGPU値とは比較しない。
CPU時計の分解能のため、encode median/p95の0 msは実処理時間ゼロを意味しない。

このベースライン版ページの初回CPU/GPU比較は小石群中央と水・油の局所サンプルで最大絶対誤差
`1.28e-5`、許容範囲内。別個の45粒子・５モデル／８ step回帰で最大絶対誤差
`1.3956e-5`、batch/単発差0、pick・drag・clone・render・rollback・永続数値statusを確認した。
これらは長時間の小石同士の接触・弾性復元や全14,600粒子の全量CPU parityを証明しない。

**未確認**: mobile・他GPU・高capacityで全粒子を動かす場合の性能、
５分以上の持続性能、dense nodeの分布、powder/界面の物理校正、tail latencyの改善。
この短い描画なし試験を、30/60 FPS達成や物理Gate合格へ読み替えない。

### 66-unit・ゼロ初速版シーン計測（2026-10-12）

水8,000・油6,000・三粒子小石200個（各辺66 world units、枠幅3%）の
14,600 active / 1,000,000 capacityを計測した。小石は14列×15行（最終行4個）、
中心間隔68 world units、個別 `objectId` 3〜202。格子 `220×150`, `h=10`,
`dt=1/240`, 重力 `(0,90)`。Chrome 154 / Apple Metal 3（fallbackなし）/
`timestamp-query`、初速ランダム化前（全粒子の初速0）の状態で暖機64 step、
静止暖機600 ms、12 batch×8 step、描画なし。
GPU確保量は217,228,520 byte。この測定は11-unit・33-unitの配置とのpaired A/Bではなく、
相対的な性能改善率は主張しない。

| profile | GPU median / p95 (ms/step) | CPU normal encode median / p95 (ms/batch) | queue wait median / p95 (ms/batch) | measured pass/batch |
|---|---:|---:|---:|---:|
| batch | 0.06448 / 0.06644 | 0 / 0 | 0.90 / 1.20 | 1 |
| stages | 0.07594 / 0.07928 | 0 / 0 | 1.10 / 1.20 | 56 |

stages profileのdispatch attribution medianはclear 0.00472、prepare 0.00558、
bucket構築 0.00865、P2G 0.02825、grid update 0.00892、G2P 0.01275、
commit 0.00415 ms/step。attributionはbatchと異なるpass構成であり、通常batch値と直接比較しない。
CPU時計の分解能のため、encode median/p95の0 msは実処理時間ゼロを意味しない。

このゼロ初速版ページの初回CPU/GPU比較は最大絶対誤差
`1.28e-5`、許容範囲内。別個の45粒子・５モデル／８ step回帰で最大絶対誤差
`1.3956e-5`、batch/単発差0、pick・drag・clone・render・rollback・永続数値statusを確認した。
これらは長時間弾性復元、頂点間の一体変形、明示的接触を検証するものではない。

### 接触追加前: 66-unit・低ランダム初速シーン計測（2026-10-12）

前項と同じ14,600 active / 1,000,000 capacityの配置・格子・時間刻み・重力を使い、
水・油の速度成分を±12、小石の各頂点の速度成分を±24 world units/sから独立に抽選した。
シーン生成seedは `0x4d504d34` に固定し、計測を再現可能にした。Chrome 154 / Apple Metal 3
（fallbackなし）/ `timestamp-query`、暖機64 step、静止暖機600 ms、12 batch×8 step、
描画なし。GPU確保量は217,228,520 byte。この記録は現在の±60/±120初速と
水–油／小石接触 dispatch を含まない。M1 の性能基準としてのみ残す。

| profile | GPU median / p95 (ms/step) | CPU normal encode median / p95 (ms/batch) | queue wait median / p95 (ms/batch) | measured pass/batch |
|---|---:|---:|---:|---:|
| batch | 0.06569 / 0.07025 | 0 / 0.10 | 0.90 / 1.00 | 1 |
| stages | 0.09391 / 0.11052 | 0 / 0 | 1.20 / 1.50 | 56 |

stages profileのdispatch attribution medianはclear 0.00571、prepare 0.00690、
bucket構築 0.01158、P2G 0.03598、grid update 0.00977、G2P 0.01672、
commit 0.00502 ms/step。attributionはbatchと異なるpass構成であり、通常batch値と直接比較しない。
CPU時計の分解能のため、encode median/p95の0 msは実処理時間ゼロを意味しない。

当時のランダム初速で再読み込みしたページの初回CPU/GPU比較は最大絶対誤差 `2.48e-5`、
当時の許容範囲内。GPU benchmark内の45粒子・５モデル／８ step回帰は最大絶対誤差
`1.3956e-5`、batch/単発差0、pick・drag・clone・render・rollback・永続数値statusを確認した。
この短時間試験は長時間弾性復元、頂点間の一体変形、明示的接触を検証するものではない。

### 2026-10-12時点: 14,600粒子・66-unit・接触prototype・高ランダム初速

接触・辺長制約を含む当時のsceneを、同じ固定条件で追加計測した。シーン seed は
`0x4d504d34`、水・油±60、小石頂点±120 world units/s、14,600 active /
1,000,000 capacity、220×150 cells (`h=10`)、`dt=1/240`、重力 `(0,90)`、
描画なし。Chrome 154 / Apple Metal 3 (fallbackなし) / `timestamp-query`、
600 ms の重力ゼロ暖機、64 step の重力暖機、12 batch×8 step。
測定した source hash は layout `509c84abf9ac089369e8ed18f92a0dd0fea754695f3ad35124a021e93a9880ca`、
shaders `1e9f3de53af6ad0db3602b2dffce1cdc2a87671ab57f62efead3d486c8a535e4`、
solver `151f3e05ebeb6233d9915b658ffe91b4908466e78fab2ff4c5d2d6e3584ad893`。
GPU確保量は `221,564,392 byte`。storage buffer binding上限は8、現在の要求数以内。
この14,600粒子の計測には現行ラボの96粒子・249 bondの破断solidを含まない。
したがって2026-10-13の14,696粒子計測との性能差を直接比較しない。

| 指標 | batch profile median / p95 | stages profile median / p95 |
|---|---:|---:|
| GPU step interval (ms/step) | 0.37094 / 0.37371 | 0.41262 / 0.42210 |
| CPU encode/submit (ms/batch) | 0 / 0.10 | 0 / 0 |
| queue completion wait (ms/batch) | 3.30 / 3.40 | 3.90 / 3.90 |
| 計測pass数 / 8-step batch | 1 | 80 |

CPU encode は `performance.now()` の時計分解能により0 msへ丸められるため、処理がゼロという意味ではない。
stages profile はdispatchごとにcompute passを分ける診断経路で、通常batchよりGPU時間が増える。
次表のstage中央値は独立したサンプルの統計値であり、合計をGPU step中央値と厳密に一致させない。

| stage | median (ms/step) | p95 (ms/step) |
|---|---:|---:|
| clearGrid | 0.00806 | 0.00819 |
| prepareParticles | 0.00697 | 0.00704 |
| buildBuckets | 0.01064 | 0.01080 |
| P2G | 0.03223 | 0.03323 |
| buildTriangleField | 0.13918 | 0.13975 |
| updateGrid | 0.01653 | 0.01681 |
| G2P | 0.02277 | 0.02316 |
| applyTriangleContactImpulses | 0.03664 | 0.03715 |
| solveBondedTriangles | 0.10010 | 0.10094 |
| commitParticles | 0.00517 | 0.00556 |

最も大きい個別stageは全grid node×200三角形を候補走査する `buildTriangleField` と、
小石同士を検査する `solveBondedTriangles`。AABB早期除外、contact impulse gather 範囲の縮小、
field broad-phase の平方距離化を加えた。変更前後各１回の参考計測ではbatch medianが
0.47325から0.37094 ms/step、SAT passが0.19493から0.10010 ms/stepへ変化したが、
反復A/Bではないため改善率として確定しない。field走査のGPU spatial broad-phase、
描画込みの実効時間倍率、他端末・長時間の性能は後続検討である。

benchmark内の45粒子・５モデル／８ step回帰は最大絶対誤差`1.3955e-5`、
batch/単発差0、pick・drag・clone・render・rollback・永続numerical statusを確認した。
実画面の初回比較は各材料の局所sampleで、速度はCPU倍精度／GPU単精度差に合わせた
absolute-plus-relative基準を用いる。１回のrandom-seed reloadでは最大絶対誤差`1.10e-2`で基準内。
固定した５ seed の同じ局所CPU/GPU比較もすべて合格し、raw最大絶対差は`1.9063e-2`。
三角形のGPU clone/readbackと、clone後の追加step・numerical statusも確認した。
これは接触のscene-scale妥当性、長時間の形状保持、運動量残差の証明ではない。

### 2026-10-13計測snapshot: 14,696粒子・破断solid込み（adaptive CFL変更前）

`/labs/mpm/kernels` の旧 `createMpmGate4Scene(0x4d504d34)` を使い、水8,000・油6,000・
三粒子小石200個（600粒子）・12×8 nodeの破断solid（96粒子、249 bond）を含む
14,696 active / 1,000,000 capacityを計測した。220×150 cells (`h=10`)、`dt=1/240`、
重力 `(0,90)`、描画なし。Headless Chrome 154 / macOS / Apple Metal 3
（fallbackなし）/ `timestamp-query`。各runで600 msの重力ゼロ暖機、64 stepの重力暖機、
12 batch×8 stepを計測した。旧sceneのGPU buffer確保量は `221,626,976 byte`。

この計測・下記のS3/S4 fixture・source hashは、adaptive CFL microstep/recoveryと
2026-10-14のscene retune前のsnapshotである。この計測値を現行sourceの性能・物理結果として
引用しない。retuned sceneの診断計測・現行source hashは末尾に記録する。

`batch` profileを独立に5回実行した。各runの12サンプルの中央値とp95は次のとおり。

| run | GPU median / p95 (ms/step) | CPU encode median / p95 (ms/batch) | queue wait median / p95 (ms/batch) | pass / 8-step batch |
|---|---:|---:|---:|---:|
| 1 | 0.41254 / 0.45756 | 0 / 0.10 | 3.80 / 4.20 | 1 |
| 2 | 0.40143 / 0.44743 | 0 / 0.10 | 3.60 / 4.00 | 1 |
| 3 | 0.40109 / 0.43035 | 0 / 0.10 | 3.80 / 3.90 | 1 |
| 4 | 0.36103 / 0.37224 | 0 / 0.10 | 3.30 / 3.40 | 1 |
| 5 | 0.39817 / 0.41114 | 0 / 0.10 | 3.60 / 3.90 | 1 |

runごとのGPU medianの中央値は `0.40109 ms/step`、runごとのp95の中央値は
`0.43035 ms/step`。第4runは他runより速い側の値で、単一結果を基準にせず反復値で扱う。
stationary warmupは600 msの実時間なので、runごとのwarmup step数とscene状態は完全一致しない。
CPU時計の分解能でencode medianは0 msに丸められるため、
CPU時間がゼロという意味ではない。queue waitはGPU timestampではなく、
GPU実行時間の代用にはしない。

別の `stages` profileは診断のためdispatchごとにcompute passを分割し、
8-step batchあたり80 passで計測した。通常batchの1 pass構成と時間を直接比較しない。
診断profileのGPU step intervalはmedian `0.45286` / p95 `0.47742 ms`。

| stage | median / p95 (ms/step) |
|---|---:|
| clearGrid | 0.00889 / 0.01341 |
| prepareParticles | 0.00794 / 0.01452 |
| buildBuckets | 0.01105 / 0.01844 |
| P2G | 0.03353 / 0.04076 |
| buildTriangleField | 0.13882 / 0.14395 |
| updateGrid | 0.01620 / 0.01815 |
| G2P | 0.02578 / 0.05711 |
| applyTriangleContactImpulses | 0.03541 / 0.03703 |
| solveBondedTriangles | 0.09796 / 0.10159 |
| commitParticles | 0.00446 / 0.02698 |

最も時間の大きいstageは `buildTriangleField` と `solveBondedTriangles`。
当時の旧sceneで性能改善に着手する場合の候補だが、接触のscene-scale物理基準を先に固定し、
最適化で接触結果が変わらないことを検証する。

### 接触候補の計算量と規模リスク

現行のtriangle collider数を `B`、halo込みgrid node数を `G` とすると、
full-scan時の `buildTriangleField` 上限は `G·B`。現行の512-body以下の経路では
16×16-node tileごとにbody候補bitsetを作り、field passがtile候補だけを調べる。
旧Gate 4の140 tile × 200 bodyではcandidate generationが最大28,000 body/tile checks/substep。
node-level candidate count自体はまだinstrumentationしていないため、dense worst caseは引き続き
`G·B`。`gatherTrianglePairContacts` はcanonical unordered pairを `B·(B−1)/2` 個調べ、
`applyTrianglePairContacts` はsparse caseでactive pairだけを、dense caseでは最大
`B·(B−1)` 個のdirected contactを処理する。512 body超は従来full scanへfallbackする。
旧sceneの `G=34,119`、`B=200` ではfull-scan field上限が約6.82M checks、SAT candidateが
19,900 unordered pairs/substep、dense directed contacts上限が39,800である。現行の `B=50` では
full-scan上限約1.71M、unordered SAT候補1,225、dense directed contact上限2,450となるが、
tile候補生成とGPU時間は末尾のretuned scene `stages` profileで1回計測した。node-level candidate
countはinstrumentしていないため、dense worst caseと`G·B`上限は引き続き残る。

| 三角形collider数 `B` | full-scan field上限 `G·B` | unordered SAT候補 `B·(B−1)/2` | dense apply上限 `B·(B−1)` |
|---:|---:|---:|---:|
| 200 | 6.82M | 19,900 | 39,800 |
| 400 | 13.65M | 79,800 | 159,600 |
| 1,000 | 34.12M | 499,500 | 999,000 |
| 5,000 | 170.60M | 12.4975M | 24.995M |
| 10,000 | 341.19M | 49.995M | 99.99M |

この表は演算時間ではなく、1 physics substep当たりの候補チェック上限である。
CFLでsubstep数が増えれば接触処理も比例して繰り返される。tile bitsetとactive pair maskは
sparse caseで候補を絞るが、密集sceneではfield `O(G·B)`、pair gather `O(B²)` のworst caseが残る。

2026-10-13の旧66-unit scene診断では `buildTriangleField` が0.13882 ms/step、
`solveBondedTriangles` が0.09796 ms/stepだった。これは現行solverへ置換される前のdispatch別profileであり、
通常batchや別GPUへの時間予測には使わない。現行sourceのfield/SAT計測値は下記A/B記録を参照する。
従って1M particle buffer capacityを、そのまま実用的なcollider数の保証と解釈しない。
具体的なbroad-phase方式とscale acceptance gateは
[性能チューニングロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md)のP5に記録する。

この実測値からの単純なGPU仕事量の導出では、既定の2×は480 step/s、
GPU step時間が約0.193秒/実時間1秒に相当する。一方32×は7,680 step/sを要し、
同じmedianではGPU計算だけで約3.080秒/実時間1秒となる。描画・CPU・scheduler・
発熱を含まない概算であり、実効時間倍率の実測ではない。32×を維持するなら、
少なくとも計算だけで約`0.13021 ms/step`以下が必要になる。

benchmark中に旧14,696粒子sceneのGPU実行と数値status確認は通過した。
別途45粒子・５モデル・８ stepのCPU/GPU回帰は最大絶対差`1.3955e-5`、
batch/単発差0。pick、drag、clone、render、rollback、永続numerical statusも確認した。
同じ第5回実行では、小規模fixtureの物理受け入れも行った。水・油2粒子の8 stepすべてで、
CPU/GPU最大絶対差`3.8653e-5`、相別質量は水・油とも`0.25`を維持、最大運動量残差
`1.1921e-7`、初回接近速度の低下と全stepでの非交差を確認した。Neo-Hookean 4粒子・3 bondでは、
failure strain `0.1`に対する入力strain `0.05 / 0.11 / 0`のうちbond index 1だけが破断し、
閾値未満bondのdamageは`0.499992`だった。2 step後の最大粒子状態誤差は`7.63e-7`、
粒子・bond damage誤差はそれぞれ`7.63e-6`以下。別fixtureでは、初期状態で破断済みの
bondがrest length時に2 step後もdamage 1 / brokenを保持した。破断fixtureの質量は4、
運動量残差は0。CPU/GPU状態・damage比較の許容差`1e-3`は既存回帰と同じ数値許容値であり、
製品品質の貫入・形状閾値ではない。

これらは制御された小規模ケースのGPU確認であり、scene全体のCPU readbackは行っていない。
旧14,696粒子sceneの界面貫入・反復接触・相別質量/運動量、実sceneの破断pattern、
長時間形状保持は未検証で、S3/S4物理受け入れゲートの合格とはしない。
layout / shaders / solverのSHA-256はそれぞれ
`1655a7f65f4dc37bc57b673f1e4b1661f70f90b2da907cb56705149f7e5559d5`、
`5d5dc93fd7a92e1dffdf07f31309822067dc305de72c95590741c6d3d8d6af57`、
`e98c0cd5f2f340bc016f9c889addfdd2d378820cd6d6fbf0df8666dd7a79a7c8`。

### 2026-10-13: canonical triangle-pair cache A/B

旧14,696粒子sceneで、contact cache導入直前の作業treeとcanonical pair cache導入後を
同じHeadless Chrome 155 / macOS / Apple Metal 3 adapterで比較した。fallbackなし、
`timestamp-query`あり。220×150 cells、cellSize 10、`dt=1/240`、重力 `(0,90)`、
描画なし、14,696 active / 1,000,000 capacity。各runは64 step warmup後に600 msの
無重力stationary warmupを行い、12個の8-step batchを測定した。stationary warmupのstep数は
runごとに異なるため、median-of-runsで比較する。通常batchは固定 `dt` の `stepBatch()` を直接測り、
interactive pageのCFL scheduler、描画、end-to-end frame rateは含まない。

| 指標 | cache導入前 | canonical cache後 |
|---|---:|---:|
| 5 runのGPU median (ms/step) | 0.688128, 0.663552, 0.688128, 0.679936, 0.679936 | 0.557056 × 5 |
| run medianの中央値 (ms/step) | 0.679936 | 0.557056 |
| run p95の中央値 (ms/step) | 0.696320 | 0.565248 |
| GPU buffer確保量 | 226,960,308 byte | 231,154,612 byte (+4,194,304 byte) |
| 通常batchのcompute pass / 8-step batch | 1 | 1 |

通常batch medianは **18.1%**、run p95 medianは **18.8%**短縮した。診断用 `stages`
profileはdispatchごとにpassを分けるため通常batch時間と直接比較しない。そのprofileでは
8-step当たり96から104 passへ増え、step interval medianは0.745472から0.614400 ms。
contact関連stageの合計は、gather 0.163840 + apply 0.188416 = 0.352256 ms/stepから、
state gather 0.008192 + canonical pair gather 0.131072 + apply 0.147456 =
0.286720 ms/stepへ減った。`buildTriangleField`は変更前0.155648、変更後0.139264 ms/stepで、
全node×body探索は残る。

cacheは最大512 active bonded bodies、4 MiBのbounded storage bufferを使う。接触式・skin・
impulse relaxationは変えず、各unordered pairのSAT結果を一度だけ計算し、cacheへ保存する。
pair passはcontact countを両bodyへatomic加算し、後続passが同じcontact recordを反対向きの
normalへ変換して再利用する。512 bodies超または初期triangle bodyなしではcacheを使わず、
従来のdirect pair scanへfallbackする。cache利用時のstorage binding要件は7 buffers/stage。
本sceneの追加確保量は4 MiBだった。

GPU acceptanceでは、三角形3 body / 9粒子の重なりfixtureを4 step実行し、CPU参照との差
最大0.000380862、運動量残差0.000018731、kinetic energy 1.2から0.080418への低下を確認した。
既存の5材料45粒子/8 step parity最大差は0.000013955、水・油contactとfracture fixtureも通過した。
これは小規模fixtureの受け入れであり、scene全体のCPU比較、mobile/長時間性能や
全sceneのcontact品質を保証しない。旧200-pebbleのSAT narrow phaseは削減したがpair candidate列挙と
triangle fieldの `O(G·B)` 探索は残る。spatial broad-phaseは
[ロードマップP5](./ROADMAP-MLS-MPM-PERFORMANCE.md)で別途扱う。

### 2026-10-14: tile field candidates and pair traversal

Headless Chrome 154 / macOS / Apple Metal 3、fallbackなし、`timestamp-query`あり。
旧Gate 4の14,696 active / 1,000,000 capacity、220×150 cells、cellSize 10、`dt=1/240`、
重力 `(0,90)`、描画なし。各runは64 warmup steps、600 ms stationary warmup後に
12×8-step batchesを測定した。`profile=batch`は通常の1 compute pass/batch。
初期値は既存のcanonical pair/gather最適化後、zero-degree skip・workgroup field cache・
active-pair bitset・tile field indexを加える前の測定である。

| 指標 | 追加最適化前 | 最終worktree |
|---|---:|---:|
| `profile=batch` GPU median / p95 (ms/step) | 0.443565 / 0.445080 | 0.231175 / 0.258003 |
| `applyTrianglePairContacts` median (ms/step) | 0.142011 | 0.013213 |
| field-related stages median (ms/step) | `buildTriangleField` 0.139653 | candidates 0.028062 + field 0.022609 |
| allocated GPU buffers | 231,154,612 byte | 231,196,340 byte |

通常batchのmedianは約47.9%低下した。field候補生成＋fieldのstage median合計は
索引前のfield stageより約63.7%低下し、pair applyは約90.7%低下した。
診断用 `profile=stages` はpassを104から112へ分割したため、profile合計同士は比較しない。
stageごとのp95には実行環境のばらつきがあり、batch p95も反復run間で一貫していない。

tile indexは16×16 node tileあたり512 body bitsetを使用し、Gate 4では8,960 bytesを追加する。
512 body超はfull field scanへfallbackする。pair cacheは最大4 MiB、pair-candidate bitsetは
最大32 KiB。小規模water/oil、triangle-pair、fracture acceptanceと188 unit tests、
`npm run check`、`npm run build`は通過した。ただし全field nodeのbrute-force differential oracle、
scene-scale接触品質、mobile GPU、CFL/readback/render込みのruntime queue latencyは未検証。

ベンチマーク前のbase revisionは `c44ba71`（clean worktree）、最適化後は同revision上の変更済みworktree。
layout / shaders / solverのSHA-256は変更前
`8d25eb5f47867b544094b0c5b4d6ab244da0cee480169d59fccacb2eb9d63129` /
`ac2cd787923d047d4b7517f29c500b00771f4b08395d8c6a88c8f6617581f92e` /
`4271f5d909c7d4c190683e36590ae8c1017a3b44c5236220d77663966666b3e7`、
変更後
`4629aba6dd57eafddcb65347f5fcbc1e1d92758e7645cfa934c93086bbec2ed5` /
`f7d344a157d2e7bb4b86a3939265d95f2e6fd0752d371636ff19b329907bb846` /
`f11c6e55b520ca314060b2c1a5bf0b571264881c4d7a3a9fe220237171083b30`。

### 2026-10-14: retuned 3,674-particle scene `stages` smoke (before EOS/contact strengthening)

現行Gate 4 scene（水2,000・油1,500・50個の三粒子小石・24粒子破断solid）を
1,000,000 capacity、220×150 cells、cellSize 10、`dt=1/240`、重力 `(0,90)`、
描画なしで1回計測した。Headless Chrome 155 / macOS / Apple Metal 3
（fallbackなし、`timestamp-query`）。64 warmup steps後に600 msのstationary warmup、
12 batch×8 stepを測定した。`profile=stages`はpassを112へ分けるため、
通常の`profile=batch`や旧14,696 particle測定と時間を直接比較しない。

| 指標 | 結果 |
|---|---:|
| active particles / capacity | 3,674 / 1,000,000 |
| allocated GPU buffers | 231,188,500 byte |
| GPU step median / p95 / max | 0.229376 / 0.286720 / 0.294912 ms |
| `buildTriangleFieldCandidates` median | 0.016384 ms/step |
| `buildTriangleField` median | 0.016384 ms/step |
| `applyTrianglePairContacts` median | 0.073728 ms/step |
| queue wait median / p95 | 2.6 / 3.0 ms/batch |

50 body・140 tileではcandidate bitsetが2 word/tile、合計1,120 byteとなる。
GPU regressionは45粒子・5材料・8 stepで最大絶対差0.0000139553、batch/single差0。
小規模water/oil contact、triangle pair contact、fracture fixtureも通過した。
ブラウザーsmokeではscene 3,674粒子の起動、canvas描画、揺れボタンの停止0→開始±240→停止0、
console errorなしを確認した。

これは`stages` profileの単一runで、GPU buffer capacityは1,000,000のまま。
反復`batch`計測、描画・CFL/readback込みのruntime latency、mobile GPU、scene全体の接触・
破断品質、速度上限付近の長時間挙動を評価した結果ではない。source hashes:
layout `57b00a93775a75dbfb71739f7331701d27a1898088b3c84107096932aaea50e7`,
shaders `29cd62ffa0303ebf72f38b7376d226e266c5303bb41897096874382bf23e0dad`,
solver `882eb942baa834c032dea10232c4d0a172016c7e2ce6e9e0eb2e9dc116762b0c`.

### 2026-10-14: fourfold liquid EOS and approach-scaled contact (`batch`)

After increasing the Gate 4 water/oil EOS coefficients to 160,000 / 120,000 and
adding the overlap- and approach-scaled normal contact impulse, the same
3,674-particle scene was measured on Headless Chrome 154 / Apple Metal 3
(hardware adapter, `timestamp-query`). Configuration remained capacity
1,000,000, 220×150 cells, cellSize 10, `dt=1/240`, gravity `(0,90)`, and no
rendering. The measured batch used one compute pass, 64 warmup steps followed
by 600 ms stationary warmup, then 12 batches of 8 steps.

| 指標 | 結果 |
|---|---:|
| allocated GPU buffers | 231,188,500 byte |
| GPU step median / p95 / max | 0.216618 / 0.256122 / 0.270127 ms |
| queue wait median / p95 | 2.4 / 2.6 ms/batch |
| CPU/GPU parity, 45 particles / 5 materials / 8 steps | max absolute error 0.0000139553 |
| batch-vs-single error | 0 |

The separate `stages` profile for this source measured 0.267246 ms median and
0.281220 ms p95; it splits the compute work into 112 passes and must not be
compared directly to the one-pass `batch` profile. The high-speed GPU phase
contact fixture used water/oil velocity components `+64` / `-64` world-units/s,
`dt=0.001`, and 8 steps. It had no particle crossing, zero phase-mass error,
maximum momentum residual `2.861e-6`, and CPU/GPU maximum absolute error
`0.000051878`.

This is one short, no-render run on one adapter. It does not establish
adaptive-CFL throughput, rendered or long-duration performance, or
scene-scale immiscibility/contact accuracy. Source hashes:
layout `57b00a93775a75dbfb71739f7331701d27a1898088b3c84107096932aaea50e7`,
shaders `39b103fe8e97cf8633b7359ad2d99c666c15ba0924a1c0424583bbd13fbd1317`,
solver `52ae85b48e1a08f4f9d54c5aa7e6a5ecf9b72aa3129fe13fbb7fea7ef285fcfa`.

### 2026-10-14: liquid reference area increased from 100 to 400

The next tuning kept the 3,674-particle scene, liquid density, particle count,
spacing, and initial positions, but multiplied each water/oil reference area
and corresponding particle mass by four. Total liquid reference mass is now
16,000 for water and 9,600 for oil. The scene particle-marker radius grows by
`sqrt(4) = 2`; solid parameters and the EOS/contact coefficients are unchanged.

The regular `npm run benchmark:mpm -- --profile=batch` run did not yield a
performance result. This harness directly advances the seeded Gate 4 scene at
fixed `dt=1/240 s`; it does not use the interactive lab's adaptive-CFL
scheduler/retry path. After the stationary warmup, the fixed-step path raised
`deformation update exceeded the adaptive substep limit at particle 509` and
rolled back. The earlier 0.216618 ms median / 0.256122 ms p95 measured the
area-100 configuration and must not be attributed to the current area-400
scene.

Separately, the interactive adaptive-CFL lab ran for 15 seconds in free-surface
mode in dedicated Headless Chrome / Apple Metal 3 with 3,674 particles, stayed
active, and reported no runtime/console errors. The particle-marker mode was
also switched on and confirmed separately, with larger markers visible and no
console errors. This confirms only a short smoke, not performance,
time-backlog bounds, or scene-scale physical quality. See the earlier area-400 verification record in
`docs/HANDOFF-MLS-MPM-2026-10-13.md`.

### 2026-10-14: MPM-only 1,600 reference area and 6,000-unit domain

The interactive MPM scene now uses a dedicated `220 × 600` grid with cell size
10 (`2,200 × 6,000` world units). Liquid particle count and spacing remain
unchanged, while water/oil reference area increases from 400 to 1,600 per
particle. At the unchanged densities 0.02 / 0.016, this gives 64,000 / 38,400
total mass and 3.2M / 2.4M reference area. The scene remains at 3,674 particles;
all initial objects move upward by 4,500 units to preserve the former bottom
clearance. The LiquidFun comparison scene retains its original grid and
configuration.

The combined liquid reference area is 5.6M, or 42.4% of the 13.2M domain area.
If the former mean `J ≈ 1.03` remains representative, the projected liquid
area is 5.77M (43.7% of the domain). With the solver's per-particle
`J = 0.5–2` clamp, the corresponding mathematical envelope is 2.8M–11.2M
(21.2%–84.8% of the domain); these are sizing calculations, not measured
global occupancy. At a fixed horizontal footprint, quadrupling reference area
would yield roughly four times the settled depth, but the actual water/oil
depths depend on their spreading footprint, contact response, and evolving
`J`. No scene-scale settled-depth measurement is available yet. EOS coefficients
remain 160,000 / 120,000; they control compression recovery, not the reference
volume itself.

The high-speed recovery path also changed: water/oil approach velocity is no
longer amplified at contact, and the fixed four-retry ceiling was removed.
Initial processing is limited to 512 microsteps; adaptive retries share a
1,024-microstep batch budget including failed work. If that budget cannot cover
the requested interval, unprocessed time is retained as backlog instead of
discarded. The browser's initial macrostep completed using 2 retries and 25
attempted microsteps. The integrated browser reported a hidden document
visibility state, so no sustained GPU run, frame timing, or rendered-scene
verification was obtained.

The targeted MPM tests passed (103 tests across 6 files), and the full unit
suite passed (208 tests across 22 files). `npm run check` reported no Svelte or
TypeScript diagnostics; `npm run build` and `git diff --check` also succeeded.
The new grid's allocation is
estimated at about 243.63 MB for the fixed one-million-particle capacity,
including the roughly 12.44 MB increase in grid-dependent buffers. The old
0.217 ms median / 0.256 ms p95 benchmark belongs to the prior area-100 setup
and is not a performance result for this configuration.

### 2026-10-15: CPU grid and triangle-field workspace reuse

The CPU reference solver now uses dense integer-indexed storage for grid nodes
and triangle-contact fields on grids of at most one million padded nodes, with
the sparse `Map` path retained for larger grids. `MpmCpuReferenceSimulation`
keeps an `MpmSolverWorkspace` for its lifetime: generation stamps reset the
active entries between steps, while grid-node and triangle-contact objects are
reused. Disposing the simulation releases both pools.

Two repeated 10-second Vitest benchmarks (2-second warmup each) used the fixed
Gate 4 workload: 3,674 particles, the same initial state and `dt=1/240`, no
rendering. The no-workspace and persistent-workspace paths measured:

| Run | No workspace mean | Persistent workspace mean | Relative speed |
|---|---:|---:|---:|
| 1 | 5.828 ms/step | 4.643 ms/step | 1.26× |
| 2 | 6.031 ms/step | 4.701 ms/step | 1.28× |

Against the earlier 8.663 ms/step pre-optimization reference on this fixed
workload, the final workspace path is about 46% lower in step latency (about
1.85× throughput). This is a fixed-state microbenchmark, not an interactive
frame-rate or sustained-scene guarantee.

A forced-GC Node heap sample after 20 fixed-state steps measured 11.42 MB after
the non-workspace path and 19.66 MB with the retained workspace, an increase of
about 8.24 MB. After `dispose()`, the heap returned to 11.43 MB. This is the
Gate 4 solver's Node/Vitest heap measurement, not total browser or GPU memory.

## 6. 参照

- [WebGPU: usage scopes](https://www.w3.org/TR/webgpu/#usage-scopes) —
  compute の dispatch はそれぞれ独立した usage scope。
- [WebGPU: compute pass](https://www.w3.org/TR/webgpu/#compute-pass-encoder) —
  dispatch の順序と resource validation を維持したまま１ pass にまとめる設計の根拠。
- [WebGPU: timestamp writes](https://www.w3.org/TR/webgpu/#dictdef-gpucomputepasstimestampwrites) —
  対応 feature が必要。非対応時は queue 時間を GPU 時間と偽装しない。
- [kernel の現在の制約](../src/lib/mpm/kernels/README.md)。
