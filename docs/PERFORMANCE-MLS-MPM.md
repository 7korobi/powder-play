# MLS-MPM パフォーマンス計測・検討指針

作成日: 2026-10-12

対象は `/labs/mpm/kernels` の**定常シミュレーション性能**。
ページロードの Lighthouse score や Core Web Vitals を GPU solver の速度の代理指標にしない。
変更手順と採否基準は [チューニングのロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md)、
材料・物理の論点は [４論点の調査](./RESEARCH-MLS-MPM-MATERIALS.md) を参照する。

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
１×を維持できずシミュレーション時間を破棄している場合、FPS が高くても solver の成功ではない。

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
| discarded time | budget 上限で破棄した simulated seconds | 高 FPS のために失った物理時間を隠さない |

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

## 3. 調査から確認できたボトルネック

1. **GPU: P2G の重複計算**。
   [WGSL](../src/lib/mpm/kernels/shaders.ts) の node list 内で
   同じ粒子の構成則を最大９回評価する。Snow／粉体の分解は特に重い。
2. **CPU: batch 内の pass encoding**。
   [GPU solver](../src/lib/mpm/kernels/solver.ts) は６ dispatch ごとに６ pass を開閉する。
   ８ step なら48 pass、128 step なら768 pass を encode する。
3. **メモリ: active と capacity の差**。
   [Gate 4 scene](../src/lib/mpm/scenes/gate4.ts) は20,000 active に対して1,000,000 capacity。
   約201 MBを確保している。これは今回の kernel 計算削減とは別の容量設定課題。

bucket の dense-node 偏り、GPU に余裕があるときの rAF/queue gate の制約は調べる候補だが、
今回まだ分布・操作 latency を定量測定していない。確認済みの bottleneck と混ぜない。

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
その測定値も現行の200個版との速度比較として解釈しない。

## 5. 実装後に残す比較

### 採用した変更

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

### 現行: 66-unit・ランダム初速シーン計測（2026-10-12）

前項と同じ14,600 active / 1,000,000 capacityの配置・格子・時間刻み・重力を使い、
水・油の速度成分を±12、小石の各頂点の速度成分を±24 world units/sから独立に抽選した。
シーン生成seedは `0x4d504d34` に固定し、計測を再現可能にした。Chrome 154 / Apple Metal 3
（fallbackなし）/ `timestamp-query`、暖機64 step、静止暖機600 ms、12 batch×8 step、
描画なし。GPU確保量は217,228,520 byte。

| profile | GPU median / p95 (ms/step) | CPU normal encode median / p95 (ms/batch) | queue wait median / p95 (ms/batch) | measured pass/batch |
|---|---:|---:|---:|---:|
| batch | 0.06569 / 0.07025 | 0 / 0.10 | 0.90 / 1.00 | 1 |
| stages | 0.09391 / 0.11052 | 0 / 0 | 1.20 / 1.50 | 56 |

stages profileのdispatch attribution medianはclear 0.00571、prepare 0.00690、
bucket構築 0.01158、P2G 0.03598、grid update 0.00977、G2P 0.01672、
commit 0.00502 ms/step。attributionはbatchと異なるpass構成であり、通常batch値と直接比較しない。
CPU時計の分解能のため、encode median/p95の0 msは実処理時間ゼロを意味しない。

ランダム初速で再読み込みしたページの初回CPU/GPU比較は最大絶対誤差 `2.48e-5`、
許容範囲内。GPU benchmark内の45粒子・５モデル／８ step回帰は最大絶対誤差
`1.3956e-5`、batch/単発差0、pick・drag・clone・render・rollback・永続数値statusを確認した。
この短時間試験は長時間弾性復元、頂点間の一体変形、明示的接触を検証するものではない。

## 6. 参照

- [WebGPU: usage scopes](https://www.w3.org/TR/webgpu/#usage-scopes) —
  compute の dispatch はそれぞれ独立した usage scope。
- [WebGPU: compute pass](https://www.w3.org/TR/webgpu/#compute-pass-encoder) —
  dispatch の順序と resource validation を維持したまま１ pass にまとめる設計の根拠。
- [WebGPU: timestamp writes](https://www.w3.org/TR/webgpu/#dictdef-gpucomputepasstimestampwrites) —
  対応 feature が必要。非対応時は queue 時間を GPU 時間と偽装しない。
- [kernel の現在の制約](../src/lib/mpm/kernels/README.md)。
