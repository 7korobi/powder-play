# MLS-MPM 作業引継ぎ・動作確認手順

記録日: 2026-10-14

この文書は、次の作業再開時に現在地・再現手順・判断上の注意をすぐ確認するための入口である。
詳細な設計、stage別gate、長期の性能履歴は末尾の既存資料を正とする。文書名は旧baseline記録に
対応したまま維持している。

## 1. 現在地

- 対象ラボ: `/labs/mpm/kernels`。Stage 1–4の操作経路と物理prototypeはあるが、全stageの
  実機受け入れやscene-scale物理受け入れを完了した状態ではない。
- 現行初期scene: 水2,000、油1,500、Neo-Hookean小石50個（各3粒子、計150）、
  6×4 node・53 bondの破断solid（24粒子）、合計3,674 active粒子。旧14,696の1/4。
- 固体 particle reference areaは旧25から100へ4倍、固体massは1から4へ4倍。液体参照面積は100から400へ
  さらに4倍、密度0.02・0.016を維持して各相の総参照体積・massも直前設定の4倍
  （水16,000・油9,600）とした。液体粒子数・間隔18・初期位置は変更していない。
  粒子 marker 半径は2倍となる。水・油のEOS係数は160,000・120,000
  （従来の40,000・30,000から4倍）へ上げ、相別grid接触に重なり量・接近速度を使う
  equal-and-opposite separation impulseを加えた。水・油burstにも同じ参照面積とEOSを適用する。
- 小石を液体範囲内へ配置し、初期速度は水・油各成分±64、小石頂点±240、破断solid各成分±80。
  小石はsurface・particle-marker両modeで塗り三角形と輪郭線を描画する。左右揺れは初期offで、
  UIボタンから既定±240 world units/s²・0.16 Hzを開始／停止できる。
- GPU capacityは1,000,000。液体volumeを4倍にする前のApple Metal 3 / Headless Chrome 154
  `batch` profile（3,674粒子、描画なし、12 sample×8 step）ではGPU buffer確保量
  231,188,500 byte、GPU step median 0.217 ms / p95 0.256 msだった。最新設定での再測定は本書末尾に記録する。
  `stages` profileはmedian 0.267 ms / p95 0.281 msだが、dispatchごとにpassを分ける診断値であり、
  通常batchとの比較には使わない。  queue完了待ちはmedian 2.4 ms / p95 2.6 msでGPU単体時間ではない。
  単一端末・短時間・描画なしの測定で、適応CFLを通した描画込みの持続性能を保証しない。
  旧14,696 sceneの221,626,976 byteは別revision・別sceneの歴史値である。
- benchmark harnessには制御S3/S4 fixtureがある。retuneでlayout・solver・shaderにも変更が入った。
  破断stateの `COPY_SRC` は時間計測後の制御fixture readbackに限って有効にする。
- `fix(mpm): preserve progress under adaptive CFL limits`（`c471ee2`）でadaptive CFL、
  rollback/retry、および短い安全刻みへの分割が追加された。続く操作改善では未処理時間の表示、
  一時停止中の明示的な時間破棄、emitter CFL事前確認、エラー時の初期シーン再起動を追加した。
- 2026-10-14のscene retune後の自動検証・browser smoke結果は、本書末尾の作業完了記録に記録する。
  2026-10-13以前の178 testsと14,696 scene benchmarkは旧構成の記録であり、新sceneの実GPU性能・
  scene-scale物理受け入れを示さない。
- 継続作業前には `git status` / `git diff` を先に確認し、既存変更をreset/checkoutで破棄しない。

## 2. 再現する検証手順

### 2.1 Node・自動検証

Node.js 22.17以降を使う。依存関係が未導入の初回だけ `npm install` を行い、その後はrepository rootで検証する。

```sh
node --version
# 初回セットアップ時のみ
npm install
# 通常の検証
npm run check
npm test
npm run build
```

adaptive CFLと操作改善を含むretune後sourceでは `npm test` が20 files / 192 tests、
`npm run check` が0 errors / 0 warnings、`npm run build` が成功した。
これらのコマンド自体は実GPUや物理品質の合格を意味しない。

### 2.2 ラボのブラウザーsmoke check

現行checkoutのVite serverを起動し、出力されたURLに `/labs/mpm/kernels` を付けて開く。
同じportに別checkoutの古いdev serverが残っていないか注意する。

```sh
npm run dev -- --host 127.0.0.1 --port 4181 --strictPort
```

確認する項目:

1. 見出しが「水・油・弾性小石・GPU破断固体 · 3,674粒子」で、statusが
   `GPU MLS-MPM を実行中です` になる。
2. 活動数・使用slotが初期状態で3,674、gridが220×150と表示される。
3. 手動重力、一時停止・1 step、表示mode切替、左右揺れの開始／停止を操作する。
   初期状態は揺れ0、開始ボタンで±240、停止ボタンで0へ戻ることを確認する。
4. emitterを水・64粒子にして `1回追加` を押す。受入64、拒否0、容量超過0となり、
   活動数・使用slotが3,674から3,738へ増えることを確認する。
5. 繰返す時はreloadして初期sceneに戻す。outletで消したslotを含め、削除slotは再利用されない。

2026-10-14のretuned scene browser smokeでは項目1・2、canvas描画、左右揺れの初期0から
開始±240・停止0への切替を確認し、起動・操作中のconsole errorは0件だった。
手動重力・一時停止/1 step・表示mode切替・emitter追加はこのsmokeでは未操作。
傾きsensorの許可・画面向き、連続追加停止、油/固体emitter、容量境界、outletの対象限定、
負荷をかけた破断は未受け入れである。モバイルsensor試験はHTTPSのAndroid Chrome /
iOS Safari実機で別途行い、stage別手順は[Stage 1–4ロードマップ](./ROADMAP-MLS-MPM-STAGES-1-4.md)に従う。
未処理時間表示・破棄ボタン、高速度emitterのCFL事前確認、停止後の再起動、pause中の連続追加停止も、
今回のUI変更後にはブラウザーで再確認していない。

### 2.3 実GPU benchmarkと制御物理fixture

このscriptはTypeScript moduleを配信するVite dev serverと、localhostだけで待ち受ける専用Chrome CDPを使う。
preview server、software/fallback adapter、`timestamp-query`のないadapterでは測定を代用せず、明示的に失敗させる。

Viteは前節のport `4181` で起動する。専用CDP endpointが既にある場合はChromeを重複起動しない。
なければmacOSでは次のように専用profileで起動する。普段使いのChrome profileは使わず、
CDP addressをloopback以外へ公開しない。

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/powder-play-mpm-benchmark-profile \
  about:blank
```

別terminalから通常batchを実行する。

```sh
npm run benchmark:mpm -- --profile=batch --output=/tmp/mpm-batch-01.json
```

これは1回分の計測である。run間のばらつきを見る時は、出力先のsuffixを変えて複数回実行し、
各runのmedianを比較する。診断用stage attributionは別に実行する。

```sh
npm run benchmark:mpm -- --profile=stages --output=/tmp/mpm-stages.json
```

`stages` profileはdispatchごとにcompute passを分割するため、通常の `batch` profileの時間と
直接比較しない。`--origin=http://127.0.0.1:PORT` と `--cdp=http://127.0.0.1:PORT` で
接続先を変更できる。harnessは自分のtargetを作成・終了し、共有browser tabは操作しない。
JSONにはperformance、45粒子/5材料の回帰、S3/S4のphysicsAcceptanceが含まれる。
CDP切断・source hashの変化・数値失敗はbenchmark失敗として扱い、出力だけを性能結果として採用しない。
Chrome起動条件と指標定義は[性能計測資料](./PERFORMANCE-MLS-MPM.md)を参照する。
adaptive CFL変更を含む現行treeの性能・fixture結果を記録するには、その変更を固定した後に実行し、
出力JSONと新しいsource hashを一組として保管する。

### 2.4 歴史的benchmark snapshot（2026-10-13、adaptive CFL・scene retune前）

5回の有効な `batch` runにおけるGPU step medianは
`0.41254 / 0.40143 / 0.40109 / 0.36103 / 0.39817 ms`。
runごとのmedianの中央値は **0.40109 ms/step**、runごとのp95の中央値は
**0.43035 ms/step**。第4runは速い側に外れた。warmupが600 msの実時間指定で、
runごとのwarmup step数・scene状態が完全一致しないため、1回の値を代表値にしない。

- **S3 water/oil contact**: 2粒子を8 step。CPU/GPU最大状態差 `3.8653e-5`、
  phase別mass各`0.25`、最大運動量残差`1.1921e-7`、8 step全てでcrossingなし。
  最初の接近stepでclosing motionも減った。
- **S4 fracture**: 4粒子・3 bond、failure strain `0.1` のfixtureでstrain
  `0.05 / 0.11 / 0` を比較し、閾値上のbond index 1だけが破断。
  閾値下bond damageは約`0.499992`。粒子state誤差は最大`7.63e-7`、
  particle/bond damage誤差は`7.63e-6`以下、質量4・運動量残差0。
  初期broken bondがrest lengthで2 step後もbroken/damage 1を保つfixtureも通過。
- **数値回帰**: 45粒子・5材料・8 stepのCPU/GPU最大差`1.3955e-5`、
  batchと単step呼出しの差0。pick / drag / clone / render / rollback /
  persistent numerical statusも確認。
- これらは小規模・制御fixtureおよび旧14,696粒子sceneの結果。現3,674粒子sceneのinterface penetration、
  相別mass/momentum、実sceneでのfracture pattern、長時間形状保持を検証した結果ではない。
  `1e-3`は数値parity toleranceであり、製品品質の貫入・形状許容値ではない。

このsnapshotのmedianからの計算だけでは、2×で約0.193 GPU秒/実時間秒、32×で約3.080 GPU秒/実時間秒を要する。
これは変更前sourceの描画・CPU・発熱を含まないcompute概算で、現行adaptive CFL経路の予測値でも
実効時間倍率の達成確認でもない。
物理受け入れを先に進め、32×はstretch goalとして扱う。

## 3. 作業を通して得た注意点

- **検証の層を混同しない**: unit/fake-GPU testはcommand order・境界・容量等を検査する。
  初回UI parityは各材料の少数probe、benchmark parityは固定した小規模全粒子fixture、
  実scene acceptanceはscene-scaleの物理量・長時間挙動を調べる。どれか一つで他を代替しない。
- **誤差値を単一thresholdで読まない**: UIの初回probeは速度等に相対許容値も使う。
  2026-10-13に共有された現行ページでは、最大絶対差`1.45e-2`（小石のvelocity.y、
  GPU/CPUとも約`-28.4`）でもUI判定は「誤差基準内」だった。
  これはbenchmark fixtureの`1e-3` thresholdやscene物理品質とは別の判定である。
- **GPU時間とqueue待ちを分ける**: 性能比較にはhardware adapterのGPU timestampを使う。
  CPU encode表示の`0 ms`は時計分解能による丸め、queue completion waitはGPU時間ではない。
- **測定条件を固定・反復する**: adapter、scene、step、warmup、描画有無を記録し、median/p95を報告する。
  `stages` attributionはpass構成自体が通常batchと異なる。
- **容量と物理忠実性を取り違えない**: 1,000,000粒子分のbuffer確保は同数を安定して動かす保証でない。
  water/oilは制御caseのnormal contact確認までで、表面張力・濡れ・接線摩擦・任意初期配置での
  完全分離は未実装または未検証。fracture threshold `0.12`もこのsceneの探索値で素材強度ではない。
- **scene改版間の数字を直結しない**: 旧20,000粒子粉体、14,600粒子小石、現行14,696粒子solid追加後は
  workloadが異なる。性能差や物理差をpaired testなしで比較しない。

### 旧sceneでの共有ブラウザー記録（2026-10-13）

共有された `127.0.0.1:4181` は現行14,696粒子routeで、GPU実行中と初回macrostepの
3 CFL microsteps表示を確認した。これは現在の未commit変更を含むbrowser smokeであり、M2の数値・時間会計gateの合格ではない。
一方 `127.0.0.1:5173` の共有tabは14,600粒子の旧表示で、consoleに
「全状態で有効な波速・速度上限がないため、CFL microstepは実行できない」という
`RangeError` が記録されていた。別dev server/古いmodule stateの可能性があるため、
このtabを現行checkoutの合否として扱わない。毎回current checkoutからserverを起動し、
表示particle数とruntime statusを確認する。

## 4. 次に進める順番

物理受け入れ優先の方針は維持する。

1. **M2 adaptive CFL**: 現在の未commit変更についてtype/unit testと実GPUを確認する。
   retry/rollbackの回数と理由、要求・完了duration、512 microstep予算超過時の停止を記録し、
   既存の変更前M0 snapshotも比較用に保ち、細かいstepのreferenceへの収束と数値保存量を確認する。
2. **M0/M3 baselineとgeometry**: CFL変更後のsource hashでM0を再計測し、材料別面積fieldと
   表示geometryをcontact/pressureと区別して校正する。
3. **S3 scene-scale**: 静止層状・斜めinterface、速度差・密度差のfixtureを増やす。
   interface crossing/penetration、反発、gap、数値拡散、相別mass/momentumを測り、
   製品として許容する貫入閾値とstep数を先に決める。
4. **S4 loaded body**: 12×8 bodyに段階的負荷を加え、damage pattern、fragment motion、
   mass/momentum、boundsを実GPUで確認する。単体後に液体荷重を加え、failure strainやstress fadeを校正する。
5. **S1/S2 device・操作受け入れ**: Android/iOSのsensor permission・画面向き、continuous emitter停止、
   水/油/固体追加、容量境界、outlet対象を記録し、統合操作を行う。
6. **Gate 5の性能baselineと必要な最適化**: acceptance thresholdを固定してから対象端末の持続性能を測る。P5等のbottleneck最適化は
   baselineと物理oracleを固定した後に行い、変更後もS3/S4を再検証する。
7. Stage 5–6 runtimeはStage 1–4統合・端末buffer予算のgateを通過した後に限る。
   変更ごとに対応するphysics fixtureとsceneで再測定し、物理条件変更を高速化と混ぜない。

詳細な未完了項目と各stageの合格条件は[Stage 1–4ロードマップ](./ROADMAP-MLS-MPM-STAGES-1-4.md)、
実測値・hash・benchmark条件は[性能計測資料](./PERFORMANCE-MLS-MPM.md)を参照する。

## 5. 関連資料

- [2D GPU MLS-MPM全体ロードマップ](./ROADMAP-MLS-MPM.md)
- [LiquidFunとGPU MLS-MPMの設計比較](./DESIGN-LIQUIDFUN-VS-GPU-MPM.md)
- [パフォーマンスチューニングロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md)
- [benchmark harness](../scripts/benchmark-mpm.mjs)
- [kernel実装と制約](../src/lib/mpm/kernels/README.md)

## 6. 2026-10-14 retune後の検証記録（液体面積100の設定）

- 自動検証: `npm test` 20 files / 198 tests、`npm run check` 0 errors / 0 warnings、
  `npm run build` 成功。
- 実GPU: `npm run benchmark:mpm -- --profile=batch`。Headless Chrome 154 / Apple Metal 3
  （fallbackなし、`timestamp-query`あり）、3,674粒子、capacity 1,000,000、220×150 grid、
  描画なし。GPU buffer確保量231,188,500 byte。12 sample×8 stepのGPU stepはmedian
  0.217 ms / p95 0.256 ms、queue完了待ちはmedian 2.4 ms / p95 2.6 ms。
  `stages` profileはmedian 0.267 ms / p95 0.281 msだが、passを112へ分割する診断値であり、
  通常batchと直接比較しない。いずれも単一端末の短時間測定で、適応CFL・描画込み・持続性能は未保証。
- 数値fixture: 45粒子・5材料・8 step回帰の最大絶対差0.0000139553、batch/single差0。
  GPUのwater/oil contact fixture（初速±64、相対接近速度128 world units/s、dt 0.001）は
  8 stepで最大差0.0000519、各相mass error 0、momentum residual 2.86e-6、
  全stepで粒子crossingなし。CPU referenceの接触fixtureは±0.5 / ±64 world units/s、
  dt 0.0001・0.001・0.01で分離方向と運動量を検査した。
  EOS圧縮回復・triangle pair contact・fractureも通過したが、いずれもscene全体や
  長時間の物理品質を保証しない。
- ブラウザー: 1440×1100の専用Headless Chromeで水・油・小石のcanvas描画、scene見出し、
  3,674 active粒子、220×150 grid、GPU実行中を確認。揺れの開始で±240、停止で0に戻り、
  その後15秒の実行中にstatus・粒子数・canvasは維持され、runtime/console errorは0件。
  表示mode切替・emitter追加・容量境界・scene-scale接触品質は未確認。
- 現行sceneの単一runであり、mobile/他adapter、scene-scale接触・破断品質は未検証。
  測定条件とsource hashは[性能計測資料](./PERFORMANCE-MLS-MPM.md)を参照。

## 7. 液体参照面積をさらに4倍にした後の確認

- Gate 4の水・油だけparticle reference areaを100から400へ4倍にした。密度、粒子数3,500、
  spacing 18 world units、初期位置は維持し、各相の総質量・参照体積は水16,000・油9,600。
  物理面積に連動してparticle-marker半径が2倍になる。固体のarea/massとEOS/contact係数は変更なし。
- 自動検証: `npm test` 21 files / 203 tests、`npm run check` 0 errors / 0 warnings、
  `npm run build` 成功。
- 現行sceneの `npm run benchmark:mpm -- --profile=batch` はGPU数値guardで失敗した。
  このharnessはGate 4を固定dtで直接進め、lab本体のadaptive CFL/retryを通さない。
  area400の初期sceneでは約600 ms warmup後の固定step中にparticle 509の変形更新上限を越えたため、
  現設定のbenchmark値は採用しない。以前の0.217 msはarea100の歴史値である。
- ブラウザーsmoke: 専用Headless Chrome / Apple Metal 3で、自由表面canvasでadaptive labを15秒実行し、
  3,674粒子・GPU実行statusを保ち、runtime/console errorなし。particle-marker modeにも切り替え、
  拡大したmarkerと実行継続を確認した。未処理時間・端末持続性能・scene-scale接触品質は別途検証が必要。
