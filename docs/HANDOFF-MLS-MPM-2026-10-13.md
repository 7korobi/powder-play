# MLS-MPM 作業引継ぎ・動作確認手順

記録日: 2026-10-13

この文書は、次の作業再開時に現在地・再現手順・判断上の注意をすぐ確認するための入口である。
詳細な設計、stage別gate、長期の性能履歴は末尾の既存資料を正とする。

## 1. 現在地

- 対象ラボ: `/labs/mpm/kernels`。Stage 1–4の操作経路と物理prototypeはあるが、全stageの
  実機受け入れやscene-scale物理受け入れを完了した状態ではない。
- 現行初期scene: 水8,000、油6,000、Neo-Hookean小石200個（各3粒子、計600）、
  12×8 node・249 bondの破断solid（96粒子）、合計14,696 active粒子。
- GPU capacityは1,000,000。Apple Metal 3 / Headless Chrome 154でのbuffer確保量は
  221,626,976 byte。これは確保量であり、同時に動かせる物体数や持続性能の保証ではない。
- benchmark harnessに制御S3/S4 fixtureを追加した。今回の受け入れ確認ではsolver/shader本体を
  変更していない。破断stateの `COPY_SRC` は、時間計測後の制御fixture readbackに限って有効にする。
- `fix(mpm): preserve progress under adaptive CFL limits`（`c471ee2`）でadaptive CFL、
  rollback/retry、および短い安全刻みへの分割が追加された。続く操作改善では未処理時間の表示、
  一時停止中の明示的な時間破棄、emitter CFL事前確認、エラー時の初期シーン再起動を追加した。
- この更新時点の自動検証は `npm test` 20 files / 178 tests、`npm run check` 0 errors /
  0 warnings、`npm run build` 成功。共有ブラウザーでの今回の操作UI smoke checkと、adaptive
  CFL変更後の実GPU性能・scene-scale物理受け入れは未実施で、旧benchmark値から推定しない。
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

adaptive CFLと操作改善を含む更新後sourceでは `npm test` が20 files / 178 tests、
`npm run check` が0 errors / 0 warnings、`npm run build` が成功した。
これらのコマンド自体は実GPUや物理品質の合格を意味しない。

### 2.2 ラボのブラウザーsmoke check

現行checkoutのVite serverを起動し、出力されたURLに `/labs/mpm/kernels` を付けて開く。
同じportに別checkoutの古いdev serverが残っていないか注意する。

```sh
npm run dev -- --host 127.0.0.1 --port 4181 --strictPort
```

確認する項目:

1. 見出しが「水・油・弾性小石・GPU破断固体 · 14,696粒子」で、statusが
   `GPU MLS-MPM を実行中です` になる。
2. 活動数・使用slotが初期状態で14,696、gridが220×150と表示される。
3. 手動重力を変え、一時停止・1 step・表示mode切替が操作できる。
4. emitterを水・64粒子にして `1回追加` を押す。以前の実操作では受入64、拒否0、
   容量超過0となり、活動数・使用slotが14,696から14,760へ増えた。
5. 繰返す時はreloadして初期sceneに戻す。outletで消したslotを含め、削除slotは再利用されない。

このsmoke checkで確認済みなのはラボ起動と上記の水64粒子の一回追加まで。
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

### 2.4 最終確認済みsnapshot（2026-10-13、adaptive CFL変更前）

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
- これらは小規模・制御fixtureの結果。14,696粒子sceneのinterface penetration、
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

### 共有ブラウザーの注意（2026-10-13）

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
