# 2D GPU MLS-MPM ロードマップ

更新日: 2026-10-12

状態: Gate 0 の GPU 操作ラボはユーザーから稼働確認の報告あり。Gate 2 の選択・局所力ドラッグ・複製は基本操作の動作確認済み。Gate 4 の現行シーンへ水–油法線接触、小石–液体の距離場接触と反力、小石同士の三角形接触、辺長制約を実装した。WebGPU batch は 14,600 粒子で数値status正常を確認し、詳細な時間・誤差・未確認範囲は性能資料に記録する。接触のscene-scale物理妥当性、長時間性能、32×、Android/iOS実機、界面・材料校正、Gate 5の製品統合は未確認。既存のβ v0.2は維持する。

## 材料表現の調査・詳細計画

LiquidFun を参考にした圧力・体積応答、自由表面、材料間接触、変形・弾性復元の知見と
調査根拠は [４論点の調査資料](./RESEARCH-MLS-MPM-MATERIALS.md) に記録する。
現在のGPU pass・buffer・時間制御を出発点にする
[材料表現のGPU実装ロードマップ](./ROADMAP-MLS-MPM-MATERIALS.md) は、
Gate 4接触prototypeの部分実装と、残る実装・検証の詳細計画であり、Gateの合格記録とは区別する。

現行solverのチューニングは [パフォーマンス計測・検討指針](./PERFORMANCE-MLS-MPM.md) と
[パフォーマンスチューニングのロードマップ](./ROADMAP-MLS-MPM-PERFORMANCE.md) に分離する。
計測とロードマップを先に完成させ、物理条件を変えないソース変更をその後まとめて行う。

現在の初期シーンは水8,000・油6,000・正三角形の Neo-Hookean 弾性小石200個
（各3粒子、計600粒子）の合計14,600粒である。小石は14列×15行（最終行4個）に中心間隔68 world unitsで配置し、
各三角形の辺長・水平幅を枠幅の3%（66 world units、直前版の2倍）、`objectId`を3〜202とした。
水・油の初速は各成分±60、小石頂点は±120 world units/sの独立擬似乱数で与える。
旧20,000粒子粉体版と途中の14,003粒子・小石1個版の長時間試験・性能値は現構成の結果ではない。
現構成のGPU buffer確保量は221,564,392 byte。初回局所CPU/GPU比較は位置・変形と速度に別々の誤差基準を使う。

**性能変更の実装結果**: 粒子別の応力・アフィン運動量precomputeと１batch１compute passを実装した。
８stepではcompute passが48から１、dispatchが48から56になる。100万capacityのGPU bufferは、
上記ブラウザー確認時の約201 MBからcache分16 MBを加えた約217 MBになった。
同一物理条件の暖機後A/B３組でGPU step medianは11.8〜16.1%短縮し、
５構成則・batch/単発・pick/drag/clone・容量・render・rollbackの小規模回帰を確認した。
これは材料間接触や自由表面の拡張、モバイル・長時間性能のGate合格ではない。

## 現在の実装状況

- **Gate 0**: `/labs/mpm` に GPU 粒子描画、要求時ピッキング、ドラッグ、複製を実装。粒子位置・速度は GPU buffer を正本とし、毎フレーム全量 readback はしない。ユーザーからラボが動作したとの報告あり。長時間・性能測定は未確認。
- **Gate 1**: `src/lib/mpm/reference/` にCPU reference、`src/lib/mpm/kernels/` にGPU上のP2G / grid update / G2PとGPU描画を実装。`/labs/mpm/kernels` で初回stepのCPU reference比較を行う。Neo-Hookeanの基準シーンを含む。GPU shader compile・数値差・挙動は実機未確認。
- **Gate 2**: `/labs/mpm/kernels` に Gate 0 の操作を統合する。選択時だけ16 byteのGPU結果を読み戻し、ドラッグはgrid上の減衰付き局所力、複製は連続した `objectId` 範囲をGPU内でコピーする。粒子と ID buffer は初期化時に固定容量を確保し、容量超過は明示的に拒否する。基本操作はユーザー確認済み。長時間操作、容量境界、readback、数値安定性、性能は未確認。
- **Gate 3**: 粒子単位でNeo-HookeanとSnow型2D弾塑性モデルを選べる。Snow型は主伸長の降伏射影と塑性体積履歴を持ち、両モデルを同一のGPU粒子配列に置ける。CPU referenceとlayoutテストは実装済み。専用の可視化シーンはGate 4ラボに置き換わった。GPU実機shader compile、混在時の数値比較、繰返し塑性負荷は未完了。
- **Gate 4 試作**: 水・油には引張応力のない弱圧縮性EOS（基準密度0.02・0.016、係数40,000・30,000）を使い、液体は体積を保つ極分解回転へ射影し、G2P後の `det(F)` は0.5〜2に制限する。粉体のDrucker-Prager風モデルと `det(F) >= 0.5` 安定化はsolverに残すが、初期シーンには使わない。現シーンは水8,000・油6,000・Neo-Hookean弾性小石200個（各3粒子、計600粒子）の計14,600粒で、GPU capacityは1,000,000。小石は独立`objectId`の正三角形（辺長66、中心間隔68 world units、λ20・μ10）で、各頂点を±120 world units/s、水・油を各成分±60から独立抽選する。小石の三辺に位置制約を適用し、水・油は相別格子速度場で法線接触を行う。液体–小石は三角形距離場と両側法線impulse、小石同士はSAT法線接触を使う。水・油の密度差による相対沈降・浮上も近似する。GPU buffer確保量は221,564,392 byte。CPU/GPU比較は各材料の局所サンプルとし、全量readbackを避ける。接線摩擦、表面張力、濡れ、完全な界面分離、連続衝突検出、scene-scale運動量・貫入誤差、長時間安定性は未実装または未検証であり、M4合格とはしない。旧20,000粒子粉体版、小石1個版、辺長33 world unitsの途中版の試験・計測は現構成を代表しない。詳細な性能・誤差結果は性能資料に記録する。
- **Gate 4 速度設定**: 描画目標15・30・60・90・120 FPS（既定60 FPS）と、0.5×〜32×の速度倍率（既定2×）を選べる。物理刻みは1/240秒に固定する。複数stepを1つのcommand batchにまとめ、実フレーム間隔に応じて1フレーム当たりの上限を調整する（15 FPS・32×では最大512 step/frame）。上限超過時は遅延を無制限に蓄積せず、超過シミュレーション時間を破棄量として表示する。実効時間倍率も表示し、新設定のPC連続検証後に実機調査へ進む。
- **未実装**: 接線摩擦、表面張力・濡れ・完全な非混和性／分離、連続衝突検出、粉体の塑性圧密・校正。
- **未確認**: Gate 1の弾性基準検証とGate 3のSnow型繰返し塑性負荷、Gate 2〜4の容量境界・操作readback、水・油・小石のscene-scale貫入／運動量／長時間復元、新しいGate 4条件での描画込み実効時間倍率、Android/iOS実機での長時間・熱・性能試験。ラボの性能表示はブラウザー側の測定値で、pass単位のGPU timestampではない。
- **統合方針**: 実験画面は `/labs/mpm` に隔離し、通常画面 `/` の β v0.2 と既存のクラッシュ対策を置き換えない。

## 1. 目的と前提

- WebGPU を使った 2D MLS-MPM の実現可能性を、小さな PoC から確認する。
- 水の自由表面に加え、ゼリー状・塑性の物体を遊びにできるかを評価する。
- GPU 状態の選択・ドラッグ・複製を早期に検証し、毎フレームの粒子状態読み戻しを避ける。
- Android Chrome と iOS Safari を対象とする。WebGPU が使えない端末では、既存エンジンを当面のフォールバックとして残す。
- 物理精度を最初から完成させず、操作性・安定性・持続性能で段階的に判断する。

MLS-MPM は候補エンジンであり、製品全体への採用は各ゲートを通過してから決める。既存エンジンと新エンジンは、検証完了まで同じ実装を置き換えない。

## 2. 最初に固定する設計境界

サブエージェントへ並行作業を依頼する前に、実装責任者が次の共有契約だけを定義する。

- 粒子データ: 位置、速度、アフィン速度行列、変形勾配、材料 ID、オブジェクト ID、生存状態。
- GPU シミュレーション状態は GPU が正本とし、CPU へ全粒子配列を毎フレーム返さない。
- 操作は小さなコマンドとして渡す。選択結果の ID など必要最小限の情報だけ、イベント時に非同期で読み戻す。
- 数値計算、GPU デバイス管理、操作、描画は別モジュールにする。UI ルートから WGSL や MLS-MPM の内部状態へ直接依存させない。
- PoC は `/labs/mpm` 相当の独立画面に置き、既存のトップ画面とシミュレーションを変更しない。
- GPU 非対応、デバイス喪失、初期化失敗は明示し、既存 β へ戻れるようにする。

共有インターフェース案:

```ts
interface MpmBackend {
  initialize(config: MpmConfig): Promise<void>;
  step(input: MpmStepInput): void;
  pick(point: WorldPoint): Promise<ParticleSelection | null>;
  beginDrag(selection: ParticleSelection, target: WorldPoint): void;
  updateDrag(target: WorldPoint): void;
  endDrag(): void;
  clone(selection: ParticleSelection, offset: WorldVector): Promise<boolean>;
  dispose(): void;
}
```

この契約では粒子配列を公開しない。`pick` や `clone` が内部で GPU 状態にアクセスし、CPU には小さな結果だけを返す。型・WGSL の buffer layout・`pick` / `clone` の失敗条件を並行作業開始前に固定する。

## 3. ロードマップと判定ゲート

### Gate 0 — 操作と GPU 状態の PoC

MLS-MPM の完全な実装より先に行う。合成粒子データを GPU buffer に置き、次を確認する。

1. 画面座標から選んだ粒子 ID を取得する。読み戻しはタップ時のみとし、非同期に行う。
2. ドラッグ入力を GPU コマンドとして送り、粒子状態を GPU 上で更新する。
3. 選択対象の複製を GPU 上で行い、容量上限・失敗を扱う。
4. 描画とシミュレーションを GPU 状態から続け、毎フレーム全粒子を CPU に転送しない。
5. 流体用の「局所的な力・追従」と、ゼリー等の「オブジェクトとしての移動」を分けられるか試す。

**合格条件**

- 1 分以上の連続操作で、粒子状態の全量 readback が発生しない。
- 選択、ドラッグ、複製が一連の操作として動作し、容量不足などの失敗を UI に返せる。
- 実機で 30 fps を目標に測定でき、WebGPU 非対応時は既存画面を壊さない。

ここで GPU 状態との操作が成立しない場合は、MLS-MPM の物理機能を増やす前に、選択単位や複製対象を見直す。

### Gate 1 — 2D MLS-MPM の数値コア

- まず単一材料、重力、閉じた境界の小さな 2D シーンを作る。
- MLS-MPM の P2G、格子更新、G2P を GPU compute pass に分ける。
- WGSL の P2G 集約方法を早期に決める。WGSL の標準 atomic は整数型であるため、固定小数点 atomics と gather 方式などを比較し、精度・オーバーフロー・性能を確認する。
- CPU の小規模 reference implementation と比較し、粒子数、境界、異常値をテストする。
- 水一種類で、傾き重力による移動、壁への衝突、停止・再開を確認する。

**実装済みの検証画面**: `/labs/mpm/kernels` で単一材料の Neo-Hookean 弾性モデルを実行し、
初回1ステップを CPU reference と照合する。
この画面を WebGPU 対応端末で開き、CPU/GPU 誤差、壁、停止・再開、数値状態を確認する。
画面が表示されるだけでは Gate 1 の合格とはせず、実機での数値結果を記録する。

**合格条件**

- 既知の初期条件で粒子が境界を抜けず、NaN / Infinity が発生しない。
- 粒子数・質量が意図せず増減しない。外力なしのテストで挙動を再現できる。
- GPU 計算と reference implementation の差を計測し、許容誤差を記録する。

### Gate 2 — 操作 PoC と MLS-MPM の統合

- Gate 0 の選択、ドラッグ、複製を MLS-MPM 粒子状態へ接続する。
- 流体は個々の粒子を固定するのではなく、選択点まわりの力または追従制約として扱う。
- ゼリー・塑性体は安定した `objectId` を使ったグループ選択を候補とする。
- 選択、ドラッグ、複製中の粒子上限、GPU buffer 再確保、操作キャンセルを確認する。

液体の連結粒子を毎タップで CPU flood-fill する設計は採らない。液体の選択意味（一点、ブラシ半径、塊）と、物体の選択意味（object ID）を分けて評価する。

**実装状況**: 単一材料の弾性クラスターを対象に、GPU上の最近傍選択、grid上の局所力ドラッグ、
`objectId` グループのGPU複製を実装する。複製時のbuffer再確保は行わず、初期化時の固定容量と
明示的な上限通知を使う。ポインターキャンセル、ページ非表示、GPU device loss でドラッグを
終了する。基本操作が動作することはユーザー確認済み。これは操作統合のPoCであり、水の構成則・
選択オブジェクトだけへの力の限定・長時間安定性・実機での30 fps達成を証明するものではない。

**実機合格条件**

- 1分以上のドラッグ操作で粒子全量をCPUへ読み戻さず、タップ時の選択結果だけを読み戻す。
- 粒子の変位が局所力として作用し、選択した物体を固定・剛体移動する挙動にならない。
- GPU複製後に粒子数と `objectId` が維持され、容量上限では複製失敗を画面に表示する。
- 30 fpsを目標に、対象端末で選択・ドラッグ・複製を確認する。Gate 1の数値条件も別に合格させる。

### Gate 3 — ゼリー状・塑性物体

- MLS-MPM の材料モデルで弾性体／塑性体を追加する。
- 物体 ID の選択、ドラッグ、複製を確認する。
- 液体・ゼリー・塑性物体を同一の粒子配列に置いたときの境界と安定性を検証する。
- ここでは剛体衝突、破砕、液体との濡れまで一度に実装しない。追加ごとに独立したシーンとテストを作る。

**コード実装**: 粒子ごとの材料指定と、柔らかいNeo-Hookean弾性クラスター、Snow型
弾塑性クラスターを同一粒子配列で計算するPoCを追加した。Snow型は2D極分解による回転応答、
主伸長の降伏射影、塑性体積 `Jp` の更新・硬化を行う。GPU複製はこの材料状態を含めて複製する。
これはGate 3の実装途中であり、実機合格ではない。材料間の明示的接触や摩擦もない。

**次の受け入れ確認**: GPU shader compileとCPU/GPU 1-step誤差、弾性体と塑性体が共有grid上で
安定して動くこと、繰返し押し込み後の `Jp` と変形、複製後に材料モデル・`Jp`・`objectId` が
維持されることをWebGPU実機で確認する。

### Gate 4 — 水・油・弾性小石の混在

- 水と油を別材料として扱い、密度差と界面の維持方法を比較する。
- 粉体構成則はsolverに残すが、現在のデモでは弾性小石群に置き換えた。粒状材料の実用化は別途検証する。
- 空気は初期 PoC では粒子を満たす気体ではなく、自由表面の外側の周囲として扱う。気体の明示的シミュレーションが必要になった時点で別途設計する。
- 混和、分離、浮沈、粉の沈降、数分間の粒子・質量安定性を個別の合格条件にする。

**コード試作**: `/labs/mpm/kernels` に水・油と弾性小石の混在シーンを実装した。水・油は
`lambda * min(J - 1, 0) * I` の体積応力だけを持ち、せん断応力を持たない。基準密度と粒子体積から
粒子質量を変える。小石はlambda 20・mu 10のNeo-Hookean材料を割り当てた三頂点の正三角形で、
200個を14列×15行（最終行4個）に中心間隔68 world unitsで配置する。各objectIdは個別で3〜202、
辺長・水平幅は66 world units（計算枠幅の3%、直前版の2倍）。各小石の3辺を位置制約し、
小石同士はSAT法線接触、水・油は相別gridで法線impulseを解く。
液体と三角形小石は距離場で接触し、法線反力を小石頂点にも返す。
初期シーンは水8,000・油6,000・小石600粒で計14,600粒、GPU capacityは1,000,000。
初回 CPU/GPU 比較は水・油と小石群中央の局所粒子サンプルだけを読み戻し、
明示的な一時停止状態確認は従来どおり全量を確認する。
性能欄には描画FPS、step/s、実効時間倍率、上限で破棄した時間、submit換算particle-steps/s、
CPU batch submit平均とqueue完了待ちを表示する。水・油は局所液体密度に基づく相対沈降・浮上の近似補正を行い、
初回 CPU/GPU 比較サンプルに両相の接触近傍を含める。
GPU queueに未完了の仕事を積み続けないよう、queue fence完了後に次frameを送信する。

**旧シーンでの確認済みGPU検証**: ローカルデスクトップ Chrome でshader compile、初回 CPU/GPU 比較、
粉体を含む20,000粒子混合シーンを既定重力90で60 simulated seconds実行し、全粒子状態を30秒ごとに確認した。
この長時間試験と性能A/Bは小石への置換前のシーンに対する結果である。66-unit現シーンではGPU shader compile、
乱数初速を含む局所 CPU/GPU 比較、８step batch とGPU pass timing、45粒子の５材料 regression を確認した。
GPU batch timing の測定値は性能資料を参照する。これは長時間弾性復元、頂点間の連続体変形、
接触後のscene-scale運動量・貫入を検証するものではない。
**既知の制限**: Android/iOS実機、5分以上の熱・持続性能、小石の長時間弾性復元は未確認。
水・油は別々のgrid速度場を持ち、法線接触を行うが、静的な重なりの位置分離は保証しない。
液体–小石と小石–小石の法線接触、および各小石の辺長制約は限定的な離散近似である。
接線摩擦、表面張力、濡れ、完全な非混和性、連続衝突検出はない。
各小石は3粒子・約6.6セル幅で、長時間の形状保持や連続体としての復元は未確認。
局所密度補正も浮沈を促す近似で、境界維持や安定した相分離を保証しない。
粉体モデルは圧密・凝集・実材料の校正を含まないため、
現段階でGate 4合格とはしない。

**判定**: 水・油の分離や粉体挙動に過大な複雑さが必要なら、MLS-MPM コアを全面拡張せず、別の界面モデルまたはハイブリッド方式と比較する。

### Gate 5 — モバイル持続性能と製品統合

- Android Chrome と WebGPU 対応 iOS Safari の実機で、静置、傾斜、ドラッグ、複製を含む 5 分以上の連続試験を行う。
- 30 fps を基準に、フレーム時間、GPU pass 時間、CPU 時間、粒子数、GPU buffer 使用量、発熱、メモリ推移を記録する。
- background / foreground、画面回転、WebGPU device loss 後の再初期化を試す。
- 非対応端末では既存エンジンへ戻し、PoC が起動不能でも β v0.2 の操作を失わないようにする。
- 実機合格後に限り、既存画面へ段階的に導入する。PoC の検証段階では既存 `src/routes/+page.svelte` を変更しない。

## 4. 並行作業の分割

まず Gate 0 で共有契約を凍結し、その後、別々のディレクトリと成果物を所有する作業を並行させる。統合責任者は契約・ルート画面・最終マージを担当し、サブエージェントには共有ファイルの編集を依頼しない。

| 作業 | 独立した担当範囲 | 成果物 | 前提・境界 |
|---|---|---|---|
| A. GPU 操作 PoC | `src/lib/mpm/interaction/` | GPU picking、drag command、GPU clone、合成粒子でのテスト | 固定済み `MpmBackend` / buffer layout に従う。MLS-MPM kernel は変更しない |
| B. MLS-MPM reference と物理テスト | `src/lib/mpm/reference/` | 2D 更新式、CPU reference、境界・質量・材料モデルのテスト | UI、GPU device 管理、WGSL ファイルは変更しない |
| C. WGSL 数値カーネル | `src/lib/mpm/kernels/` | P2G 集約 microbenchmark、格子更新、G2P kernel、精度測定 | 共有 buffer layout に従う。最初は水一材料。操作コードは変更しない |
| D. WebGPU 対応・ライフサイクル | `src/lib/mpm/platform/` | capability probe、device loss / dispose / restore、端末計測の記録形式 | 物理式・particle layout・既存トップ画面は変更しない |
| E. 統合担当 | `src/routes/labs/mpm/+page.svelte` と共通契約 | PoC 画面、backend 選択、成果物の接続 | A-D の契約を変更しない。競合を解消し、対象端末で統合検証する |

### 並行実行の依存関係

```text
共有型・WGSL layout・合格条件を凍結
        ├── A. GPU 操作 PoC ──────────────┐
        ├── B. MLS-MPM reference ──┐      │
        ├── C. P2G/G2P microbench ─┴──────┤
        └── D. WebGPU platform probe ──────┤
                                           └── E. 統合・実機ゲート
                                                 ├── water vertical slice
                                                 ├── gel / plastic
                                                 └── water / oil / powder
```

B と C は物理式・数値精度を共通の小さなテストベクトルで合わせるが、ファイルは分離する。A は合成粒子だけで操作系を先に進めるため、MLS-MPM の実装待ちにならない。D は対象ブラウザーと device loss の制約を早く発見し、WebGPU が使えない場合のフォールバック判断を独立して報告する。

全サブエージェントに共通する禁止範囲:

- 他担当のディレクトリ、`src/routes/+page.svelte`、`package.json`、共有契約を無断変更しない。
- ライセンス未確認コードをコピーしない。参考実装からコードを取り込む場合は、元 repository のライセンスと同梱物を確認し、必要な attribution を保持する。
- 性能を実機計測せず「モバイルで高速」と結論しない。

## 5. 主要な技術リスクと判断

- **GPU scatter 集約**: P2G の float 加算をどう集約するかが最初の数値・性能リスク。固定小数点 atomic の誤差と上限を測る。
- **粒子選択の意味**: MLS-MPM の液体粒子群はセル CA の連結成分と同じではない。流体操作と固体グループ操作を分ける。
- **複製容量**: GPU 上の可変粒子数、空き slot、失敗通知を最初に設計する。複製時だけ全状態を CPU にコピーする抜け道は作らない。
- **複数材料**: MLS-MPM を選ぶだけでは水・油の非混和、粉体、気体の相互作用は得られない。単一水から段階的に加える。
- **WebGPU 対応差**: iOS の対応 OS と Android GPU に制約がある。PoC は `navigator.gpu` と adapter の両方を実機確認し、非対応時は既存版へ戻す。
- **GPU メモリ・context loss**: 長時間操作、バックグラウンド復帰、端末発熱を含む soak test を統合前の必須ゲートにする。

## 6. 参照資料

- [MLS-MPM 論文: A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling](https://yuanming.taichi.graphics/publication/2018-mlsmpm/mls-mpm-cpic.pdf)
- [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) — MLS-MPM / SPH の WebGPU 参考実装。依存ライブラリではなく、コード採用時は MIT 条件と移植範囲を確認する。
- [jeantimex/fluid](https://github.com/jeantimex/fluid) — WebGPU の FLIP/PIC 比較対象。
- [WebGPU 実装状況](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)
- [Apple Safari 26 Release Notes](https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes)
