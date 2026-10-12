# 2D GPU MLS-MPM ロードマップ

更新日: 2026-10-12

状態: Gate 0 の GPU 操作ラボはユーザーから稼働確認の報告あり。Gate 1 の CPU 数値
reference と GPU MLS-MPM kernel は実装済みだが、実機での shader compile と数値検証は未完了。
Gate 2 の GPU 選択・局所力ドラッグ・オブジェクト複製は `/labs/mpm/kernels` に統合し、
ユーザーから基本操作の動作確認あり。Gate 3 の Neo-Hookean / Snow 型弾塑性と Gate 4 の水・油・
粉体構成則はコード試作済み。Gate 1〜4 の GPU 実機受け入れ、Gate 4 の界面・粒状挙動検証、
Gate 5 の持続性能と製品統合が残る。既存の β v0.2 は維持する。

## 現在の実装状況

- **Gate 0**: `/labs/mpm` に GPU 粒子描画、要求時ピッキング、ドラッグ、複製を実装。粒子位置・速度は GPU buffer を正本とし、毎フレーム全量 readback はしない。ユーザーからラボが動作したとの報告あり。長時間・性能測定は未確認。
- **Gate 1**: `src/lib/mpm/reference/` にCPU reference、`src/lib/mpm/kernels/` にGPU上のP2G / grid update / G2PとGPU描画を実装。`/labs/mpm/kernels` で初回stepのCPU reference比較を行う。Neo-Hookeanの基準シーンを含む。GPU shader compile・数値差・挙動は実機未確認。
- **Gate 2**: `/labs/mpm/kernels` に Gate 0 の操作を統合する。選択時だけ16 byteのGPU結果を読み戻し、ドラッグはgrid上の減衰付き局所力、複製は連続した `objectId` 範囲をGPU内でコピーする。粒子と ID buffer は初期化時に固定容量を確保し、容量超過は明示的に拒否する。基本操作はユーザー確認済み。長時間操作、容量境界、readback、数値安定性、性能は未確認。
- **Gate 3**: 粒子単位でNeo-HookeanとSnow型2D弾塑性モデルを選べる。Snow型は主伸長の降伏射影と塑性体積履歴を持ち、両モデルを同一のGPU粒子配列に置ける。CPU referenceとlayoutテストは実装済み。専用の可視化シーンはGate 4ラボに置き換わった。GPU実機shader compile、混在時の数値比較、繰返し塑性負荷は未完了。
- **Gate 4 試作**: 水・油に引張応力を持たない弱圧縮性EOSを使い、基準密度と基準体積から粒子質量を計算する。粉体は圧力依存のせん断降伏を持つDrucker-Prager風の試作モデル。CPU reference、GPU state、WGSL、mixed-material画面、CPU/layoutテストを実装した。全材料が単一のgrid速度場を共有するため、水・油の界面維持・浮沈・明示的接触は未実装で、粉体も校正済み粒状モデルではない。
- **未実装**: 水・油の界面張力／非混和性、密度差による浮沈、材料間接触・摩擦、粉体の塑性圧密・校正。
- **未確認**: Gate 1〜4のGPU実機shader compile・CPU/GPU数値検証、Gate 2〜4の長時間・容量境界・readback・性能試験。
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

### Gate 4 — 水・油・粉の混在

- 水と油を別材料として扱い、密度差と界面の維持方法を比較する。
- 粉は液体と同じ材料モデルに押し込まず、粒状材料として扱う方法を検証する。
- 空気は初期 PoC では粒子を満たす気体ではなく、自由表面の外側の周囲として扱う。気体の明示的シミュレーションが必要になった時点で別途設計する。
- 混和、分離、浮沈、粉の沈降、数分間の粒子・質量安定性を個別の合格条件にする。

**コード試作**: `/labs/mpm/kernels` に水・油・粉体の混在シーンを追加した。水・油は
`lambda * min(J - 1, 0) * I` の体積応力だけを持ち、せん断応力を持たない。基準密度と粒子体積から
粒子質量を変える。粉体は圧力に応じて対数主ひずみの偏差を制限する簡易せん断降伏モデルを使う。
GPU particle layoutは密度・体積・摩擦係数を保持し、readbackと複製でも引き継ぐ。

**既知の制限**: 全材料が単一のgrid速度場を共有し、界面張力、非混和性、浮力、粒子接触はない。
したがって水・油の境界維持や浮沈、粉体の液体との接触をこの試作だけで示せるとは限らない。
粉体モデルは圧密・凝集・実材料の校正を含まない。shader compile・GPU/CPU比較・長時間挙動は
WebGPU実機で未確認であり、現段階でGate 4合格とはしない。

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

- [MLS-MPM 論文: The Material Point Method for Simulating Continuum Materials](https://yuanming.taichi.graphics/publication/2018-mlsmpm/mls-mpm-cpic.pdf)
- [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) — MLS-MPM / SPH の WebGPU 参考実装。依存ライブラリではなく、コード採用時は MIT 条件と移植範囲を確認する。
- [jeantimex/fluid](https://github.com/jeantimex/fluid) — WebGPU の FLIP/PIC 比較対象。
- [WebGPU 実装状況](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)
- [Apple Safari 26 Release Notes](https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes)
