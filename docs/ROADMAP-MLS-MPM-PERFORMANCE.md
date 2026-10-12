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
14,600粒子・200個の小石を使う現シーンで同じA/B条件を再測定した結果ではない。
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

## 後続: 今回実装しない調査

- mobile で capacity と機能 profile を見直し、５分以上の持続性能を測る。
- dense node の list 長・偏りを収集して sort/tile gather を比較する。
- active node/tile と full-grid の crossover を測る。
- physics と render の decoupling、queue gate、pick の大粒子数 search を比較する。
- SoA、workgroup size、subgroup、`f16` は精度と対応端末を確認するまで採用しない。
- CFL と体積保存・界面機能は材料ロードマップで別に進める。

## ソース訂正前チェック

- [x] 計測する指標・観点・固定条件を記録。
- [x] 現行コード・実 GPU baseline と数値基準を確認。
- [x] 実装内容、対象、メモリ／limits、回帰条件、採否基準を確定。
- [x] この計画を起点に P1/P2/P3 のソース変更をまとめて実施。
- [x] 結果に基づいて P4 の採否と文書更新を実施。
