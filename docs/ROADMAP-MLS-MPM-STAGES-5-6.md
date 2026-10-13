# 気体・熱・相変化ロードマップ（ステージ5–6）

更新日: 2026-10-13

状態: **設計・調査ロードマップのみ。Stage 5–6のruntime solver、GPU buffer、shader、UIは未実装。**
Stage 1–4の物理受け入れ計画は[別ロードマップ](./ROADMAP-MLS-MPM-STAGES-1-4.md)、共有する前提と6段階の全体像は[全体設計](./DESIGN-LIQUIDFUN-VS-GPU-MPM.md)を参照する。

## 1. 目的と実装開始条件

気泡・気体と熱による相変化を、液体や固体から質量を消したり、phase labelだけを書き換えたりする見た目上の処理にしない。閉じた系でmass、momentum、energyを追跡し、温度による密度・体積変化を一貫させる。

本計画はStage 5（気体・キャビテーション・気泡）とStage 6（熱輸送・潜熱・相変化）を別々の実装・受け入れ単位にする。Stage 5が未完了でも固体–液体の熱伝導研究は独立に進め得るが、**液体–気体の相変化、蒸気泡、沸騰はStage 5の質量移送を通過するまで実装しない**。

次をStage 5 runtime着手前の入口ゲートとする。

- Stage 1–4の統合テストで、実GPUのphase contact・mass accounting・time stepping・bond passを検証済み。
- 対象モバイルGPUのbuffer予算、dispatch数、持続性能を測定済み。
- `material identity` と `physical phase` を別概念に保ち、massをphase切替で変えないデータ設計を合意済み。
- 周囲の空気圧境界だけを扱うか、計算上のgas phaseも持つかを明示的に選べる。

## 2. Stage 5 — 気体・キャビテーション・気泡

### S5.0 モデルを選ぶ前の調査

最初から領域全体にair particleを追加しない。以下の候補を小規模GPU prototypeと資源測定で比較する。

| 対象 | 候補 | 判断材料 |
|---|---|---|
| 領域外の空気 | 外気圧の境界条件、明示的なgas field | 開放境界、気液interfaceへの圧力作用、追加state量 |
| 液中の気泡 | 弱圧縮性gas phase field、cell/binに属するbubble marker、閉じ込め気体のlumped state | `p(V)`応答、移流・変形、分裂/合体、解像度、CPU metadataとGPU state |
| cavitation | 閾値圧でのphase mass flux、局所nucleation＋bubble state | 質量・運動量保存、pressure hysteresis、数値安定性 |

推奨する順序は、まず外気圧をboundary conditionとして定義し、次に**孤立した1気泡**を圧縮・浮上させる最小caseを比べること。気泡の見た目を先に描画しても、圧力・体積を持つgas stateがなければ物理実装とは数えない。

### S5.1 外気圧と小規模gas model

1. gauge pressureとambient pressureの符号・境界条件を定義する。
2. 弱圧縮性EOS候補を1つずつ試し、密度・音速上限とCFL step数を測る。
3. bubble pressure-volume関係、圧縮・復元、周囲液体へ返る法線impulseを小さなcontrolled caseにする。
4. bubble marker方式と連続gas field方式を同じcell size・粒子数で比較し、memory、dispatch、移動・形状再構成を記録する。

**受け入れ条件**: 初期圧力と閉じ込め気体のmassからpressureが再現可能に決まり、圧縮後にboundedな復元が生じる。液体への力は反作用と対になり、NaN/Infinity、負mass、未申告のpressure clampがない。CFL上限を満たせない入力は明示的に拒否・停止する。

### S5.2 bubble lifecycle

1. まず単一bubbleの位置、等価体積、mass、pressureを管理する。
2. 移流・浮力・圧縮をGPU上の既存phase fieldまたはnode-owned couplingへ接続する。
3. 見た目用輪郭と力学用のbubble volume/pressureを別々に検証する。
4. 合体・分裂の粒子/marker topology変更は、安定ID、決定論的prefix-sumまたはGPU gatherで扱う。CPU全粒子readback、float atomic scatterを前提にしない。

**受け入れ条件**: bubbleが液体を貫通せずboundedに移動し、指定pressure-volume関係を保つ。閉じた系でmassが保存され、局所transfer後の各phase massが非負。bubble create/merge/splitの前後でmassとmomentumの差を計測し、許容誤差と境界条件を記録する。複数bubbleの性能は単一bubbleの合格後に測る。

### S5.3 cavitationと液体–気体mass transfer

1. cavitation onset pressure、nucleation hysteresis、最小transfer量、最大transfer率を定義する。
2. liquidからgasへ移す**同じmass**のmomentumをどちらの速度場へ移すか、transfer eventを明示する。
3. gas側へ移した液体を削除しない。mass・momentum・後続Stage 6でのenthalpyをsource/destinationで一度だけ更新する。
4. 圧力境界、自由表面、解像度、time stepを変え、泡の増殖が格子ノイズで暴走しないか検査する。

**受け入れ条件**: 液相と気相の総mass drift、総momentum残差、pressure boundedness、negative mass発生数、bubble lifecycleをGPU-controlled testsで確認する。cavitationを無効化したとき既存のwater/oil結果が変化しない。単に低圧セルを液体から除く実装は不合格とする。

## 3. Stage 6 — 熱輸送・潜熱・相変化

### S6.0 保存変数と熱モデルの決定

Stage 6はtemperatureだけを各粒子へ追加して始めない。物質固有の性質と現在の相を分ける。

```text
MaterialDefinition
  materialId, constitutiveModel, phaseId
  referenceDensity, mechanical/EOS parameters
  heatCapacity, conductivity, latentHeat, transitionRange
  density/volume law, schema version

ParticleThermalState (future)
  mass, specificEnthalpy
  temperature, phaseFraction (derived or versioned)

PhaseTransfer (future)
  sourcePhase, destinationPhase
  transferredMass, transferredMomentum, transferredEnthalpy
  stable transfer key
```

現行particle paddingや摩擦・damage用slotを熱量へ暗黙転用しない。追加stateのbuffer memoryとstorage bindingsは機能別に予算化する。`specificEnthalpy × mass`をenergy accountingの基準にし、温度・phase fractionは材料のenthalpy curveから求める。相ごとのvolumeは `mass / density(T, phase)` から導き、massを書き換えて膨張を表現しない。

### S6.1 heat transportのみ

1. 相変化なしの均質材料と異なる2物質の小規模caseを作る。
2. GPU grid-faceで等量反対向きのheat fluxを使うexplicit conductionと、Jacobi等のimplicit diffusionを比較する。
3. wall/insulated/open boundaryのheat fluxを明示し、mobileで必要なpass数・memory・安定stepを測る。

**受け入れ条件**: 断熱閉領域で総enthalpyが数値許容範囲内に保たれ、均一材料内で温度が拡散・均一化する。境界から熱を加えた場合のenergy changeが与えたheat fluxと一致する。CPU referenceとの比較・GPU版の解像度依存性を記録する。

### S6.2 solid–liquid transition

1. まず材料を一種類（例: 水）に絞り、融解・凝固のenthalpy curve、latent heat、phase fraction区間を設定する。
2. 明示的なphase transferを使い、同じ粒子mass・momentum・enthalpyをsolid/liquid stateへ移す。
3. 温度依存密度・体積変化と外部圧力・容器応答を接続する。相変化で粒子を複製・削除しない。
4. 加熱・冷却の往復、境界熱流束、異なるtime stepでhysteresisやenergy driftを測る。

**受け入れ条件**: 潜熱区間ではenergy投入に応じてphase fractionが変化し、意図しない温度跳躍がない。閉じた系の総massが保存され、総energy変化が境界heat fluxに一致する。体積がdensity lawに従い、膨張・収縮時にmassを生成・喪失しない。融解後の液体がStage 3のinterface/contactと共存するcaseを確認する。

### S6.3 liquid–gas transition（Stage 5依存）

1. Stage 5のgas stateとpressure-volume検証が合格してから、水など対象物質を限定して蒸発・凝縮を加える。
2. phase transferはmass、momentum、enthalpyを同一eventで保守的に移す。
3. closed vessel、開放境界、加熱・冷却でgas pressure、液量、温度、volumeを確認する。

**受け入れ条件**: 気化・凝縮でmass/energy ledgerが閉じ、潜熱plateauとgas EOSのpressure responseが同時に成立する。圧力がCFL/solver限界を越えた場合は停止または限定的な明示エラーとし、黙ってclampしない。

## 4. 実装順序・共通ゲート

1. S5.0の表現比較を調査と短いGPU prototypeで行い、memory・dispatch・CFLを含む採用理由を記録する。
2. Stage 5の外気圧、単一bubble、bubble lifecycle、cavitationを別々のcommit/gateとして検証する。
3. Stage 6はheat transportを先に独立検証し、次にsolid–liquid transitionを追加する。
4. Stage 5のmass transferとgas EOSが合格した後にのみliquid–gas phase changeを接続する。
5. 各gateでCPU小規模reference、GPU deterministic case、保存量診断、実機性能を揃える。資料の数値には環境・粒子数・解像度・step数・許容誤差を付ける。

全Stageで、異なる物質（water/oil等）と同一物質の物理相（solid/liquid/gas）を混同しない。毎frame全粒子readbackを導入しない。負mass、非有限値、未解決のtransfer、capacity超過はユーザーに見える失敗とし、正常終了のように隠さない。

## 5. 現時点の非目標

- 今回のStage 1–4実装へgas field、bubble API、temperature、enthalpy bufferを追加すること。
- 化学反応、溶解・乳化、蒸発潜熱以外の反応熱、3Dの気泡形状を一度に扱うこと。
- 証拠のない物理精度、材質固有の係数、モバイル持続性能を主張すること。
