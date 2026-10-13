# インタラクティブ2D GPU MLS-MPMロードマップ（ステージ1–4）

更新日: 2026-10-13

状態: ステージ1–4の操作経路と物理prototypeは実装済み。S3/S4の小規模な制御GPU fixtureは通過したが、scene-scale物理受け入れとS1/S2の実機確認は未完了。現在の推奨実施順は[全体ロードマップ](./ROADMAP-MLS-MPM.md)と[引継ぎ手順](./HANDOFF-MLS-MPM-2026-10-13.md)を参照する。本書のS1–S4の並びは機能別の整理で、直近の作業順を示すものではない。

## 1. 目的と完了の定義

このロードマップは、端末の傾きで重力を操作し、材料を注入・排出し、水と油の界面および弾性固体の破断を試せる2Dサンドボックスを、段階的に「実機で再現可能な挙動」へ近づける。

画面に操作部品や材料色が表示されるだけでは完了としない。各段階で、操作経路、GPU状態の更新、質量・運動量等の物理量、対象端末での実行結果を別々に検証する。現行実装・テスト・未確認事項は [DESIGN-LIQUIDFUN-VS-GPU-MPM.md §8](./DESIGN-LIQUIDFUN-VS-GPU-MPM.md) に記録した。

### 現在の状態

| 段階 | 実装状況 | 確認できたこと | 受け入れ前に残ること |
|---|---|---|---|
| 1. 傾き重力 | センサー変換、許可状態、手動重力fallbackを実装 | センサー変換の単体テスト、手動重力UI | Android/iOS実機で許可・拒否・画面向き・センサー欠落を確認 |
| 2. emitter / outlet | 決定論的batch追加、連続追加、停止、outletと容量報告を実装 | ブラウザーで水64粒子を追加し、受入64・拒否0・容量超過0を確認 | 油・固体・連続停止・容量境界・outlet除去の一連の実行を確認 |
| 3. 水・油界面 | 相別格子状態、相別法線接触、面積場の表示を実装 | 2粒子・8 stepの実GPU接触fixtureでCPU/GPU差3.87e-5、相別mass各0.25、最大運動量残差1.19e-7、全step非交差 | scene-scale界面貫入、層状・斜め界面、任意初期条件での分離、曲率力・表面張力 |
| 4. 固体破断 | 12×8 node・249 bondのGPU damage/broken prototypeを実装 | 4粒子・3 bondの実GPU fixtureで閾値下/上/無負荷を比較しbond 1だけ破断。rest lengthの初期破断stateも2 step保持 | 12×8 sceneへの負荷・流体連成でのpattern、fragment挙動、解像度・性能の評価 |

自動検証の過去の基準点は `npm test` 20 files / 170 tests、`npm run check` 0 errors / 0 warnings、`npm run build` 成功。これは各物理ゲートの合格や、実機WebGPUでの完全な検証を意味しない。その後、作業treeにCFL microstep/recoveryの未commit変更があるため、このテスト数と以下のGPU結果は変更前sourceの記録であり、現行treeでは再実行が必要。
変更前sourceでは、2026-10-13に14,696粒子sceneの実GPU benchmarkを5回実行し、GPU step medianの中央値0.40109 ms/step、p95中央値0.43035 ms/stepを記録した。小規模S3/S4 GPU fixtureの結果もCPU referenceとの制御比較であり、scene-scaleや長時間の受け入れではない。詳細は[性能計測資料](./PERFORMANCE-MLS-MPM.md)を参照する。

## 2. Stage別の実装状況と受け入れ条件

### S1 — 傾き重力を端末で受け入れる

**実装済み**: DeviceOrientationのsampleをscreen axes上の2D accelerationへ変換し、iOSのpermission request、`enabled` / `denied` / `unsupported`状態、手動X/Y重力fallbackを用意した。シミュレーション場そのものは回転しない。

**残作業**

1. Android ChromeとiOS SafariのHTTPS環境で、センサー許可操作を個別に確認する。
2. 水平、前後・左右の傾き、portrait/landscape、拒否、API非対応、イベント未着を記録する。
3. センサーの値が停止・再開・速度倍率変更後も一貫して反映されることを確認する。
4. センサーを許可しない場合でも手動重力の操作だけでシミュレーションを継続できることを確認する。

**受け入れ条件**: 上記各入力から期待する符号・大きさの2D重力が得られ、画面を回さず粒子の運動方向が変わる。許可拒否・未対応で例外や操作不能状態にならない。端末・OS・ブラウザー・画面向きごとの結果を記録する。

### S2 — emitter / outletを操作として成立させる

**実装済み**: seed、形状、材料、粒子数、位置、間隔、初速を指定するsource、repeatableな一回追加と連続追加、停止、液体用outlet、受入・拒否・容量超過の件数表示を実装した。GPU stateが正本で、outletで削除したslotは現状再利用しない。

**残作業**

1. 同一seed・同一burst番号で、材料ごとの位置と初速が再現することを検証する。
2. 連続追加を開始・停止し、停止後に新規slotが増えないことを確認する。
3. 水・油・Neo-Hookean固体の追加を個別に行い、材料・形状・指定速度がGPU stateへ反映されることを確認する。
4. 残容量未満、ちょうど容量、容量超過、範囲外・不正入力を試し、成功数と拒否数を一致させる。
5. outletが水・油だけを除去し、固体を残すこと、active countと予約済みslotが区別されることを確認する。
6. GPUの両ping-pong state、append境界、outlet後のpass順序を実WebGPUで確認する。容量不足を隠すための成功扱いはしない。

**受け入れ条件**: 各操作後の件数がGPU stateと一致し、決定論的再現性、材料別の追加、停止動作、outletの対象限定が成立する。不正入力と容量超過は明示され、out-of-bounds write、NaN/Infinity、固体の誤削除がない。slot非再利用のため長時間使用時に容量が枯渇する挙動をUIで説明し、再利用/compactionは別の性能・安全性計画なしに導入しない。

### S3 — 水・油のinterfaceを物理量で評価する

**実装済み**: 水・油を異なるphase/materialとして保ち、mass・momentum・volume・node velocityを相別に集計する。混在nodeの法線contactはnode-ownedで行い、面積場は自由表面の可視化に使う。表示用占有場はcontactや圧力の代替ではない。

**S3.1 現行contactのGPU受け入れ**

2026-10-13の制御fixtureでは、既存CPU referenceと同じ接近中の水・油2粒子を8 step実行した。
各stepのGPU readbackでphase identity、相別mass各0.25、最大運動量残差`1.1921e-7`、
全step非交差を確認し、CPU/GPU状態の最大絶対差は`3.8653e-5`だった。
状態比較の`1e-3`は数値回帰の許容値であり、
製品品質の貫入閾値ではない。

**残作業**

1. 静止した層状interface、斜めinterface、複数密度・速度差のfixtureを追加する。
2. これらの配置でinterface crossing/penetration、押し戻し、反発振動、空隙、数値拡散を複数step計測する。
3. scene-scaleの相別mass・momentumと物理的に許容できる貫入・分離基準を決める。
4. 任意初期条件での混和・分離を確認する。CPU referenceとの比較だけでなくGPU結果も測る。

**S3.2 表面張力を追加する判定と実装**

現状の相別法線接触だけでは、油滴のまとまりや界面曲率による復元力を保証しない。S3.1後にGPU占有場の法線・曲率推定を評価し、表面張力を有効化する条件を決める。追加する場合は、水–空気、油–空気、水–油の係数を分け、表示用の平滑化と物理用interface fieldを分離する。wetting/contact angleは表面張力とは別の後続機能とする。

**受け入れ条件**: 制御ケースで相ごとの質量変化、総運動量残差、interface crossing、液滴形状の変化を測定し、試験条件・許容誤差・step数を固定して記録する。表面張力を実装する場合、2D静止円の圧力差 `Δp ≈ γ/R`、静止時のspurious current、解像度を変えた傾向を検査する。見た目の色分けやCPUだけのテストでは合格としない。完全な混和防止や任意解像度での物理忠実性は別途主張しない。

### S4 — 固体の損傷・破断を再現可能にする

**実装済みprototype**: 12×8 nodeのconnected Neo-Hookean bodyと249本のbondを持つ。引張strainの探索閾値0.12を用いてGPU上でdamage/brokenを更新し、damageを可視化する。破断後はbond stateを維持し、bond bodyのcloneは拒否する。

2026-10-13の制御GPU fixtureでは、failure strain`0.1`に対し入力strain`0.05 / 0.11 / 0`の3 bondを2 step実行し、
閾値上のbond index 1だけがbroken、閾値下のbond damageは約`0.5`、無負荷bondは未破断となった。
粒子状態の最大CPU/GPU差は`7.63e-7`、damage差は`7.63e-6`以下、質量4・運動量残差0。
別fixtureではrest lengthの初期broken bondが2 step後もdamage 1で維持された。
このfixtureの閾値0.1は式の確認用であり、scene prototypeの0.12を素材強度として校正した結果ではない。

**残作業**

1. 12×8 bodyに負荷を段階的に加え、期待pattern・damage局所化・fragmentの相対移動をGPUで確認する。
2. 破断前後のmass/momentum、有限値、grid境界をこのsceneで確認する。
3. 粗密複数の解像度、step幅、材料係数でfailure patternを比較し、threshold、stress fade、bond stiffnessを分離して校正する。
4. 水・油の接触荷重を加えたcaseを作り、contact impulseとfracture passの順序・結果を比較する。
5. target mobile device上でparticle/bond数、buffer容量、持続時間、FPS/step timeを測る。1M particle capacityの確保量だけで性能を主張しない。

**受け入れ条件**: 閾値の上下で結果が再現し、局所damageが安定して伝播し、broken bondが復活しない。破断が質量消失・NaN・境界逸脱を起こさない。解像度ごとの結果と限界を公開する。`0.12`は現状の配置用探索値であり、素材固有の強度・fracture energyとは呼ばない。

### S1–S4統合ゲート

個別テスト通過後に、センサーまたは手動重力、連続emitter、水油interface、固体への流体荷重、破断、outletを同一sceneで順に操作する。

- Android/iOSの対象実機・ブラウザー、画面向き、許可状態、GPU/driverを記録する。
- 14,696粒子の初期sceneと、追加後の代表粒子数の両方で連続動作・停止・再開を行う。
- 実WebGPU shader compile/dispatch、数値status、GPU parity、renderとcomputeを確認する。CPU referenceやfake-GPU testを実機結果として扱わない。
- mobileでの持続FPS、処理時間、メモリ、熱、容量上限を測り、目標未達なら容量・pass・sceneを調整して再測定する。
- 各gateの結果と未確認を記録してからStage 5–6のruntime設計に進む。

## 3. ゲーム規模へ進む前の接触拡張性ゲート

現在の衝突方式は少数の三角形collider向けprototypeである。grid node数 `G=34,119`、
collider数 `B=200` では、三角形field構築は最大 `G·B ≈ 6.82M` node/body候補/substep、
小石同士のSAT候補列挙は `B·(B−1) ≈ 39,800` pair/substepとなる。1,000 colliderでは
約34.12M node/body候補と999k pair候補まで増える。AABB rejectionは詳細判定を省くが、
現在は候補生成自体を空間的に絞っていない。実測条件と式は
[性能計測資料](./PERFORMANCE-MLS-MPM.md)、最適化案は
[性能チューニングロードマップP5](./ROADMAP-MLS-MPM-PERFORMANCE.md)を参照。

大きな構造物を多数particleで表す負荷と、三角形collider数を増やす負荷は分けて測る。
内部fracture graphが疎ならbond updateは概ね `O(P + E)` だが、構造表面を多数の三角形へ
分解して全node/all-pair探索へ渡すと再び `O(G·B)` / `O(B²)` に近づく。

**製品規模へ進む前の作業**

1. 200 / 400 / 800 / 1,000以上のcollider、並びに96 / 384 / 1,536以上のparticleから成る
   疎な大構造を作り、grid・step・seedを固定したsparse sceneとdense sceneを計測する。
2. node/body候補数、body pair数、詳細判定数、cell list長とoverflow、contact pass p50/p95、
   CFL substep数、全体frame時間、memoryをGPUで記録する。
3. uniform grid body-cell listでtriangle fieldを局所候補化し、body pairをcanonical unique pairに
   絞ってからSATを行う。大型colliderはhierarchical grid/BVH/object-level surface proxyを比較する。
4. 小規模brute-force結果をoracleとし、接触漏れ、double impulse、penetration、
   momentum residual、順序依存を比較する。dense worst caseの `O(B²)` 残存を隠さない。
5. target mobile GPUで、ユーザーが増やす物体数・構造解像度に対する上限を測定から定め、
   上限超過は明示的に拒否する。

**受け入れ条件**: sparse caseで候補数が全node×bodyと全body pairから局所参照数へ下がる。
brute-force oracleに対する接触漏れがなく、overflowを黙って無接触扱いにしない。dense caseを
含む最終sceneのp95時間・buffer memoryがtarget mobile予算内に入り、許容scene budgetを明記する。
このゲートは現在未実装であり、1M粒子buffer capacityを物体数の性能保証として扱わない。

## 4. Stage 5–6へ進む前の停止条件

Stage 1–4の実装が存在しても、S1–S4統合ゲート未通過なら気体・熱とのcouplingを追加しない。特に、相別mass/momentumのGPU確認、安定したtime stepping、有限容量・エラー通知、実機での最低限の持続測定が済むまでは、新しいphase fieldや温度bufferを粒子layoutへ追加しない。
