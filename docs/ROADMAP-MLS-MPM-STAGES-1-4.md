# インタラクティブ2D GPU MLS-MPMロードマップ（ステージ1–4）

更新日: 2026-10-13

状態: ステージ1–4の操作経路と物理prototypeは実装済み。自動テストと一部ブラウザー操作は確認したが、物理受け入れゲートは未完了。設計境界は[全体設計](./DESIGN-LIQUIDFUN-VS-GPU-MPM.md)を参照する。

## 1. 目的と完了の定義

このロードマップは、端末の傾きで重力を操作し、材料を注入・排出し、水と油の界面および弾性固体の破断を試せる2Dサンドボックスを、段階的に「実機で再現可能な挙動」へ近づける。

画面に操作部品や材料色が表示されるだけでは完了としない。各段階で、操作経路、GPU状態の更新、質量・運動量等の物理量、対象端末での実行結果を別々に検証する。現行実装・テスト・未確認事項は [DESIGN-LIQUIDFUN-VS-GPU-MPM.md §8](./DESIGN-LIQUIDFUN-VS-GPU-MPM.md) に記録した。

### 現在の状態

| 段階 | 実装状況 | 確認できたこと | 受け入れ前に残ること |
|---|---|---|---|
| 1. 傾き重力 | センサー変換、許可状態、手動重力fallbackを実装 | センサー変換の単体テスト、手動重力UI | Android/iOS実機で許可・拒否・画面向き・センサー欠落を確認 |
| 2. emitter / outlet | 決定論的batch追加、連続追加、停止、outletと容量報告を実装 | ブラウザーで水64粒子を追加し、受入64・拒否0・容量超過0を確認 | 油・固体・連続停止・容量境界・outlet除去の一連の実行を確認 |
| 3. 水・油界面 | 相別格子状態、相別法線接触、面積場の表示を実装 | 小規模CPU referenceで相identity、相別mass、接触momentumを検査 | 実GPUでの反復界面、任意初期条件での分離、曲率力・表面張力 |
| 4. 固体破断 | 12×8 node・249 bondのGPU damage/broken prototypeを実装 | 閾値・永続性等のCPU reference、GPU pipeline/layoutのfake tests | 実GPU破断試験、流体荷重を含む再現性、解像度・性能の評価 |

自動検証の基準点は `npm test` 20 files / 170 tests、`npm run check` 0 errors / 0 warnings、`npm run build` 成功。これは各物理ゲートの合格や、実機WebGPUでの完全な検証を意味しない。

## 2. 実装と検証の順序

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

1. 静止した層状interface、接近する水・油、斜めinterfaceを少数粒子の再現可能なfixtureにする。
2. GPU実行前後に相identity・相別mass・total momentumを測り、誤差許容値をfloat32基準試験から定めて記録する。
3. 位置順序だけでなく、interface crossing/penetration、押し戻し、反発振動、空隙、界面の数値拡散を複数stepで計測する。
4. CPU referenceとの比較で、水・油それぞれの速度と接触impulseを確認する。

**S3.2 表面張力を追加する判定と実装**

現状の相別法線接触だけでは、油滴のまとまりや界面曲率による復元力を保証しない。S3.1後にGPU占有場の法線・曲率推定を評価し、表面張力を有効化する条件を決める。追加する場合は、水–空気、油–空気、水–油の係数を分け、表示用の平滑化と物理用interface fieldを分離する。wetting/contact angleは表面張力とは別の後続機能とする。

**受け入れ条件**: 制御ケースで相ごとの質量変化、総運動量残差、interface crossing、液滴形状の変化を測定し、試験条件・許容誤差・step数を固定して記録する。表面張力を実装する場合、2D静止円の圧力差 `Δp ≈ γ/R`、静止時のspurious current、解像度を変えた傾向を検査する。見た目の色分けやCPUだけのテストでは合格としない。完全な混和防止や任意解像度での物理忠実性は別途主張しない。

### S4 — 固体の損傷・破断を再現可能にする

**実装済みprototype**: 12×8 nodeのconnected Neo-Hookean bodyと249本のbondを持つ。引張strainの探索閾値0.12を用いてGPU上でdamage/brokenを更新し、damageを可視化する。破断後はbond stateを維持し、bond bodyのcloneは拒否する。

**残作業**

1. 負荷を段階的に増やす決定論的caseで、閾値未満では未破断、閾値超過では想定bondだけが破断することをGPUで確認する。
2. damageが隣接粒子にだけ伝わること、破断後にbondが再生成されないこと、fragmentが相対移動できることを確認する。
3. 外力が既知の小規模caseで、粒子数・質量・総運動量・有限値・grid境界を確認する。
4. 1種類のnode間隔だけでなく粗密複数の解像度、step幅、材料係数でfailure patternを比較する。threshold、stress fade、bond stiffnessを分離して校正する。
5. 水・油の接触荷重を加えたcaseを作り、contact impulseとfracture passの順序・結果を比較する。
6. target mobile device上でparticle/bond数、buffer容量、持続時間、FPS/step timeを測る。1M particle capacityの確保量だけで性能を主張しない。

**受け入れ条件**: 閾値の上下で結果が再現し、局所damageが安定して伝播し、broken bondが復活しない。破断が質量消失・NaN・境界逸脱を起こさない。解像度ごとの結果と限界を公開する。`0.12`は現状の配置用探索値であり、素材固有の強度・fracture energyとは呼ばない。

### S1–S4統合ゲート

個別テスト通過後に、センサーまたは手動重力、連続emitter、水油interface、固体への流体荷重、破断、outletを同一sceneで順に操作する。

- Android/iOSの対象実機・ブラウザー、画面向き、許可状態、GPU/driverを記録する。
- 14,696粒子の初期sceneと、追加後の代表粒子数の両方で連続動作・停止・再開を行う。
- 実WebGPU shader compile/dispatch、数値status、GPU parity、renderとcomputeを確認する。CPU referenceやfake-GPU testを実機結果として扱わない。
- mobileでの持続FPS、処理時間、メモリ、熱、容量上限を測り、目標未達なら容量・pass・sceneを調整して再測定する。
- 各gateの結果と未確認を記録してからStage 5–6のruntime設計に進む。

## 3. Stage 5–6へ進む前の停止条件

Stage 1–4の実装が存在しても、S1–S4統合ゲート未通過なら気体・熱とのcouplingを追加しない。特に、相別mass/momentumのGPU確認、安定したtime stepping、有限容量・エラー通知、実機での最低限の持続測定が済むまでは、新しいphase fieldや温度bufferを粒子layoutへ追加しない。
