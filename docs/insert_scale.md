# `tasks/airec/reach_bracelet.py` 実装メモ

このドキュメントは、`tasks/airec/reach_bracelet.py` におけるブレスレット開口判定と、それを用いた報酬計算の整理です。

主に以下の 2 つの量を扱います。

| シンボル | 意味 |
|---|---|
| `inside_opening_soft` | 手首がブレスレット開口断面の楕円内にあるほど 1 に近づくソフトゲート |
| `fingers_inside_opening_soft` | 5 指先が同じ楕円基準で開口内にある度合いの重み付き平均 |

これらは `compute_rewards` 内で深度報酬や手首センタリング報酬に掛けられます。  
特に、両手の距離 `ee_euclidean_distance` が `ee_distance_threshold` 未満のときだけ、深度・センタリング系の報酬が有効になります。

---

## 1. 座標系と前提

`goal_*_pos`, `goal_wrist_pos`, `thumb_target` などは env-local 座標です。

剛体ブレスレットの開口フレームには、剛体根のワールド姿勢

```math
q_{\mathrm{open}} = \texttt{object.data.root_quat_w}
```

を用いて変換します。

点 $\mathbf{p}^{\mathrm{env}}$ を開口座標へ変換する式は次の通りです。

```math
\mathbf{p}^{\mathrm{open}}
=
R(q_{\mathrm{open}})^{\top}
\left(
\mathbf{p}^{\mathrm{env}} - \mathbf{c}
\right),
\qquad
\mathbf{c} = \mathrm{goal\_cent\_pos}
```

実装では以下に対応します。

```python
p_rel = point - goal_cent_pos
p_open = quat_apply_inverse(open_quat_w, p_rel)
```

開口ローカル座標では、以下のように扱います。

| 軸 | 意味 |
|---|---|
| x | 開口断面の横方向 |
| y | 開口断面の縦方向 |
| z | 挿入深さ方向 |

手首の開口座標は以下です。

```math
(x, y, z) = \mathrm{wrist\_in\_open}[env, 0:3]
```

---

## 2. 主要パラメータ

### 2.1 `ReachBraceletEnvCfg` で設定する値

| 名前 | 型 | 既定値 | 役割 |
|---|---:|---:|---|
| `bracelet_desired_insert_depth` | `float` | `0.0` | 開口ローカル z 方向の目標挿入深さ |
| `bracelet_inside_opening_std` | `float` | `0.15` | 楕円外に出たときの指数減衰スケール |
| `bracelet_rim_offset_*` | `tuple[float, float, float]` | N/S/E/W 各オフセット | 剛体根フレームから見たリム点。楕円半径の計算に使用 |

`bracelet_inside_opening_std` は、開口外に出たときのゲートの落ち方を決めます。  
値が大きいほど、楕円外でも報酬が落ちにくくなります。

---

### 2.2 `compute_rewards` 内のローカル定数

| 名前 | 値 | 用途 |
|---|---:|---|
| `depth_reward_scale` | `5.0` | 手首深度報酬 `r_depth_distance` の倍率 |
| `depth_thumb_reward_scale` | `0.5` | 親指深度報酬 `r_depth_thumb_distance` の倍率 |
| `depth_pinky_reward_scale` | `0.5` | 小指深度報酬 `r_depth_pinky_distance` の倍率 |
| `ee_distance_threshold` | `0.3` | 両手距離がこの値未満のとき、深度・センタリング報酬を有効化 |

`distance_reward(d, std)` に使われる `std` は以下です。

| 報酬対象 | `std` |
|---|---:|
| 手首深度 | `0.1` |
| 親指深度 | `0.03` |
| 小指深度 | `0.06` |
| 手首 XY センタリング | `0.04` |
| 手首 3D センタリング | `0.16` |

---

### 2.3 ゼロ除算防止用の下限値

楕円半径が 0 になることを避けるため、以下の値を使います。

```math
\epsilon = 10^{-4}
```

実装上は `rad_eps = 1e-4` です。

---

### 2.4 指ごとの重み

`fingers_inside_opening_soft` は、5 指のゲート値の重み付き平均です。

| 指 | 重み |
|---|---:|
| thumb | `0.20` |
| fore | `0.25` |
| middle | `0.30` |
| ring | `0.25` |
| pinky | `0.10` |

重みの合計は 1.10 です。  
そのため、厳密には `fingers_inside_opening_soft` は最大 1.10 になり得ます。  
「0 から 1 のゲート」として使いたいなら、重み合計を 1.00 に正規化する必要があります。ここ、地味に見落とすと報酬設計が静かにズレます。

---

## 3. テンソル形状

バッチサイズを $N = \texttt{num\_envs}$、部分更新時の環境数を $M = |\texttt{env\_ids}|$ とします。

| シンボル | 形状 | dtype | 更新箇所 |
|---|---:|---|---|
| `wrist_in_open` | `(N, 3)` | float | `_compute_intermediate_values` |
| `east_in_open` | `(N, 3)` | float | `_compute_intermediate_values` |
| `west_in_open` | `(N, 3)` | float | `_compute_intermediate_values` |
| `north_in_open` | `(N, 3)` | float | `_compute_intermediate_values` |
| `south_in_open` | `(N, 3)` | float | `_compute_intermediate_values` |
| `thumb_goal_pos` ... `pinky_goal_pos` | `(N, 3)` | float | FrameTransformer 由来 |
| `thumb_tip_o` ... `pinky_tip_o` | `(M, 3)` | float | 剛体ブロック内の一時変数 |
| `radius_x`, `radius_y` | `(M,)` | float | 楕円半径 |
| `radial_normalized` | `(M,)` | float | `wrist_radial_normalized` に格納 |
| `inside_opening_soft` | `(N,)` | float | 手首開口ゲート |
| `thumb_radial_normalized` ... `pinky_radial_normalized` | `(N,)` | float | 各指の正規化径向二乗 |
| `fingers_inside_opening_soft` | `(N,)` | float | 指先開口ゲートの重み付き和 |

---

## 4. 計算式

以下は、`object_type == "rigid"` かつ `object` が存在する場合の計算です。

---

### 4.1 楕円半径

開口断面の x 方向半径を $r_x$、y 方向半径を $r_y$ とします。

```math
r_x
=
\max
\left(
\frac{1}{2}
\left|
e^{\mathrm{open}}_x
-
w^{\mathrm{open}}_x
\right|,
\epsilon
\right)
```

```math
r_y
=
\max
\left(
\frac{1}{2}
\left|
n^{\mathrm{open}}_y
-
s^{\mathrm{open}}_y
\right|,
\epsilon
\right)
```

ここで、

- $e^{\mathrm{open}}$ は `east_in_open`
- $w^{\mathrm{open}}$ は `west_in_open`
- $n^{\mathrm{open}}$ は `north_in_open`
- $s^{\mathrm{open}}$ は `south_in_open`

です。

つまり、x 半径は east-west の x 座標差、y 半径は north-south の y 座標差から計算します。

---

### 4.2 手首の正規化径向二乗

手首位置を

```math
(x, y, z) = \mathrm{wrist\_in\_open}
```

としたとき、手首の正規化径向二乗 $R_{\mathrm{wrist}}$ は次の通りです。

```math
R_{\mathrm{wrist}}
=
\left(
\frac{x}{r_x}
\right)^2
+
\left(
\frac{y}{r_y}
\right)^2
```

この値が 1 以下なら、手首は楕円内にあります。

```math
R_{\mathrm{wrist}} \leq 1
\quad
\Longleftrightarrow
\quad
\text{inside ellipse}
```

---

### 4.3 手首ゲート `inside_opening_soft`

楕円外に出た量を次のように定義します。

```math
\mathrm{outside}
=
\max
\left(
R_{\mathrm{wrist}} - 1,\ 0
\right)
```

手首ゲートは次の式で計算します。

```math
\mathrm{inside\_opening\_soft}
=
\exp
\left(
-
\frac{\mathrm{outside}}{\sigma}
\right)
```

ただし、

```math
\sigma
=
\max
\left(
\mathrm{bracelet\_inside\_opening\_std},\ 10^{-6}
\right)
```

です。

性質は以下です。

| 条件 | `outside` | `inside_opening_soft` |
|---|---:|---:|
| $R_{\mathrm{wrist}} \leq 1$ | `0` | `1` |
| $R_{\mathrm{wrist}} > 1$ | 正の値 | 0 に近づく |

つまり、楕円内では完全に 1、楕円外では指数的に減衰します。

---

### 4.4 各指先のゲート

各指先 $d$ について、開口座標を

```math
\mathbf{p}^{\mathrm{open}}_d
=
\left(
p^{\mathrm{open}}_{d,x},
p^{\mathrm{open}}_{d,y},
p^{\mathrm{open}}_{d,z}
\right)
```

とします。

各指の正規化径向二乗 $R_d$ は次の通りです。

```math
R_d
=
\left(
\frac{p^{\mathrm{open}}_{d,x}}{r_x}
\right)^2
+
\left(
\frac{p^{\mathrm{open}}_{d,y}}{r_y}
\right)^2
```

各指のソフトゲートは次のように計算されます。

```math
\texttt{inside}_d
=
\exp
\left(
-
\frac{
\max
\left(
R_d - 1,\ 0
\right)
}{\sigma}
\right)
```

この式も手首と同じで、楕円内では 1、外に出るほど 0 に近づきます。

---

### 4.5 指全体のゲート `fingers_inside_opening_soft`

5 指のゲート値を重み付き和でまとめます。

```math
\begin{aligned}
\mathrm{fingers\_inside\_opening\_soft}
=&\ 0.20\,\texttt{inside}_{\mathrm{thumb}} \\
&+ 0.25\,\texttt{inside}_{\mathrm{fore}} \\
&+ 0.30\,\texttt{inside}_{\mathrm{middle}} \\
&+ 0.25\,\texttt{inside}_{\mathrm{ring}} \\
&+ 0.10\,\texttt{inside}_{\mathrm{pinky}}
\end{aligned}
```

注意点として、重み合計は 1.10 です。

```math
0.20 + 0.25 + 0.30 + 0.25 + 0.10 = 1.10
```

そのため、全指が楕円内にある場合、

```math
\mathrm{fingers\_inside\_opening\_soft} = 1.10
```

になります。

---

## 5. object type ごとの分岐

| 条件 | `inside_opening_soft` | `fingers_inside_opening_soft` |
|---|---:|---:|
| 剛体ブレスレット | 式どおり | 式どおり |
| `_use_glove` かつ非剛体リムパス | `1.0` | `1.0` |
| リムなし、またはその他 | `0.0` | `0.0` |

剛体ブレスレット以外では、楕円判定を実質的に無効化しています。

---

## 6. `compute_rewards` での使われ方

距離報酬は共通して以下の形です。

```math
\mathrm{distance\_reward}(d, \mathrm{std})
=
1 - \tanh
\left(
\frac{d}{\mathrm{std}}
\right)
```

---

### 6.1 深度報酬

手首の深度距離は、開口ローカル z 座標と目標挿入深さの差で計算されます。

```math
\mathrm{depth\_distance}
=
\left|
z_{\mathrm{wrist}}
-
\mathrm{bracelet\_desired\_insert\_depth}
\right|
```

手首深度報酬は次の形です。

```math
\begin{aligned}
r_{\mathrm{depth}}
=&\
\mathrm{distance\_reward}
\left(
\mathrm{depth\_distance},\ 0.1
\right) \\
&\times
\mathrm{inside\_opening\_soft} \\
&\times
\mathrm{fingers\_inside\_opening\_soft} \\
&\times
\mathrm{depth\_reward\_scale} \\
&\times
\mathbb{I}
\left[
\mathrm{ee\_euclidean\_distance}
<
\mathrm{ee\_distance\_threshold}
\right]
\end{aligned}
```

この報酬だけ、`inside_opening_soft` と `fingers_inside_opening_soft` の両方が掛かります。

---

### 6.2 親指・小指の深度報酬

親指と小指の深度報酬には、`inside_opening_soft` は掛かりません。  
掛かるのは `fingers_inside_opening_soft` です。

親指側は概念的に次の形です。

```math
\begin{aligned}
r_{\mathrm{thumb\_depth}}
=&\
\mathrm{distance\_reward}
\left(
\texttt{depth\_thumb},\ 0.03
\right) \\
&\times
\mathrm{fingers\_inside\_opening\_soft} \\
&\times
\mathbb{I}
\left[
\texttt{top\_height}
>
\texttt{thumb\_height}
>
\texttt{bottom\_height}
\right] \\
&\times
\mathrm{depth\_thumb\_reward\_scale} \\
&\times
\mathbb{I}
\left[
\mathrm{ee\_euclidean\_distance}
<
\mathrm{ee\_distance\_threshold}
\right]
\end{aligned}
```

小指側も同様で、`std = 0.06` と `depth_pinky_reward_scale` を使います。

---

### 6.3 手首センタリング報酬

手首の XY センタリング報酬は次の形です。

```math
\begin{aligned}
r_{\mathrm{wrist\_center\_xy}}
=&\
\mathrm{distance\_reward}
\left(
\texttt{wrist\_xy\_center},\ 0.04
\right) \\
&\times
\mathrm{wrist\_center\_alignment\_scale} \\
&\times
\mathrm{fingers\_inside\_opening\_soft} \\
&\times
\mathbb{I}
\left[
\mathrm{ee\_euclidean\_distance}
<
\mathrm{ee\_distance\_threshold}
\right]
\end{aligned}
```

手首の 3D センタリング報酬も同様で、`std = 0.16` を使います。

ここでも `inside_opening_soft` は掛かりません。

---

## 7. 高さとして渡されている値

`_get_rewards` から `compute_rewards` に渡している height 系の引数は、実際には開口フレームの y 成分です。

| 引数名 | 実際のテンソル |
|---|---|
| `wrist_height` | `wrist_in_open[:, 1]` |
| `top_height` | `north_in_open[:, 1]` |
| `bottom_height` | `south_in_open[:, 1]` |
| `thumb_height` | `thumb_in_open[:, 1]` |
| `pinky_height` | `pinky_in_open[:, 1]` |

注意点として、`thumb_in_open` は `thumb_target` を開口座標に変換したものです。  
これは実指先である `thumb_goal_pos` とは別です。

深度用の指ゲートでは、`thumb_goal_pos` などから作られる `thumb_tip_o` 側を使います。

---

## 8. ログ出力

`extras["log"]` には、以下のような値が記録されます。

| キー | 内容 |
|---|---|
| `inside_opening_soft` | 手首の開口ゲート |
| `fingers_inside_opening_soft` | 指全体の開口ゲート |
| `wrist_radial_normalized` | 手首の正規化径向二乗 |
| `thumb_radial_normalized` | 親指の正規化径向二乗 |
| `fore_radial_normalized` | 人差し指の正規化径向二乗 |
| `middle_radial_normalized` | 中指の正規化径向二乗 |
| `ring_radial_normalized` | 薬指の正規化径向二乗 |
| `pinky_radial_normalized` | 小指の正規化径向二乗 |

TensorBoard などで見る場合は、まず `inside_opening_soft`, `fingers_inside_opening_soft`, `wrist_radial_normalized` を確認すると挙動を追いやすいです。

---

## 9. 調整指針

| 目的 | 変更箇所 |
|---|---|
| 楕円外でも報酬を落としすぎない | `bracelet_inside_opening_std` を大きくする |
| 楕円外に出たら強く罰したい | `bracelet_inside_opening_std` を小さくする |
| 開口断面サイズを変えたい | `bracelet_rim_offset_*` を調整する |
| 指ごとの寄与を変えたい | `fingers_inside_opening_soft` の重みを変更する |
| 手首深度報酬を強めたい | `depth_reward_scale` を大きくする |
| 親指・小指の深度報酬を強めたい | `depth_thumb_reward_scale`, `depth_pinky_reward_scale` を大きくする |
| 距離に対して報酬を鋭敏にしたい | `distance_reward` の `std` を小さくする |
| 距離に対して報酬を緩やかにしたい | `distance_reward` の `std` を大きくする |

---

## 10. 実装上の注意点

### 10.1 `fingers_inside_opening_soft` は最大 1.10

現在の重み合計は 1.10 です。

この設計が意図的なら問題ありません。  
ただし、「ゲート値は最大 1」という前提で報酬を設計している場合は、以下のように重みを正規化した方が安全です。

```python
fingers_inside_opening_soft = (
    0.20 * thumb_inside
    + 0.25 * fore_inside
    + 0.30 * middle_inside
    + 0.25 * ring_inside
    + 0.10 * pinky_inside
) / 1.10
```

または、重み自体の合計を 1.00 に調整します。

---

### 10.2 `inside_opening_soft` が掛かる報酬は限定的

`inside_opening_soft` が直接掛かるのは、手首深度報酬 `r_depth_distance` だけです。

親指深度、小指深度、手首センタリング系には `inside_opening_soft` は掛からず、`fingers_inside_opening_soft` のみが掛かります。

ここを混同すると、「手首が楕円外なのにセンタリング報酬が残る」ように見える可能性があります。  
これはバグとは限らず、現在の式の仕様です。

---

### 10.3 height は z ではなく y

`top_height`, `bottom_height`, `thumb_height`, `pinky_height` は名前に `height` とありますが、実体は開口フレームの y 成分です。

```python
top_height = north_in_open[:, 1]
bottom_height = south_in_open[:, 1]
```

z 方向の挿入深さとは別なので、読むときに混ぜないように注意してください。名前で油断すると、コードが人間に罠を仕掛けてきます。

---

## 11. まとめ

この実装では、ブレスレット開口断面を楕円として近似し、手首と指先がその楕円内にあるかをソフトゲートとして評価します。

中心となる設計は以下です。

1. 開口フレームに点を変換する
2. east-west, north-south から楕円半径を求める
3. 手首と各指先について、正規化径向二乗を計算する
4. 楕円外に出た分だけ指数的にゲートを減衰させる
5. そのゲートを深度報酬やセンタリング報酬に掛ける

特に確認すべき点は以下です。

- `fingers_inside_opening_soft` の重み合計が 1.10 であること
- `inside_opening_soft` が掛かる報酬は `r_depth_distance` に限定されること
- `height` 系の値は開口フレームの y 成分であり、z 方向深度ではないこと

この 3 点を把握しておけば、報酬が意図せず強すぎる・弱すぎる・残りすぎる原因をかなり絞り込めます。
