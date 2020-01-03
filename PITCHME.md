## 深層学習を利用した
## クラスタリング技術


DL4US最終課題

takus

---
## 動機
@snap[text-left]
深層学習の成功に伴い、AIというキーワードで機械学習とデータを活用できないかという機運が高まっている。
しかし、データを集めて活用しようと思っても、アノテーションに時間と手間がかかるという困難がある。
特に深層学習になると必要なデータ量が多く、アノテーションにかかる時間と手間は膨大なものであり、学習の段階まで行けないこともままあるように思われる。

そこで効率的にアノテーションを行うため、深層学習を利用したクラスタリング技術の調査・比較を行う。
従来からあるクラスタリング技術に加えて、深層学習を利用することで、どの程度精度向上が見込めるのかを確認する。

@snapend

---
## クラスタリング技術
@snap[text-left]
深層学習を利用したクラスタリング技術は大きく分けて「AutoEncoders based」「Generative Model Based」「Direct Cluster Optimization」の3種類がある。[1]

今回は以下2つの技術を実装し、クラスタリングの精度を比較した。

- 「AutoEncoders based」のDeep Embedded Clustering(DEC)[2,3]
- 「Generative Model Based」のVariational Deep Embedding(VaDE)[4,5]

またベースラインとして、K-Means[6]とAutoEncoderによる潜在ベクトルをK-Meansでクラスタリングする手法(AE+K-Means)を使用した。

@snapend

---
## Deep Embedded Clustering(DEC)
 1. AutoEncoderにより事前学習し、潜在ベクトルをK-Meansでクラスタリングしセントロイドを算出
 2. サンプルi、クラスタjにおいて、潜在ベクトル`\(z_i\)`とセントロイド`\(\mu_j\)`の差が自由度1のt分布に従うと仮定(予測値)

 `\[
     \tiny
    q_{ij} = \frac{ \left( 1 + |z_i - \mu_j|^2 \right)^{-1}}{ \sum_{j'} \left( 1 + |z_i - \mu_{j'}|^2 \right)^{-1}}
\]`

 3. 正解の分布は予測値を二乗して標準化(正解値)

`\[
    \tiny
    p_{ij} = \frac{ \frac{q_{ij}^2}{f_j} }{ \sum_{j'} \frac{q_{ij'}^2}{f_j'}}
\]`

`\[
    \tiny
    f_j = \sum_i q_{ij}
\]`

 4. 2と3の分布においてカルバック・ライブラー情報量を最小化するように学習

---
## Variational Deep Embedding(VaDE)
AEを事前学習し、潜在ベクトルをGaussian Mixture Model(GMM)でクラスタリング。潜在ベクトルの平均と分散を初期値として算出。Variational Autoencoder(VAE) により復元誤差が最小に、クラスタの事前確率とのカルバック・ライブラー情報量が最大になるように学習(式17)

---
## データと評価指標
@snap[text-left]
### データ
MNISTとFashion MNIST[7]を使用。train, test両方使いデータ件数は70,000

### 評価指標
以下の3指標[1]を使用。10回実行した結果の中央値

1. Unsupervised Clustering Accuracy (ACC)
2. Normalized Mutual Information (NMI)
3. Adjusted Rand Index (ARI)

@snapend

---
## 結果(MNIST)
|モデル|ACC|NMI|ARI|
|---|---|---|---|
|K-Means||||
|AE + K-Means||||
|DEC||||
|VaDE|||

---
## 結果(Fashion MNIST)
|モデル|ACC|NMI|ARI|
|---|---|---|---|
|K-Means|0.47|0.51|0.35|
|AE + K-Means|0.56|0.56|0.41|
|DEC|0.59|0.63|0.47|
|VaDE|0.59|0.59|0.45|

---
## まとめ

---
## 参考文献
1. [Deep Clustering](https://deepnotes.io/deep-clustering)
2. [DEC(論文)](http://proceedings.mlr.press/v48/xieb16.pdf)
3. [DEC(コード)](https://github.com/XifengGuo/DEC-keras)
4. [VaDE(論文)](https://arxiv.org/pdf/1611.05148.pdf)
5. [VaDE(コード)](https://github.com/slim1017/VaDE)
6. [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)
7. [MNIST, Fashion MNIST](https://keras.io/ja/datasets/)

---
