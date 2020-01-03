### 深層学習を利用したクラスタリング技術
DL4US最終課題

takus

---
### 動機

深層学習の成功に伴い、AIというキーワードで機械学習とデータを活用できないかという機運が高まっている。
しかし、データを集めて活用しようと思っても、アノテーションに時間と手間がかかるという困難がある。
特に深層学習になると必要なデータ量が多く、アノテーションにかかる時間と手間は膨大なものであり、学習の段階まで行けないこともままあるように思われる。

そこで効率的にアノテーションを行うため、深層学習を利用したクラスタリング技術の調査・比較を行う。
従来からあるクラスタリング技術に加えて、深層学習を利用することで、どの程度精度向上が見込めるのか、実用できるレベルなのかを確認する。

---
### クラスタリング技術
深層学習を利用したクラスタリング技術は大きく分けて「AutoEncoders based」「Generative Model Based」「Direct Cluster Optimization」の3種類がある。[1]

- AutoEncoders(AE) based: AEにより次元圧縮された表現ベクトルからクラスタリングを行う
  - AE + K-Means: AEによる潜在ベクトルをK-Meansでクラスタリング
  - DEC[2,3]: AEによる潜在ベクトルをK-Meansでクラスタリングしたクラスタリング中心を事前学習。潜在ベクトルとクラスタ中心の差がt分布に従うと仮定して学習。損失関数はカルバック・ライブラー情報量。正解の分布は、予測値を強化して標準化したもの（式3）
- Generative Model Based: 生成モデルにより学習した潜在ベクトルからクラスタリングを行う
  - VaDE[4,5]: AEを事前学習し、潜在ベクトルをGaussian Mixture Model(GMM)でクラスタリング。潜在ベクトルの平均と分散を初期値として算出。Variational Autoencoder(VAE) により復元誤差が最小に、クラスタの事前確率とのカルバック・ライブラー情報量が最大になるように学習(式17)
- Direct Cluster Optimization
  - 未検証

またベースラインとして、K-Means[6]を使用する。

---
### 個別技術の詳細1, 2, 3...

---
### データと評価指標
データはKerasで使用できるMNISTとFashion MNISTを使用する。[7]

評価指標は以下の3つとし、10回実行した結果の中央値を使用する。[1]
1. Unsupervised Clustering Accuracy (ACC)
2. Normalized Mutual Information (NMI)
3. Adjusted Rand Index (ARI)

---
### 実験結果

#### MNIST
|モデル|ACC|NMI|ARI|
|---|---|---|---|
|K-Means||||
|AE + K-Means||||
|DEC||||
|VaDE|||

#### Fashion MNIST
|モデル|ACC|NMI|ARI|
|---|---|---|---|
|K-Means|0.47|0.51|0.35|
|AE + K-Means|0.56|0.56|0.41|
|DEC|0.59|0.63|0.47|
|VaDE|0.59|0.59|0.45|

---
### まとめ

---
### 参考文献
1. Deep Clustring: https://deepnotes.io/deep-clustering
2. DEC(論文): http://proceedings.mlr.press/v48/xieb16.pdf
3. DEC(コード): https://github.com/XifengGuo/DEC-keras/blob/master/run_exp.py
4. VaDE(論文): https://arxiv.org/pdf/1611.05148.pdf
5. VaDE(コード): https://github.com/slim1017/VaDE
6. scikit-learn: https://scikit-learn.org/stable/modules/clustering.html
7. MNIST, Fashion MNIST: https://keras.io/ja/datasets/

---
