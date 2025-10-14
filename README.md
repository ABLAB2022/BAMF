# Bayesian Attentional Matrix Factorization

- Conference: 2024.11.1.

- First Author: [Wang,J.](https://github.com/jayarnim)

- Corresponding Author: [Lee,J.](https://github.com/jaylee07)

## idea

Implicit feedback represents observable user actions rather than direct preference statements. So it inherently suffers from ambiguity as a signal of true user preference. To address this issue, this study reinterprets the ambiguity of implicit feedback signals as a problem of epistemic uncertainty regarding user preferences and proposes a latent factor model that incorporates this uncertainty within a Bayesian framework.

Specifically, the behavioral vector of a user, which is learned from implicit feedback, is restructured within the embedding space using attention mechanisms applied to the user’s interaction history, forming an implicit preference representation. Similarly, item feature vectors are reinterpreted in the context of the item’s interaction history.

This study replaces the deterministic attention scores with stochastic attention weights treated as random variables whose distributions are modeled using a Bayesian approach. Through this design, the proposed model effectively captures the uncertainty stemming from implicit feedback within the vector representations of users and items.

## architecture

the original BAMF applies only the user’s history as reference information in the Bayesian Attention Modules:

![01](/desc/origin.png)

however, after the conference, additional experimental results showed that using not only the user’s history but also the item’s history improved performance. so, we corrected our model to:

![02](/desc/improved.png)

## notation

### idx

- $u$: target user
- $i$: target item
- $v \in R_{i}^{+} \setminus \{u\}$: history users of target item (target user $u$ is excluded)
- $j \in R_{u}^{+} \setminus \{i\}$: history items of target user (target item $i$ is excluded)

### vector

- $p \in \mathbb{R}^{M \times K}$: user id embedding vector (we define it as global behavior representation)
- $q \in \mathbb{R}^{N \times K}$: item id embedding vector (we define it as global behavior representation)
- $c_{u} \in \mathbb{R}^{M \times K}$: user context vector (we define it as local preference representation)
- $c_{i} \in \mathbb{R}^{N \times K}$: item context vector (we define it as local preference representation)

### function

- $\mathrm{bam}(q,k,v)$: bayesian attention module (only single head)
- $\mathrm{layernorm}(\cdot)$: layer normalization
- $\odot$: element-wise product
- $\oplus$: vector concatenation
- $\mathrm{ReLU}$: activation function, ReLU
- $\sigma$: activation function, sigmoid
- $W$: linear transformation matrix
- $h$: linear trainsformation vector
- $b$: bias term

## modeling

### user representation

- user id embedding:

$$
p_{u}=\mathrm{embedding}(u)
$$

- user-hist. bam:

$$
c_{u}=\mathrm{bam}(p_{u}, \forall q_{j}, \forall q_{j})
$$

- user final representation vector:

$$
z_{u}=\mathrm{layernorm}(p_{u} \odot c_{u})
$$

### item representation

- item id embedding:

$$
q_{i}=\mathrm{embedding}(i)
$$

- item-hist. bam:

$$
c_{i}=\mathrm{bam}(q_{i}, \forall p_{v}, \forall p_{v})
$$

- item final representation vector:

$$
z_{i}=\mathrm{layernorm}(q_{i} \odot c_{i})
$$

### agg, matching & predict

- general matrix factorization:

$$
z=z_{u} \odot z_{i}
$$

- logit:

$$
x_{u,i}=h^{T}(Wz+b)
$$

- prediction:

$$
\hat{y}_{u,i}=\sigma(x_{u,i})
$$

### objective function

$$
\mathcal{L}_{\mathrm{ELBO}}:= \sum_{(u,i)\in\Omega}{\left(\mathrm{NLL} + \sum_{j \in R_{u}^{+} \setminus \{i\}}{\mathrm{KL}^{(u,j)}} + \sum_{v \in R_{i}^{+} \setminus \{u\}}{\mathrm{KL}^{(v,i)}} \right)}
$$

- apply `bce` to pointwise `nll`:

$$
\mathcal{L}_{\mathrm{BCE}}:=-\sum_{(u,i)\in\Omega}{y_{u,i}\ln{\hat{y}_{u,i}} + (1-y_{u,i})\ln{(1-\hat{y}_{u,i})}}
$$

- apply `bpr` to pairwise `nll`:

$$
\mathcal{L}_{\mathrm{BPR}}:=-\sum_{(u,pos,neg)\in\Omega}{\ln{\sigma(x_{u,pos} - x_{u,neg})}+\lambda_{\Theta}\Vert \Theta \Vert^{2}}
$$

### bayesian attention modules application

we use `prod` and `concat` as attention score functions, proposed by "NAIS: Neural attentive item similarity model for recommendation (He et al., 2018)".

- `prod` function:

$$
f(q,k)=h \cdot \mathrm{ReLU}(W \cdot [p \oplus q] + b)
$$

- `concat` function:

$$
f(q,k)=h \cdot \mathrm{ReLU}(W \cdot [p \odot q] + b)
$$

if the number of key is too large, attention weights become flat. so, in simplex projection, we introduced smoothing factor $0 < \beta \le 1$, to increase the individual weight values, and sharpening factor $1 \le \tau$, to widen the gap between weights.

$$
\alpha= \frac{s^{\tau}}{(\sum{s^{\tau}})^{\beta}}
$$

## experiment

### setting

to measure the performance of the proposed model, we used the following dataset:

- [movielens latest small](https://grouplens.org/datasets/movielens/latest/)

we divided the dataset into a ratio of 8:1:1 and used each for `trn`, `val`, and `tst`. negative sampling ratio @ `trn`, `val` is 1:4 (pointwise), 1:1 (pairwise). negative sampling ratio @ `tst` is 1:99.

additionally, a `leave-one-out` dataset was created to monitor early stopping epoch using performance evaluation metrics(`ndcg`). the reason validation loss was not used as a criterion is because of the perceived discrepancy between the performance evaluation metrics and the loss function.

initially, model performance for early stopping was evaluated on the leave-one-out dataset every five epochs to reduce computational cost. later, the evaluation procedure was refined by replacing repeated sampling and averaging with the expected attention scores, enabling performance validation at every epoch.

the maximum length of a user’s interaction history is about 2,000 items, and the top 10% of users have histories of around 400 items. to improve computational efficiency, each history was truncated up to 400 items according to their TF-IDF scores.

### result

- pointwise learning (attention score function `concat` is applied)

    | top_k | hit_ratio | precision | recall | map | ndcg |
    |:------:|:----------:|:----------:|:--------:|:------:|:------:|
    | 5 | 0.862295 | 0.394754 | 0.303068 | 0.225563 | 0.470217 |
    | 10 | 0.955738 | 0.312951 | 0.432504 | 0.279262 | 0.485796 |
    | 15 | 0.977049 | 0.259563 | 0.506017 | 0.301525 | 0.497536 |
    | 20 | 0.981967 | 0.221967 | 0.550273 | 0.313100 | 0.504902 |
    | 25 | 0.988525 | 0.199016 | 0.592796 | 0.322449 | 0.515446 |
    | 50 | 0.996721 | 0.134262 | 0.697822 | 0.342110 | 0.544622 |
    | 100 | 1.000000 | 0.086836 | 0.801550 | 0.355990 | 0.578916 |

- pointwise learning (attention score function `prod` is applied)

    | top_k | hit_ratio | precision | recall | map | ndcg |
    |:------:|:----------:|:----------:|:--------:|:------:|:------:|
    | 5 | 0.878689 | 0.415738 | 0.311538 | 0.239060 | 0.497262 |
    | 10 | 0.954098 | 0.325574 | 0.442823 | 0.294098 | 0.509249 |
    | 15 | 0.978689 | 0.271913 | 0.515465 | 0.317305 | 0.520918 |
    | 20 | 0.985246 | 0.233607 | 0.565203 | 0.330207 | 0.529837 |
    | 25 | 0.995082 | 0.209311 | 0.604733 | 0.339656 | 0.539434 |
    | 50 | 0.996721 | 0.140656 | 0.711178 | 0.360930 | 0.568152 |
    | 100 | 0.998361 | 0.090639 | 0.804510 | 0.374832 | 0.598809 |

- pairwise learning (attention score function `concat` is applied)

    | top_k | hit_ratio | precision | recall | map | ndcg |
    |:------:|:----------:|:----------:|:--------:|:------:|:------:|
    | 5 | 0.814754 | 0.337377 | 0.265073 | 0.192894 | 0.402172 |
    | 10 | 0.932787 | 0.274098 | 0.399305 | 0.240260 | 0.427426 |
    | 15 | 0.954098 | 0.231148 | 0.472754 | 0.260851 | 0.444057 |
    | 20 | 0.968852 | 0.203689 | 0.523902 | 0.273406 | 0.458043 |
    | 25 | 0.980328 | 0.182885 | 0.564664 | 0.281970 | 0.469224 |
    | 50 | 0.998361 | 0.124492 | 0.687062 | 0.301819 | 0.505482 |
    | 100 | 1.000000 | 0.081443 | 0.784817 | 0.314826 | 0.540677 |

- pairwise learning (attention score function `prod` is applied)

    | top_k | hit_ratio | precision | recall | map | ndcg |
    |:------:|:----------:|:----------:|:--------:|:------:|:------:|
    | 5 | 0.803279 | 0.336393 | 0.257059 | 0.189611 | 0.407527 |
    | 10 | 0.906557 | 0.268525 | 0.380639 | 0.234973 | 0.423831 |
    | 15 | 0.947541 | 0.226448 | 0.455761 | 0.254688 | 0.439570 |
    | 20 | 0.972131 | 0.196475 | 0.504426 | 0.265234 | 0.450350 |
    | 25 | 0.985246 | 0.177115 | 0.546550 | 0.273634 | 0.461773 |
    | 50 | 1.000000 | 0.124131 | 0.672987 | 0.294446 | 0.500109 |
    | 100 | 1.000000 | 0.080803 | 0.778475 | 0.307230 | 0.535381 |