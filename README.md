# Bayesian Attentional Matrix Factorization

- Conference: 2024.11.1.

- [First Author](https://github.com/jayarnim), [Corresponding Author](https://github.com/jaylee07)

## idea

Implicit feedback represents observable user actions rather than direct preference statements. So it inherently suffers from ambiguity as a signal of true user preference. To address this issue, this study reinterprets the ambiguity of implicit feedback signals as a problem of epistemic uncertainty regarding user preferences and proposes a latent factor model that incorporates this uncertainty within a Bayesian framework.

Specifically, the behavioral vector of a user, which is learned from implicit feedback, is restructured within the embedding space using attention mechanisms applied to the user’s interaction history, forming an implicit preference representation. Similarly, item feature vectors are reinterpreted in the context of the item’s interaction history.

This study replaces the deterministic attention scores with stochastic attention weights treated as random variables whose distributions are modeled using a Bayesian approach. Through this design, the proposed model effectively captures the uncertainty stemming from implicit feedback within the vector representations of users and items.

## architecture

the original BAMF applies only the user’s history as reference information in the Bayesian Attention Modules. however, after the conference, additional experimental results showed that using not only the user’s history but also the item’s history improved performance. so, we corrected our model to:

![01](/desc/architecture.png)

### notation

#### idx

- $u$: target user
- $i$: target item
- $v \in R_{i}^{+} \setminus \{u\}$: history users of target item (target user $u$ is excluded)
- $j \in R_{u}^{+} \setminus \{i\}$: history items of target user (target item $i$ is excluded)

#### vector

- $p \in \mathbb{R}^{M \times K}$: user id embedding vector (we define it as global behavior representation)
- $q \in \mathbb{R}^{N \times K}$: item id embedding vector (we define it as global behavior representation)
- $c_{u} \in \mathbb{R}^{M \times K}$: user context vector (we define it as local preference representation)
- $c_{i} \in \mathbb{R}^{N \times K}$: item context vector (we define it as local preference representation)

#### function

- $\mathrm{bam}(q,k,v)$: bayesian attention module (only single head)
- $\mathrm{layernorm}(\cdot)$: layer normalization
- $\odot$: element-wise product
- $\oplus$: vector concatenation
- $\mathrm{ReLU}$: activation function, ReLU
- $\sigma$: activation function, sigmoid
- $W$: linear transformation matrix
- $h$: linear trainsformation vector
- $b$: bias term

### modeling

#### user representation

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

#### item representation

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

#### agg, matching & predict

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
\hat{y}_{u,i}=x_{u,i}
$$

#### objective function

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

to measure the performance of the proposed model, we used the following dataset:

- [movielens latest small](https://grouplens.org/datasets/movielens/latest/)

we divided the dataset into a ratio of 8:1:1 and used each for `trn`, `val`, and `tst`. negative sampling ratio @ `trn`, `val` is 1:4 (pointwise), 1:1 (pairwise). negative sampling ratio @ `tst` is 1:99.

additionally, a `leave-one-out` dataset was created to monitor early stopping epoch using performance evaluation metrics(`ndcg`). the reason validation loss was not used as a criterion is because of the perceived discrepancy between the performance evaluation metrics and the loss function.