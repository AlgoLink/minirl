<h1 align="center"><a href="https://github.com/AlgoLink/minirl">minirl</a></h1>
<p align="center">
  <em>基于Numpy的深度强化学习</em>
</p>

---

这个 repo 包含一组从头开始使用 numpy 实现的用于强化学习的应用程序和算法。包括的算法q-learning、基于深度神经网络的REINFORCE、Actor-Critic和ppo等。

## 项目结构

    .
    ├── core
        ├── bandit.py               # EpsilonGreedy/UCB/LinUCB/ThompsonSampling algorithm
        ├── smab.py                 # stochastic Multi-Armed Bandit (sMAB)
        ├── cmab.py                 # contextual Multi-Armed Bandit (cMAB) based on Thompson Sampling
        ├── onlineCluster.py        # online k-means using Lloyd's algorithm
        ├── pg.py                   # REINFORCE algorithm
        ├── deep_q_learning.py      # Deep Neural Network based Q-learning
        ├── ac.py                   # Actor-Critic algorithm
        ├── ppo.py                  # Proximal Policy Optimization
        ├── DynaQ.py                # Dyna-Q algorithm
        ├── DynaQ_plus.py           # Time-based model for planning in Dyna-Q+
    ├── preprocessing
        ├── feature_transformer.py  # OneHotEncoder/TargetEncoder
        ├── scaler.py               # StandardScaler/MinMaxScaler/MaxAbsScaler
        ├── stats.py                # runningReward
    ├── common                      
        ├── net.py                  # Common-deep-network
        ├── optim.py                # Optimizer        
    └── README.md

## 技术架构

<img src="resources/art.jpg">