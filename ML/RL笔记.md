# RL Basis

[1]: https://datawhalechina.github.io/easy-rl/#/chapter1/chapter1?id=major-components-of-an-rl-agent



### Sequential data

The history data is the collection of the observation , action and reward.
$$
H = (O_1,A_1,R_1,\cdots,O_t,R_t)
$$

## Types of RL Agents

**根据 agent 学习的东西不同，我们可以把 agent 进行归类。**

- ```
  基于价值的 agent(value-based agent)
  ```

  - 这一类 agent 显式地学习的是价值函数，
  - 隐式地学习了它的策略。策略是从我们学到的价值函数里面推算出来的。

- ```
  基于策略的 agent(policy-based agent)
  ```

  - 这一类 agent 直接去学习 policy，就是说你直接给它一个状态，它就会输出这个动作的概率。
  - 在基于策略的 agent 里面并没有去学习它的价值函数。

- 把 value-based 和 policy-based 结合起来就有了 `Actor-Critic agent`。这一类 agent 把它的策略函数和价值函数都学习了，然后通过两者的交互得到一个最佳的行为。

- value based 也称间接法，policy based称直接法。

# Markov Process

### Markov Property

如果一个状态转移是符合马尔可夫的，那就是说一个状态的下一个状态只取决于它当前状态，而跟它当前状态之前的状态都没有关系。

### Value function

$$
G_t = R_{t+1} + \gamma R_{t+2}+\gamma^2R_{t+3}+\cdots+\gamma^{T-t-1}R_T
$$

where  $\gamma$  denotes the discount factor,
$$
V^\pi(s) = \mathbb{E}(G_t|s_t = s)
$$
Q-function:
$$
Q^\pi(s,a) = \mathbb{E}[G_t|s_t=s,A_t=a] \\
Q^\pi(s,a) = R(s,a)+\gamma\sum_{s^{'} \in S}P(s^{'}|s,a)V^{\pi}(s^{'})
$$

第二条公式是model-based的公式，价值函数$V^{\pi}(s)$ 是基于policy的估计，取同一个状态，不同的policy就会有不同的价值函数，所以目标就是找让价值函数最大的policy。
$$
\begin{align}
V_\pi(s) &= \mathbb E_{A}(Q_\pi(s,a)) \\
&= \sum_{a\in A} \pi(a|s;\theta)Q_\pi(s,a)
\end{align}
$$
 从直觉上讲，Bellman optimality equation 表达了这样一个事实：最佳策略下的一个状态的价值必须等于在这个状态下采取最好动作得到的回报的期望。



## MC

### incremental MC





## TD

时间差分法，Q-learning与Sarsa算法用到的算法，Qlearning使用offpolicy方法，这种方法对状态价值的判断容易存在过估计问题（overestimate）。TD的目的就是得到一个对`Q(s,a)`的完美估计`Q*(s,a)`,而Q*(s,a)如下：
$$
Q_*(s,a) = max_\pi\mathbb{E}[U_t | S_t=s, A_t=a] \\
Q_*(s,a) = \mathbb{E}_{S_{t+1}\sim p(.|S_t,a_t)}[R_t + \gamma max_a Q_*(S_{t+1},a)|S_t=s_t,A_t=a_t]
$$
Actually here, the optimal Q function $Q_*(s,a) = max_a Q_*(s,a)$.

所以得到`Q_*(s,a)`，根据这个估值函数下做的策略就能获得最高回报，MC和TD对以上期望的估计方式不同，MC通过所有步骤，TD通过单步数据更新，因此TD希望通过采样令到：
$$
\mathbb{E}_{(s_t,s_{t+1})\sim D}[Q_*(s_t,a_t) - (R_t + \gamma max_a Q_*(s_{t+1},a))] = 0
$$
这就是我们的objective function, D是整个环境包含的state space稍微修改下，理想状态下，就是要找到`Q_*(s,a)`的参数 $\omega$ 使得上式子能够成立，那么我们通过采样后获得数据，可以按如下式子做估计：
$$
J(\omega) = 1/N\sum_jl(\omega, x_j) \\
x_j : (s_t^j,a_t^j,r_t^j,s_{t+1}^j) \sim D \\
l(\omega, x_j) = [Q_\omega(s_t^j,a_t^j) - (r_t^j + \gamma max_a Q_\omega(s_{t+1}^j,a))]^2
$$


### n step TD

n-step TD generalize both one-step TD and MC. 用实际观测$U_t$去估计价值网络的值，典型MC方法。

不确定的结论：MC就是多步TD，计算$y_t$需把每一步reward加起来。
$$
y_t = \sum_{t}\gamma^tr_t + \gamma^{t+1}Q(s_{t+1},a_{t+1})
$$


### Bootstraping

用一个估算去更新同类估算。

### Behavior policy

The action the agent actually take while exploring the environment. e.g. epsilon greedy or  softmax.

### Target Policy

'Inference' mode, it's deterministic. According to the Q function or the policy function to choose action. Basically it's for calculating the optimizing target in TD learning.

### on policy and off policy

当target policy与behavior policy不一样时为off policy。Sarsa是典型onpolicy，再根据Qfunction进行更新时采取的action`a'` 是基于更新前的$\pi$ 进行决定因此$a^{'} \sim \pi_t$; Q-Learning是 基于以前的探索经验进行更新，无论是表格还是神经网络方法，都用到replay buffer。表格型replay buffer就是Q表，因此DQN里target network为之前的Qnetwork的输出。

sarsa在更新q值时用的action就是接下来就要用的action，sarsa的policy  $\pi$ 就是（epsilon greedy），相对于Qlearning来说，少了点随机性。

for example in DQN:
$$
Q(S_t,a) = r + max_aQ(S_{t+1}, a)
$$
in SARSA:
$$
Q(S_t,a) = r + Q(S_{t+1}, a_{t+1})
$$

$$

$$


# DRL

### Policy Gradient (REINFORCE)

REINFORCE算法通过`MC`的方法计算`episode reward`，来更新策略网络。

##### objective：

$$
J(\theta) = \mathbb{E}_S\left[V(S)\right].
$$

The goal is to maximize the objective function.

### PPO

[详解近端策略优化(ppo，干货满满) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/471826751)

[重要性采样(Importance Sampling)详细学习笔记 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/342936969)

#### Importance Sampling

$$
\mathbb E_{x\sim p}(f(x)) = \mathbb E_{x\sim q}(\frac{f(x)}{q(x)})
$$

$f(x)$ can be any function and $q(x)$ is a probability distribution function.

##### motivation

Maximize the trajectory reward, while gradient update the model, the behavior policy have been changed.





### DQN

The parameter update manner of tabular method and the DQN are different。

#### Over Estimate

In pendulum, theoretically, the maximum value of $Q_*(s, a)$ should not over 0, and the reward gain from the environment is <= 0, but actually some estimation $Q_w(s,a)$ would output a positive number, so it would cause that some of the Q value would over 0.

refer to deeprl v1.

### Actor Critic

`AC`和`REINFORCE`都是policy gradient方法，不同点在于reinforce用的环境给的reward去更新，AC用一个价值网络的输出去更新，



## Distributed RL

A3C 与 Gorila的区别是，一个是类似on-policy learning，一个是off-policy learning，Gorila每次探索都会学习，A3C收集一段数据后再更新。  

## FRL

The global objective function is as follow:


$$
\min {F(w)} = \sum_{k=1}^Np_kF_k(w)
$$
In the Q learning setting, our goal is to find the optimal estimation function `Q_*(s,a)` ,   where
$$
Q_*(s,a) = \mathbb{E}_{S_{t+1}\sim p(.|S_t,a_t)}[R_t + \gamma max_a Q_*(S_{t+1},a)|S_t=s_t,A_t=a_t]
$$
we use TD to estimate the expectation,  we hope that we could find $w_*$   such that
$$
\mathbb{E}_{(s_t,s_{t+1})\sim D}[(Q_*(s_t,a_t) - (R_t + \gamma max_a Q_*(s_{t+1},a)))^2] = 0
$$
so the objective function can be:
$$
F(w) = \mathbb{E}_{S\sim D}[(Q_w(s_t,a_t) - (R_t + \gamma max_a Q_w(s_{t+1},a)))^2]
$$
rewrite it:
$$
F(w) = 1/N\sum_j^Nl(w;x_j) \\
l(\omega, x_j) = [Q_\omega(s_t^j,a_t^j) - (r_t^j + \gamma max_a Q_\omega(s_{t+1}^j,a))]^2 \\
x_j : (s_t^j,a_t^j,r_t^j,s_{t+1}^j) \sim D
$$
above is the centralize objective function, i.e. we only have the global agent, and the data $x_j$ is collected from the joint distribution `D`:
$$
P(x) = P(s_t,a_t,r_t,s_{t+1}) = P(s_t)P_w(a_t|s_t)P(s_{t+1}|s_t,a_t)
$$
therefore the data collected from exploration is affected by two distrbution $P(s)$ and $P_w(a|s)$, specifically it is affected by the environment and the current policy of the agent.

with above,  we denote the local objective function of the device `k`as:
$$
F_k(w_k) = 1/n_k\sum_j^{n_k}l(w_k;x_j^k) \\
x_j^k : (s_t^j,a_t^j,r_t^j,s_{t+1}^j) \sim D_k
$$
In this case, if  the environment of device k is different or the policy of the agent k is different, the distribution $D_k$ would be different, and the collected data would be non iid.

The update of the parameters $w_k$:
$$
w_{t+i+1}^k = w_{t+i}^k - \eta\nabla F_k(w_{t+i}^k;x_{t+i}^k), i = 0,1,..,E \\
\nabla F_k(w_{t+i}^k;x_{t+i}^k) = 1/n_k\sum_j^{n_k}\nabla l(w_t^k;x_j^k)
$$
$w_t$ is the parameters loaded from server at time step $t$ .  $E$ is the update frequency.  The update of `FedAvg` in global:
$$
w_{t+E} = \sum_{k =1}^N p_kw_{t+E}^k
$$
$p_k=n_k/n$ , We  assume $n_0 = n_1... = n_k$  and the environment is stationary i.e. $P(s)$  of different device is identical, to prove that when E=1, above update is equivalent to centralized mode update:
$$
w_{t+1} = \sum_{k =1}^N p_kw_{t+1}^k\\
= \sum_{k =1}^N p_k[w_t^k - \eta\nabla F_k(w_t^k;x_t^k)]\\
=\sum_{k=1}^N \frac{n_k}{n}w_t^k - \eta\sum_{k=1}^N \frac{n_k}{n}\frac{1}{n_k}\sum_{j=1}^{n_k}\nabla l(w_t^k;x_{tj}^k) \\
$$
when $n_0 = n_1... = n_k$ and $w_t^k = w_t$:
$$
\sum_{k=1}^N \frac{n_k}{n}w_t^k - \eta\sum_{k=1}^N \frac{n_k}{n}\frac{1}{n_k}\sum_{j=1}^{n_k}\nabla l(w_t^k;x_{tj}^k) \\
=w_t - \eta\nabla  F(w_t; x_t)
$$
where $x_t=\left\{x_t^k\right\}k\in N$ .

consider that in distributed dqn, there are two ways to collect the data, one is Gorila dqn, randomly choose data from local replay buffer which contains the data collected based on old policy, so the data is from vary distribution. The other is asynchrounous one step dqn, compute the gradient based on current policy. In fedavg, when E = 1, and if the environment is identity, the collected data would be from the identity distribution, but however when E > 1, each local data collected from the device are based on different local policy, so the local data within the device would be very relavant, but data between the device would be very different, this could be affect the averaging. So I think we sill need the replay buffer to randomly collect data.

```python
func ClientUpdate(N, agent, global_net):
    if global_step N % M == 0
    	agent.target_net sync from server
    agent.policy_net sync from server
    for i = 0 ... E, E = 0,1,2,3
    	agent Explore T episodes, save experience <s_t,a,r,s_t+1> to replay buffer
        for b = 0...B
        	sample minibatch data from replay buffer
        	compute gradient gi
        	update agent.policy_net
        
     return parameters of policy_net 

func ServerUpdate(wk, N):
    wt+1 = weighted sum(wk)
    N = N + 1 # server update times
    return N

###################original version dqn need multithread to realize multi agent version
func ClientUpdate():
    for i= 0... episode
    	while True
        	explore one trajectory, save to replay buffer
            sample minibatch data
            compute gradient g
            update policy net
            if done break
    	if i % M == 0
    	update target net
```

when E = 1, B = 1, T = 1(1 trajectory), this could be equal to single agent update.

 but note that the above condition  is `not` strictly equal to original dqn algorithm, need multiprocessing to achieve this.

## Offline RL

### BCQ

[【强化学习 119】BCQ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/136844574)

1. Absent data
2. Model Bias
3. Training Mismatch

The BCQ algorithm is based on the typical Q-learning algorithm (and also TD3). The policy model can be decomposed as a Generative model $G_{\omega}$ and a perturbation model $\zeta_{\phi}(s,a)$, action $a$ is generated by the G model and $\zeta$ adjust the action.

It's worth to see the proof of theorem 1. 
$$
\phi \leftarrow argmax_{\phi}Q_{\theta}(s, G_{\omega}(s)+\zeta_{\phi}(s,a)) \\
\phi \leftarrow argmax_{\phi}Q_{\theta}(s, \pi(s))
$$
The first eqation is the update of the policy in BCQ, second one is the update of policy in TD3.

### CQL





### Fisher BRC



### TD3+BC

