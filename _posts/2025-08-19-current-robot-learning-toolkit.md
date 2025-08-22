---
layout:     post
title:      "Imitation Learning and Sim-to-Real Reinforcement Learning: The Current Toolkit of Robot Learning"
date:       2025-08-18 19:30:02
excerpt:    ""
---

<b>Note:</b> This is a time of great excitement, hype, and frenzy in intelligent robotics. Robot learning pioneers, entrepreneurs, and venture capitalists are all (over)-promising a future of fully autonomous, human-like robots capable of handling every physically laborious task. Some demos even suggest such a future may be just around the corner. Are we really that close though?

This article is my attempt to step back from the hype and focus on the technology itself: the <i>current toolkit</i> of robot learning. As the title suggests, I’ll explore imitation learning, sim-to-real transfer, and fine-tuning in real-world deployments - both to understand their potential and to see more clearly where their limitations lie.
<hr>


<br>

A key defining characteristic of robots is that they operate in the physical world. They perceive their surroundings through sensors and manipulate their environment through things that move. Classically, the desired behaviors of such systems have been hand-engineered: humans study the sensor data, form an intuition about the task, and then encode a set of rules or control strategies. While effective in carefully controlled settings, this approach becomes brittle and impractical when tasks are complex or when the environment is dynamic and unstructured. Learning-based methods aim to overcome these limitations by leveraging data or experience rather than relying solely on hardcoded logic. 


<div class="row">
    <div class="col-xs-10" style="margin: 0 auto; display: table;">
        <img src="{{ site.baseurl }}/img/RL_IL/agent_env_loop.png" style="display: block; max-width: 100%; height: auto;">
        <center><caption> <b>Agent-Environment interaction (Source: Sutton & Barto 2018)</b> </caption></center>
    </div>
</div>

Within robotics, two broad categories of machine learning approaches have emerged: Reinforcement Learning (RL) and Imitation Learning (IL).

Specifically:
1. Reinforcement learning (RL) is the problem of learning how to map states to actions in order to maximize a numerical reward signal over an extended period of time. The reward is a function of the state and action. In RL, the learner is not told what to do; rather, it must discover which actions yield the most reward from trial-and-error interactions. 
2. Imitation learning (IL) is the problem of learning a policy from a collection of demonstrations. IL differs from RL in that it does not need a reward or any additional metric. A primary assumption in IL is that the demonstrations are obtained from an optimal or near-optimal policy. If the learned policy closely mimics the expert, performance is strong.

The categories differ in the learning signals used to improve behavior. For example, the use of a reward signal is the key differentiator between RL and IL. I will discuss sub-categories of these learning approaches based on the source of the experience.  


RL sub-categories:
- Sim-to-real
- Real-time RL
- Offline RL

# Data Source

- Human Demonstrations
  - <span style="color: forestgreen;">Well-suited for pre-training/behavior cloning</span>
  - <span style="color: firebrick;">Significant time and resources required, including tele-operation setup and skilled robot pilots to provide high-quality demonstrations </span>
- Unlabeled Videos
  - Provide strong visual representations for downstream policy learning
- Policy Evaluation on Robots
  - Well-suited for online learning/RL/fine-tuning
- Simulation
  - Low-cost, scalable, and reproducible 


# Representative Methods in Imitation Learning

## Behavior Cloning

Pomerleau (1988)

$$
\begin{aligned}
&\textbf{Behavior Cloning (BC)} \\[2mm]
&\text{Input: Expert dataset } \mathcal{D} = \{(s_i, a_i)\}_{i=1}^N, \text{ policy } \pi_\theta \\[2mm]
&\text{for iteration = 1 to max\_iters do} \\[2mm]
&\quad \text{Sample mini-batch } \{(s_j, a_j)\} \text{ from } \mathcal{D} \\
&\quad \text{Compute loss: } \mathcal{L}(\theta) = \frac{1}{|\text{batch}|} \sum_j \|\pi_\theta(s_j) - a_j\|^2 \\
&\quad \text{Update policy: } \theta \gets \theta - \eta \nabla_\theta \mathcal{L}(\theta) \\
&\text{end for} \\[2mm]
&\text{Output: Trained policy } \pi_\theta
\end{aligned}
$$

<br>
## Inverse Reinforcement Learning

$$
\begin{aligned}
&\textbf{Inverse Reinforcement Learning (IRL) – Abbeel \& Ng (2004)} \\[2mm]
&\text{Input: Expert trajectories } \mathcal{D}_E = \{\tau_i\}_{i=1}^N, \\
& \text{     feature mapping } \phi(s) \in \mathbb{R}^k, \text{ discount factor } \gamma \in [0,1) \\[1mm]
&\text{Compute expert feature expectations:} \\
&\quad \mu_E = \frac{1}{|\mathcal{D}_E|} \sum_{\tau \in \mathcal{D}_E} \sum_{t=0}^{T} \gamma^t \phi(s_t) \\[1mm]
&\text{Initialize: } \pi_0 \text{ and } \hat{\mu}_0 = \mathbb{E}_{\pi_0}\left[\sum_{t=0}^{T} \gamma^t \phi(s_t)\right] \\[1mm]
&\text{for iteration } i = 1 \text{ to max\_iters do} \\
&\quad \text{Solve for reward weights } w_i \text{ by:} \\
&\quad \quad w_i = \arg\max_{\|w\|_2 \le 1} \; w^T (\mu_E - \hat{\mu}_{i-1}) \\[1mm]
&\quad \text{Define reward function: } R_i(s) = w_i^T \phi(s) \\[1mm]
&\quad \text{Compute optimal policy under } R_i: \pi_i = \arg\max_\pi \mathbb{E}_\pi \left[\sum_{t=0}^{T} \gamma^t R_i(s_t) \right] \\[1mm]
&\quad \text{Update policy feature expectations: } \hat{\mu}_i = \mathbb{E}_{\pi_i}\left[\sum_{t=0}^{T} \gamma^t \phi(s_t)\right] \\[1mm]
&\quad \text{If } \|\mu_E - \hat{\mu}_i\|_2 \le \epsilon, \text{ break} \\
&\text{end for} \\[1mm]
&\text{Output: Recovered reward weights } w_i \text{ and learned policy } \pi_i
\end{aligned}
$$

<br>
## Diffusion Policies

### Training

$$
\begin{aligned}
&\textbf{for } \text{each training step:} \\[6pt]
&\quad (s_{1:H}, a_{1:H}) \sim \mathcal{D} 
    && \# \text{sample trajectory of states/actions} \\[6pt]

&\quad k \sim \text{Uniform}(\{1, \dots, T\})
    && \# \text{sample noise step} \\[6pt]

&\quad \epsilon \sim \mathcal{N}(0, I) 
    && \# \text{sample Gaussian noise} \\[6pt]

&\quad \bar{\alpha}_k = \prod_{i=1}^k (1 - \beta_i) 
    && \# \text{cumulative noise coefficient} \\[6pt]

&\quad a_t^{(k)} = \sqrt{\bar{\alpha}_k}\, a_t 
    + \sqrt{1 - \bar{\alpha}_k}\,\epsilon
    && \# \text{noisy action at step $k$} \\[6pt]

&\quad \hat{a}_t = f_\theta(s_t, a_t^{(k)}, k) 
    && \# \text{model prediction (denoised action)} \\[6pt]

&\quad \mathcal{L}(\theta) = \| \hat{a}_t - a_t \|^2
    && \# \text{loss function} \\[6pt]

&\quad \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)
    && \# \text{gradient update}
\end{aligned}
$$


### Inference

$$
\begin{aligned}
&\textbf{for each environment step $t$:} \\[6pt]

&\quad a_t^{(T)} \sim \mathcal{N}(0, I)
    && \# \text{start from Gaussian noise} \\[6pt]

&\quad \textbf{for } k = T, T-1, \dots, 1: \\[6pt]
&\qquad a_t^{(k-1)} = f_\theta(s_t, a_t^{(k)}, k) 
    && \# \text{iterative denoising} \\[6pt]

&\quad a_t = a_t^{(0)} 
    && \# \text{final clean action} \\[6pt]

&\quad s_{t+1} \sim p(s_{t+1} \mid s_t, a_t)
    && \# \text{step environment}
\end{aligned}
$$


## DDPM Pseudocode (Compared to Diffusion Policy)

### Training

$$
\begin{aligned}
&\text{for each training step:} \\[4pt]

&\quad x_0 \sim p_\text{data}(x_0) 
    && \# \text{sample clean image instead of trajectory} \\[2pt]

&\quad k \sim \text{Uniform}(\{1, \dots, T\}) \\[2pt]

&\quad \epsilon \sim \mathcal{N}(0, I) \\[2pt]

&\quad \bar{\alpha}_k = \prod_{i=1}^k (1 - \beta_i) \\[2pt]

&\quad \textcolor{midnightblue}{x_k = \sqrt{\bar{\alpha}_k}\, x_0 + \sqrt{1 - \bar{\alpha}_k}\, \epsilon} 
    && \# \text{forward noising on image, replacing noisy action} \\[2pt]

&\quad \textcolor{midnightblue}{\hat{\epsilon}_\theta = f_\theta(x_k, k)} 
    && \# \text{predict noise instead of denoised action} \\[2pt]

&\quad \textcolor{midnightblue}{\mathcal{L}(\theta) = \| \epsilon - \hat{\epsilon}_\theta \|^2} \\[2pt]

&\quad \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)
\end{aligned}
$$

### Inference

$$
\begin{aligned}
&\text{for each sample:} \\[2pt]

&\quad x_T \sim \mathcal{N}(0, I) \\[2pt]

&\quad \text{for } k = T, T-1, \dots, 1: \\[2pt]
&\qquad \textcolor{midnightblue}{\hat{\epsilon}_\theta = f_\theta(x_k, k)} \\[2pt]
&\qquad \textcolor{midnightblue}{x_{k-1} = \frac{1}{\sqrt{\alpha_k}} \Big(x_k - \frac{1-\alpha_k}{\sqrt{1-\bar{\alpha}_k}} \hat{\epsilon}_\theta\Big) + \sigma_k z, \quad z \sim \mathcal{N}(0,I)} \\[2pt]

&\quad x_0
\end{aligned}
$$



<br>
## Large Behavior Models

<br>
## Vision-Language Action Models

<br>
# Representative Methods in Reinforcement Learning

<br>
## Learning from scratch on real robots

<br>
## Sim-to-real

### Domain Randomization

### Student-Teacher Distillation


# Marriage of Imitation Learning and Reinforcement Learning

The distinction between RL and IL centers on the source of the learning signal — reward in the case of RL, and demonstrations in the case of IL. But in practice the boundary is less rigid than it appears. Increasingly, research explores hybrid approaches that combine the efficiency of IL with the flexibility of RL, such as bootstrapping RL from demonstrations or inferring rewards from expert trajectories.

Ultimately, the success of robot learning will hinge on integrating diverse forms of experience. Whether through reinforcement, imitation, or their intersection, the goal remains the same: enabling robots to act reliably and adaptively in the unpredictable physical world.

- Closed-loop control and RL agent–environment interaction loop
- In IL, the reward is optionally used only to measure the effectiveness of learned policies; it does not drive policy learning.
- In RL, rewards are used to learn better behaviors.
- IL — BC and IRL
- RL — Sim-to-real (student–teacher distillation), online RL on physical systems (mostly actor–critic methods for continuous control)
- Pros and cons of both
- Why is it tricky to combine the two?
- Ash and Adams work
- Loss of plasticity angle
- Objectives of different magnitudes and varying error norms


## Additional Comments
- Dr. Russ Tedrake’s lecture notes, papers, and talks are a fantastic source for the nitty-gritty details of imitation learning approaches. 

## References
1. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
2. Tedrake, R. (2023). Underactuated robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation. Course Notes for MIT 6.832. https://underactuated.csail.mit.edu.