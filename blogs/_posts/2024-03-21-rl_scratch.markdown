---
layout: post
comments: true
#title:  "Reproducing and Improving LeCun et al. 1989 ConvNet"
title: "Reinforcement Learning From Scratch"
excerpt: "By Jiayi, Yiran, Yitao (continually updating ...)"
date:   2024-03-21 14:00:00
mathjax: false
---

<style>
.post-header h1 {
    font-size: 35px;
}
.post pre,
.post code {
    background-color: #fcfcfc;
    font-size: 13px; /* make code smaller for this post... */
}
</style>

### Overview

- Books
	- Theory of Deep Learning
	- RL: an Introduction (Sutton)
- Open source code (strongly recommend!)
	- 动手学深度学习 https://zh.d2l.ai/
	- 动手学强化学习 HRL https://hrl.boyuai.com/

### Pre-requisites

- create conda environment
- pip install packages

### RL: an Introduction (Sutton)

Reference materials below do not need to be read in full! They are to serve your learning, not to add extra pressure~

- Free Online Book
  - http://incompleteideas.net/book/RLbook2020.pdf
- Official Slides
  - https://drive.google.com/drive/folders/1cMJWR90IkMxWngWpjOtD-qdLS_a7KiYL
- Exercise Solutions
  - Send in your solutions for a chapter, get the official ones back @richsutton.com
  - https://github.com/LyWangPX/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions
  - https://github.com/brynhayder/reinforcement_learning_an_introduction/tree/master/exercises

- Notes
  - https://github.com/brynhayder/reinforcement_learning_an_introduction/blob/master/notes/notes.pdf

- Demo
  - https://cs.stanford.edu/people/karpathy/reinforcejs/index.html

- Code
  - Official Code in C and Lisp http://incompleteideas.net/book/code/code2nd.html
  - Hands-on Reinforcement Learning https://hrl.boyuai.com/

- Courses
  - https://www.davidsilver.uk/teaching/
  - https://web.stanford.edu/class/cs234/modules.html

- Key points
	- Q learning
	    - [https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf) p35
	    - [https://stats.stackexchange.com/questions/184657/what-is-the-difference-between-off-policy-and-on-policy-learning](https://stats.stackexchange.com/questions/184657/what-is-the-difference-between-off-policy-and-on-policy-learning)
	        - The reason that Q-learning is off-policy is that it updates its Q-values using the Q-value of the next state 𝑠’ and the _greedy action_ 𝑎’. In other words, it estimates the _return_ (total discounted future reward) for state-action pairs assuming a greedy policy were followed despite the fact that it's not following a greedy policy.
	- Relationship Between DP and TD
	    - [https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf) p41
	- How is TD derived? Relationship between TD and DP, MC

### Playing Atari with DQN

(Part A and B can be done in reverse order.)

#### Part A: Algorithm Implementation

- Based on pseudocode from Sutton, implement Policy Iteration (including Policy Evaluation and Policy Improvement), Value Iteration, Sarsa, n-step Sarsa, and Q-learning algorithms.

  - If implementing feels a bit challenging, focus on the core and skip less critical parts.

- Read Chapter 4 (DP) and Chapter 5 (TD) of [HRL](https://hrl.boyuai.com/) and compare the code you write with reference code.
  - In addition to correctness of code logic, you can also pay attention to how the reference code encapsulates functions, classes, and names variables.

#### Part B: Getting familiar with Environment

- Run the reference code of HRL in provided Cliff Walking and Frozen Lake environments. Remember to execute the provided print functions and visualization tools to intuitively understand the interaction between policies and environments. Feel free to tweak the code and experiment with it as you like.
  - Compare [HRL's implementation of Cliff Walking environment](https://hrl.boyuai.com/chapter/1/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%AE%97%E6%B3%95#42-%E6%82%AC%E5%B4%96%E6%BC%AB%E6%AD%A5%E7%8E%AF%E5%A2%83) with [Openai Gym's version](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py)
    - Explain this line: `P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]`
    - Why is `P` unnecessary for TD policies?
  - Read [Frozen Lake environment](https://hrl.boyuai.com/chapter/1/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%AE%97%E6%B3%95#45-%E5%86%B0%E6%B9%96%E7%8E%AF%E5%A2%83), make sure you read through the [original Openai Gym's code](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py) and [document](https://gymnasium.farama.org/environments/toy_text/frozen_lake/), and understand how Dynamic Programming policies you implemented interact with Frozen Lake Env.
    - How to create a default map of size 8x8?
    - What would be the result of  `print(env.P[6])`(for map size 4x4)? Read the code and think yourself, and then run a check.
    - What would be the result of  `print(env.P[15])`(for map size 4x4)? What makes the difference? (refer to `frozen_lake.py`)
    - Is the default environment set to be slippery or not? How to explicitly set it?
- Write a simplest environment and embed it into PPO (or any algorithm) in Stable Baselines 3
  - References you might find helpful:
    1. Example code of PPO in this [link](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#example).
    2. `custom_gym_env.ipynb` in this [link](https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb).
    3. Using Custom Environments in this [link](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html).
    4. Vectorized Environments in this [link](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).
    5. `stable_baselines3.common.vec_env` & `stable_baselines3.common.envs`
- Run arbitrary OpenAI Gym's classic environment using Stable Baselines 3 or any algorithm (DP, TD, etc).

#### Part C: DQN

- Paper
  - Human-level control through deep reinforcement learning
  - Playing atari with deep reinforcement learning
- Video Demo
  - Google DeepMind's Deep Q-learning playing Atari Breakout! [[link](https://www.youtube.com/watch?v=V1eYniJ0Rnk)]
- PyTorch Tutorial [[link](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)]
- CS285 hw3: 2 Deep Q-Learning
  - Handout in this [link](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw3.pdf)
  - Code in this [link](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw3)

#### Part D: Atari

Finally

- Implement DQN algorithm using Atari environment in OpenAI Gym.

Not yet finished

- Be wary of non-breaking bugs, [here](https://openai.com/research/openai-baselines-dqn)