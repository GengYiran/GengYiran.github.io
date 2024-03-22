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

## Reinforcement Learning From Scratch

By [Jiayi Ni](https://rochelleni.github.io/), [Yiran Geng](https://gengyiran.github.io/), [Yitao Liang](https://scholar.google.com/citations?user=KVzR1XEAAAAJ&hl=en) (continually updating ...)

### Overview

- Books
  
  - Theory of Deep Learning
  - RL: an Introduction (Sutton)
  
- Open source code (strongly recommend!)
  - 动手学深度学习: [[link](https://zh.d2l.ai/)]
  - 动手学强化学习: HRL
    - Book: [https://hrl.boyuai.com/](https://hrl.boyuai.com/)
    - Code: [https://github.com/boyu-ai/Hands-on-RL](https://hrl.boyuai.com/)

### Pre-requisites

- create conda environment
- pip install packages
- check GitHub issues

### RL: an Introduction (Sutton)

Reference materials below do not need to be read in full! They are to serve your learning, not to add extra pressure ~

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

**DQN by DeepMind** showcases for the first time a reinforcement learning algorithm that directly utilizes a convolutional neural network to handle pixel input and play various Atari games. DQN serves as the foundation for deep reinforcement learning, but mastering this algorithm requires basic familiarity with reinforcement learning and neural networks. This **step-by-step** guidance is to equip you with the fundamentals.

[Part A](#a) and [Part B](#b) are crucial for understanding policies and environments in reinforcement learning. [Part C](#c) is for neural network implementation. It's strongly recommended that you go through each line and complete all the tasks listed. While it might take some time, trust me, you would reap what you sow. 

These sections can be tackled in any order. It's more natural to start with [Part A](#a), but if you're not confident in your coding skills, starting with [Part B](#b) might be easier for you ~ To help you assess your comprehension, I've included a set of self-check questions for each individual task in [Part B](#b).

Hope you enjoy! :)

<h4 id="a">Part A: Algorithm Implementation</h4>

- Based on pseudocode from Sutton, implement Policy Iteration (including Policy Evaluation and Policy Improvement), Value Iteration, Sarsa, n-step Sarsa, and Q-learning algorithms.

  - If implementing feels a bit challenging, focus on the core and skip less critical parts.

- Read Chapter 4 (DP) and Chapter 5 (TD) of [HRL](https://hrl.boyuai.com/) and compare the code you write with reference code.
  - In addition to correctness of code logic, you can also pay attention to how the reference code encapsulates functions, classes, and names variables.

<h4 id="b">Part B: Getting familiar with Environment</h4>

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
    1. Example code of PPO in this [[link](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#example)].
    2. `custom_gym_env.ipynb` in this [[link](https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb)].
    3. Using Custom Environments in this [[link](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)].
    4. Vectorized Environments in this [[link](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)].
    5. `stable_baselines3.common.vec_env` & `stable_baselines3.common.envs`
- Run arbitrary OpenAI Gym's classic environment using Stable Baselines 3 or any algorithm (DP, TD, etc).

<h4 id="c">Part C: Neural Network Implementation</h4>

Through part A and B, we have become thoroughly familiar with the interactive relationship between policies and environments in reinforcement learning. Before delving into deep reinforcement learning, let's first acquaint ourselves with the basic implementation of neural networks.

- Write a basic neural network using PyTorch, including `__init__` and `forward` functions, a loss function and an optimizer. Data-loading, training and testing processes should all be included.
  - You can refer to PyTorch tutorial for neural networks: [https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
  - There are plenty of other tutorials with detailed examples available, but I won't post it here to encourage you to give it a try yourself!
- CS231n hw1 Q4: Two-Layer Neural Network: https://cs231n.github.io/assignments2023/assignment1/#q4-two-layer-neural-network

Multilayer Perceptrons (MLP) are enough for a simpler environment like Cart Pole, with non-image states as input (coordinates and velocity of the car). But unfortunately, your evil mentor wishes you to implement DQN in the classic Atari environment, where convolutional layers become a necessity in your network structure to extract features from images. So it wouldn't hurt to learn some knowledge about convolutional neural networks. Think positively; this might be preparing for your future escape to computer vision:)

- CS231n notes: https://cs231n.github.io/convolutional-networks/
- CS231n hw2 Q4: CNN https://cs231n.github.io/assignments2023/assignment2/#q4-convolutional-neural-networks

#### Part D: DQN

- Paper
  - Human-level control through deep reinforcement learning
  - Playing atari with deep reinforcement learning
- Video Demo
  - Google DeepMind's Deep Q-learning playing Atari Breakout! [[link](https://www.youtube.com/watch?v=V1eYniJ0Rnk)]
- PyTorch Tutorial [[link](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)]
- CS285 hw3: 2 Deep Q-Learning
  - Handout in this [[link](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw3.pdf)]
  - Code in this [[link](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw3)]

#### Part E: Atari

- Paper
  - The arcade learning environment: An evaluation platform for general agents
  - Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents
- Openai Gym
  - [https://gymnasium.farama.org/environments/atari/](https://gymnasium.farama.org/environments/atari/)

#### Finallly!

- Implement DQN algorithm using Atari environment in OpenAI Gym.

#### Oops... not yet finished!

Be wary of non-breaking bugs! Check out openai's blog and code after your own implementation.

- Blog: https://openai.com/research/openai-baselines-dqn
- Code: https://github.com/openai/baselines/tree/master/baselines/deepq
- Results: https://github.com/openai/baselines-results/blob/master/dqn_results.ipynb

# Alphago Zero

After comprehending the interrelations among Dynamic Programming (DP), Monte Carlo (MC), and Temporal Difference (TD) through the first 8 chapters of Sutton's book, you've grasped the essence of reinforcement learning. Then, you would realize that all algorithms in the entire field of sequential control are variations of the algorithms you already know, lying somewhere between Monte Carlo and Dynamic Programming. (Except for policy gradient methods and imitation learning, which approximate the optimal policy directly and do not need to form an approximate value function). Now it's time to read AlphaGo Zero to personally experience this point. You would be amazed at how Monte Carlo, tree search, and neural networks, those things already familiar to you, can be combined.

#### Notes from Yitao

1. MCTS is the manifestation of Monte Carlo in trees, which can be understood as Monte Carlo plus multi-armed bandits, where each choice at a state is akin to pulling a lever on a slot machine. Although many people equate multi-armed bandits with one-step RL, they lead to different algorithmic development paths from multi-step sequential RL. Algorithms for multi-armed bandits can be ingeniously designed, but if there are many subsequent steps and it's uncertain when they end, many intricate algorithms may not work out, and only simple ideas can scale up quickly (the ease of scaling up for TD is also why TD is more commonly used compared to Monte Carlo).

2. People prefer using TD in scenarios where the state-action space is large. The efficiency of Monte Carlo in Go is higher than TD because Go has a more structured action space, where each move is fundamentally consistent across layers. At such times, the limitation of TD lies in its bootstrapping; the current estimate of state value depends on the estimate of the next state. If the estimate for the next step is inaccurate it leads to significant bias, whereas Monte Carlo has theoretical guarantees of being unbiased.