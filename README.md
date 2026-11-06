# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium


## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description


The FrozenLake environment is a 4×4 grid containing 16 states.

- S (Start): Starting position of the agent.
- F (Frozen): Safe frozen surface.
- H (Hole): Dangerous position that ends the episode.
- G (Goal): Destination that gives a reward of 1.

The agent can perform four actions:

```
Action	  Direction
0	      Left
1	      Down
2	      Right
3	      Up
```

The environment uses is_slippery=True, which means the agent may move in an unintended direction


---

## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---


## Algorithm
- Start the program.
- Import Gymnasium, NumPy and Matplotlib.
- Create the FrozenLake environment.
- Initialize the Q-table with zeros.
- Set the learning rate, discount factor and epsilon values.
- Select actions using the epsilon-greedy policy.
- Generate a complete episode.
- Calculate returns by traversing the episode backwards.
- Update the Q-table using the Monte Carlo update rule.
- Reduce epsilon after each episode.
- Repeat the process for the specified number of episodes.
- Extract the greedy policy from the Q-table.
- Display the Q-table, state-value function and learned policy.
- Plot the learning curve.
- Stop the program


## Python Program

#### Monte Carlo Control


```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", map_name="4x4",
               is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((n_states, n_actions))
episode_rewards = []

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------
def generate_episode(epsilon):

    episode = []
    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------
epsilon = epsilon_start

for episode_num in range(num_episodes):

    episode = generate_episode(epsilon)

    G = 0

    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        Q[state, action] += alpha * (G - Q[state, action])

    total_reward = sum(step[2] for step in episode)

    episode_rewards.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------
optimal_policy = np.argmax(Q, axis=1)

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------
def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Name: ")
    print("Register Number: ")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(optimal_policy)


success_rate = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:", success_rate)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")

plt.grid(True)
plt.show()

env.close()

```

---

## Output


Final Q-table:

<img width="275" height="305" alt="image" src="https://github.com/user-attachments/assets/9c0c5fab-6d71-40d4-a37f-3427fab4493b" />




Estimated State-Value Function:

<img width="301" height="132" alt="image" src="https://github.com/user-attachments/assets/df47b990-264c-4225-9987-2ac91cb96f35" />






Learned Policy:

Average reward over last 1000 episodes:

<img width="377" height="134" alt="image" src="https://github.com/user-attachments/assets/c498e041-b8ef-4fb3-9722-271810fc81a8" />




<img width="612" height="414" alt="image" src="https://github.com/user-attachments/assets/a0ae0520-d220-445d-8cae-dd950326babf" />



---

## Result

Thus, the On-Policy Monte Carlo Control algorithm was successfully implemented using the Gymnasium FrozenLake environment. The Q-table was updated using Monte Carlo returns, and the learned policy, state-value function and learning curve were obtained.


---

## Inference

The Monte Carlo Control algorithm learns an improved policy through repeated episodes. The Q-table estimates the action values, and the epsilon-greedy method balances exploration and exploitation. The learned policy helps the agent reach the goal while avoiding holes. The learning curve shows the improvement in the agent's performance during training.





---

