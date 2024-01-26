# Actor_Critic_Reinforcement_Learning

## Actor-Critic Algorithm

In Actor-Critic algorithms, we simultaneously train an actor (policy) and a critic (value function). The actor guides the agent's actions, and the critic evaluates the state or state-action values.

### Mathematical Formulation

The objective function in Actor-Critic methods can be written as follows:

\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \left( \log \pi_{\theta}(a_t|s_t) \cdot A_{\pi_{\theta}}(s_t, a_t) + \beta \cdot V_{\phi}(s_t) \right) \right]
\]

Where:
- \(J(\theta)\) is the objective function to be maximized with respect to the actor's parameters \(\theta\).
- \(\tau\) represents a trajectory sampled from the policy \(\pi_{\theta}\).
- \(\pi_{\theta}(a_t|s_t)\) is the probability of taking action \(a_t\) in state \(s_t\) under policy \(\pi_{\theta}\).
- \(A_{\pi_{\theta}}(s_t, a_t)\) is the advantage function, representing the advantage of taking action \(a_t\) in state \(s_t\).
- \(\beta\) is an optional entropy regularization term to encourage exploration.
- \(V_{\phi}(s_t)\) is the value function provided by the critic with parameters \(\phi\).

The actor's parameters (\(\theta\)) are updated using the policy gradient, while the critic's parameters (\(\phi\)) are updated to minimize the mean squared error between the estimated value and the actual returns.

Feel free to customize this formula based on your specific Actor-Critic implementation.
