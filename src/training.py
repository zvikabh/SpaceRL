import collections
import dataclasses
import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm

import celestial_mechanics as cm


# State vector contains:
# 6 entries for shapeship position (2), velocity (2), angle (1), angular velocity (1)
# 4 entries for Earth position (2), velocity (2)
# 4 entries for Luna position (2), velocity (2)
# 1 entry for time
STATE_VECTOR_LEN = 15

# Number of episodes to generate and train on per epoch.
BATCH_SIZE = 100

# o3 says that it's a good idea to train over all batch results a few times before creating a new batch.
# I guess this makes sense since the network is improving but still vaguely similar to the original episodes.
# Collecting the data indeed takes about 100x more time than training a single epoch.
EPOCHS_PER_BATCH = 3

# Number of steps on which to train simultaneously.
MINIBATCH_SIZE = 2048

# Optimizer settings
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.5

# PPO parameters
CLIP_EPS = 0.05  # epsilon for clipping in the PPO policy surrogate loss.
VALUE_LOSS_IMPORTANCE = 1  # Relative importance of the value loss component, usually in [0.5, 1].
ENTROPY_BONUS_IMPORTANCE = 0.01  # Increase this to encourage more exploration.


class Timer:
  def __init__(self, title: str = '', disable: bool = False):
    self.title = title
    self.disable = disable

  def __enter__(self):
    self.start_time = datetime.datetime.now()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not self.disable:
      elapsed = datetime.datetime.now() - self.start_time
      print(f'{self.title} completed in {elapsed}')


@dataclasses.dataclass
class EpisodeStepResult:
  """Torch-ready data for a single step in an episode.

  Attributes:
    state_tensor: State before taking action. Vector of length STATE_VECTOR_LEN.
    action: Action taken (integer in the range 0-3).
    action_log_prob: Probability for taking this action, per current policy.
    reward: Reward for this single step.
    state_value: Critic's estimate for the value of the current state.
    discounted_return: Discounted return for this action based on reward for this timestep plus (discounted) all future
      timesteps.
    advantage: Generalized Advantage Estimate (GAE) for the current state.
  """
  state_tensor: list[float]
  action: int
  action_log_prob: float
  reward: float
  state_value: float
  discounted_return: Optional[float] = None
  advantage: Optional[float] = None


class ActorCriticNetwork(nn.Module):
  """Network with an Actor making decisions and a Critic grading them.

  The network architecture is 2 fully connected shared layers, plus two heads:
  one for the actor (4 categorical outputs) and one for the critic (1 grade output).
  """

  def __init__(self):
    super().__init__()

    self.shared_body = nn.Sequential(
      nn.Linear(STATE_VECTOR_LEN, 256),
      nn.ReLU(),
      nn.Linear(256, 64),
      nn.ReLU()
    )

    # Actor head: outputs action logits.
    # There are 4 possible actions: idle, right, left, both.
    self.actor_head = nn.Linear(64, 4)

    # Critic head: outputs a single value (the state-value)
    self.critic_head = nn.Linear(64, 1)

  def forward(self, state):
    r"""Forward pass in the actor-critic model.

    Args:
      state: State vector $s$.

    Returns:
      action_dist: Policy recommendation distribution $\pi_\theta(s)$ given the current state.
        A 4-element categorical distribution.
      state_value: Learned state value estimate for the current state, $V_phi(s)$.
    """
    body_output = self.shared_body(state)
    action_logits = self.actor_head(body_output)
    state_value = self.critic_head(body_output).squeeze(-1)
    action_dist = Categorical(logits=action_logits)
    return action_dist, state_value


def play_single_episode(
    network: ActorCriticNetwork, gamma: float = 0.997, lam: float = 0.95, verbose: bool = False,
) -> tuple[list[EpisodeStepResult], cm.EpisodeTerminationReason]:
  """Plays a single episode starting at the given state.

  Important: `state` is modified and contains the final state on return.

  Args:
    network: Used to sample the next state.
    gamma: Discount factor. The default value of 0.997 means that the reward half-life is about 250 steps or about
      10 seconds, which feels about right given the game mechanics.
    lam: Lambda value for the Generalized Advantage Estimator.
    verbose: If True, print debug output.
  """
  state = cm.build_initial_state()
  step_results: list[EpisodeStepResult] = []
  with torch.no_grad():  # We are using this to generate episode data, not to train.
    while True:
      if state.time > cm.EPISODE_TIME:
        if verbose:
          print(f'Episode ended: Timeout after {cm.EPISODE_TIME}')
        termination_reason = cm.EpisodeTerminationReason.REACHED_TIME_LIMIT
        break
      state_vec = state.to_vector()
      action_dist, state_value = network(torch.tensor(state_vec))
      action_int = action_dist.sample().item()
      action = cm.Action.from_int(action_int)
      state.spaceship.apply_action(action, time_step=cm.SEC_PER_FRAME)
      celestial_exceptions = state.update_positions(time_step=cm.SEC_PER_FRAME)
      reward = state.update_returns(celestial_exceptions, time_step=cm.SEC_PER_FRAME)
      step_results.append(
        EpisodeStepResult(
          state_tensor=state_vec,
          action=action_int,
          action_log_prob=action_dist.log_prob(torch.tensor(action_int)),
          reward=reward,
          state_value=state_value,
        )
      )
      if any(isinstance(collision_ex.obj, cm.Spaceship) for collision_ex in celestial_exceptions):
        if verbose:
          print(f'Episode ended: {celestial_exceptions[0]}')
        termination_reason = celestial_exceptions[0].termination_reason()
        break

  cur_discounted_return = 0.
  for step_result in reversed(step_results):
    cur_discounted_return = step_result.reward + gamma * cur_discounted_return
    step_result.discounted_return = cur_discounted_return

  last_adv = 0.
  for t in reversed(range(len(step_results))):
    if t == len(step_results) - 1:
      # Last step
      next_v = 0.
    else:
      next_v = step_results[t + 1].state_value
    delta = step_results[t].reward + gamma * next_v - step_results[t].state_value
    last_adv = delta + gamma*lam*last_adv
    step_results[t].advantage = last_adv

  return step_results, termination_reason


def ppo_update_loop(network: ActorCriticNetwork, show_timing_info: bool = False):
  # Collect data for current model.
  with Timer('Data collection', disable=not show_timing_info):
    episodes, termination_reasons = zip(*[play_single_episode(network) for _ in range(BATCH_SIZE)])
    all_steps = sum(episodes, start=[])
    termination_reasons = collections.Counter([tr.name for tr in termination_reasons])

  # Convert to tensors. All tensors have shape=(n_steps,) unless otherwise specified.
  states = torch.tensor([step.state_tensor for step in all_steps], dtype=torch.float32)  # shape=(n_steps, STATE_VECTOR_LEN)
  actions = torch.tensor([step.action for step in all_steps], dtype=torch.int32)
  old_logp = torch.tensor([step.action_log_prob for step in all_steps], dtype=torch.float32)
  values = torch.tensor([step.state_value for step in all_steps], dtype=torch.float32)
  advantages = torch.tensor([step.advantage for step in all_steps], dtype=torch.float32)
  returns = advantages + values

  print(
    f'Avg return: {returns.mean():.3f}, Avg steps per episode: {len(all_steps)/len(episodes):.1f}, '
    f'Termination reasons: {termination_reasons}'
  )

  # Normalize advantages tensor
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

  # Optimize
  optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
  with Timer('Training', disable=not show_timing_info):
    for epoch in range(EPOCHS_PER_BATCH):
      idx = torch.randperm(len(all_steps))
      for minibatch_start in range(0, len(all_steps), MINIBATCH_SIZE):
        minibatch = idx[minibatch_start : minibatch_start+MINIBATCH_SIZE]
        mb_states = states[minibatch]
        mb_actions = actions[minibatch]
        mb_advantages = advantages[minibatch]
        mb_returns = returns[minibatch]
        mb_old_logp = old_logp[minibatch]

        mb_action_dist, mb_state_value = network(mb_states)
        mb_logp = mb_action_dist.log_prob(mb_actions)
        mb_entropy = mb_action_dist.entropy().mean()

        mb_prob_ratio = (mb_logp - mb_old_logp).exp()
        surrogate1 = mb_prob_ratio * mb_advantages
        surrogate2 = torch.clamp(mb_prob_ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = F.mse_loss(mb_state_value, mb_returns)

        loss = policy_loss + VALUE_LOSS_IMPORTANCE * value_loss - ENTROPY_BONUS_IMPORTANCE * mb_entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), MAX_GRAD_NORM)
        optimizer.step()

  # Summary stats



def main():
  network = ActorCriticNetwork()
  with Timer('Episode'):
    episode, termination_reason = play_single_episode(network, verbose=True)
  print(f'Terminated after {len(episode)} steps: {termination_reason}')
  print(f'Discounted return at t=0: {episode[0].discounted_return:.3f}')
  print(f'Rewards: {[step.reward for step in episode]}')

  print('\n\n')

  for _ in tqdm.tqdm(range(100)):
    ppo_update_loop(network)


if __name__ == '__main__':
  main()
