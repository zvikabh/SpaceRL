import copy
import dataclasses
import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import celestial_mechanics as cm


# State vector contains:
# 6 entries for shapeship position (2), velocity (2), angle (1), angular velocity (1)
# 4 entries for Earth position (2), velocity (2)
# 4 entries for Luna position (2), velocity (2)
# 1 entry for time
STATE_VECTOR_LEN = 15

# Optimizer settings
LEARNING_RATE = 3e-4
LR_DECAY_RATE = 0.999


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


def state_vector_to_state(step: EpisodeStepResult) -> cm.State:
  state_vector = step.state_tensor
  if len(state_vector) != STATE_VECTOR_LEN:
    raise ValueError(f'State vector must have length {STATE_VECTOR_LEN}, got {len(state_vector)}')

  state = cm.build_initial_state()
  state.spaceship.update_from_vector(state_vector[:6])
  state.target.update_from_vector(state_vector[6:10])
  state.other_objects[0].update_from_vector(state_vector[10:14])
  state.time = datetime.timedelta(seconds=state_vector[14] * 60)
  state.rl_discounted_return = step.discounted_return
  return state


def episode_steps_to_recorded_episode(
        steps: list[EpisodeStepResult],
        termination_reason: cm.EpisodeTerminationReason
) -> cm.RecordedEpisode:
    if not steps:
        raise ValueError('Cannot convert empty step list to RecordedEpisode')
    initial_state = state_vector_to_state(steps[0])
    final_state = state_vector_to_state(steps[-1])
    actions_taken = [cm.Action.from_int(step.action) for step in steps]
    return cm.RecordedEpisode(
      initial_state=initial_state,
      final_state=final_state,
      actions_taken=actions_taken,
      termination_reason=termination_reason
    )


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
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
    )

    # Actor head: outputs action logits.
    # There are 4 possible actions: idle, right, left, both.
    self.actor_head = nn.Linear(32, 4)

    # Critic head: outputs a single value (the state-value)
    self.critic_head = nn.Linear(32, 1)

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
    # action_dist = Categorical(logits=action_logits)
    return action_logits, state_value


def load_checkpoint(
    filename: str
) -> tuple[ActorCriticNetwork, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
  network, optimizer, scheduler, _ = new_model()
  checkpoint = torch.load(filename)
  network.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  scheduler.load_state_dict(checkpoint['scheduler'])
  current_batch_num = checkpoint['batch_num']
  return network, optimizer, scheduler, current_batch_num


def new_model() -> tuple[ActorCriticNetwork, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
  """Same API as load_checkpoint, but returns a new randomly-initialized model."""
  network = ActorCriticNetwork()
  optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_RATE)
  return network, optimizer, scheduler, 0


def save_checkpoint(network, optimizer, scheduler, batch_num, filename) -> None:
  checkpoint = {
    'batch_num': batch_num,
    'model': network.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
  }
  torch.save(checkpoint, filename)


def play_single_episode(
    network: ActorCriticNetwork,
    initial_state: cm.State,
    gamma: float = 0.997,
    lam: float = 0.95,
    verbose: bool = False,
) -> tuple[list[EpisodeStepResult], cm.EpisodeTerminationReason]:
  """Plays a single episode starting at the given state.

  Important: `state` is modified and contains the final state on return.

  Args:
    network: Used to sample the next state.
    initial_state: State from which the episode will begin.
    gamma: Discount factor. The default value of 0.997 means that the reward half-life is about 250 steps or about
      10 seconds, which feels about right given the game mechanics.
    lam: Lambda value for the Generalized Advantage Estimator.
    verbose: If True, print debug output.
  """
  scripted_network = torch.jit.script(network)
  scripted_network.eval()
  state = copy.deepcopy(initial_state)
  step_results: list[EpisodeStepResult] = []
  with torch.no_grad():  # We are using this to generate episode data, not to train.
    while True:
      if state.time > cm.EPISODE_TIME:
        if verbose:
          print(f'Episode ended: Timeout after {cm.EPISODE_TIME}')
        termination_reason = cm.EpisodeTerminationReason.REACHED_TIME_LIMIT
        break
      state_vec = state.to_vector()
      action_logits, state_value = scripted_network(torch.tensor(state_vec))
      action_dist = Categorical(logits=action_logits)
      action_int = action_dist.sample().item()
      action = cm.Action.from_int(action_int)
      state.spaceship.apply_action(action, time_step=cm.SEC_PER_FRAME)
      celestial_exceptions = state.update_positions(time_step=cm.SEC_PER_FRAME)
      reward = state.update_returns(action, celestial_exceptions, time_step=cm.SEC_PER_FRAME)
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
