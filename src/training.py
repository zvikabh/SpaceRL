import argparse
import collections
import datetime
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F

import celestial_mechanics as cm
import rl_model


# Number of episodes to generate and train on per epoch.
BATCH_SIZE = 50

# o3 says that it's a good idea to train over all batch results a few times before creating a new batch.
# I guess this makes sense since the network is improving but still vaguely similar to the original episodes.
# Collecting the data indeed takes about 100x more time than training a single epoch.
EPOCHS_PER_BATCH = 10

# Number of steps on which to train simultaneously.
MINIBATCH_SIZE = 2048

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


def ppo_update_loop(
        network: rl_model.ActorCriticNetwork,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        batch_num: Optional[int] = None,
        show_timing_info: bool = False,
        save_best_episode: bool = False,
):
  # Collect data for current model.
  with Timer('Data collection', disable=not show_timing_info):
    episodes: list[list[rl_model.EpisodeStepResult]] = []
    termination_reasons: list[cm.EpisodeTerminationReason] = []
    for _ in range(BATCH_SIZE):
      initial_state = cm.build_initial_state()
      episode, termination_reason = rl_model.play_single_episode(network, initial_state)
      episodes.append(episode)
      termination_reasons.append(termination_reason)
    all_steps = sum(episodes, start=[])
    termination_reasons_cnt = collections.Counter([tr.name for tr in termination_reasons])

  if save_best_episode:
    best_idx, best_episode = max(enumerate(episodes), key=lambda ep: sum(step.reward for step in ep[1]))
    best_episode_termination_reason = termination_reasons[best_idx]
    recorded_episode = rl_model.episode_steps_to_recorded_episode(best_episode, best_episode_termination_reason)
    with open(f'episodes/best_episode_{batch_num:06d}.json', 'w') as f:
      json.dump(recorded_episode.to_json_dict(), f)

  # Convert to tensors. All tensors have shape=(n_steps,) unless otherwise specified.
  states = torch.tensor([step.state_tensor for step in all_steps], dtype=torch.float32)  # shape=(n_steps, STATE_VECTOR_LEN)
  actions = torch.tensor([step.action for step in all_steps], dtype=torch.int32)
  old_logp = torch.tensor([step.action_log_prob for step in all_steps], dtype=torch.float32)
  values = torch.tensor([step.state_value for step in all_steps], dtype=torch.float32)
  advantages = torch.tensor([step.advantage for step in all_steps], dtype=torch.float32)
  returns = advantages + values

  # Normalize advantages tensor
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

  # Optimize
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

  scheduler.step()

  # Summary stats
  batch_str = f'#{batch_num}: ' if batch_num is not None else ''
  loss_str = (
    f'Loss {loss:+6.4f} (P: {policy_loss:+6.4f}, V: {value_loss*VALUE_LOSS_IMPORTANCE:+6.4f}, '
    f'E: {-mb_entropy*ENTROPY_BONUS_IMPORTANCE:+6.4f})'
  )
  print(
    f'{batch_str}{loss_str}, Avg return: {returns.mean():.3f}, '
    f'Avg steps: {len(all_steps)/len(episodes):.1f}, '
    f'Termination: {termination_reasons_cnt}'
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--save_best_episode_every', action='store', default=20, type=int,
    help='If nonzero, store the best episode once every N batches'
  )
  parser.add_argument(
    '--save_checkpoint_every', action='store', default=20, type=int,
    help='If nonzero, store a checkpoint once every N batches'
  )
  parser.add_argument(
    '--load_checkpoint', action='store', default=None,
    help='If specified, loads the initial checkpoint from this file'
  )
  args = parser.parse_args()

  os.makedirs('episodes', exist_ok=True)
  os.makedirs('checkpoints', exist_ok=True)


  if args.load_checkpoint:
    network, optimizer, scheduler, start_iteration = rl_model.load_checkpoint(args.load_checkpoint)
  else:
    network, optimizer, scheduler, start_iteration = rl_model.new_model()

  for i in range(start_iteration, 1000):
    save_episode = (args.save_best_episode_every and (i % args.save_best_episode_every == 0))
    save_ckpt = (args.save_checkpoint_every and (i % args.save_checkpoint_every == 0))
    ppo_update_loop(network, optimizer, scheduler, batch_num=i, save_best_episode=save_episode)
    if save_ckpt:
      rl_model.save_checkpoint(network, optimizer, scheduler, batch_num=i, filename=f'checkpoints/batch_{i:06d}.ckpt')


if __name__ == '__main__':
  main()
