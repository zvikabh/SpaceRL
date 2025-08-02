import argparse
import copy
import datetime
import json
from typing import cast, Optional

import pygame as pg

import celestial_mechanics as cm
import sprites


def game_loop(
    state: cm.State, recorded_actions: Optional[list[cm.Action]], with_graphics: bool = True
) -> tuple[cm.EpisodeTerminationReason, list[cm.Action]]:
  """Main graphics loop.

  Args:
    state: Initial state, will be modified.
    recorded_actions: Optional.
      If specified, replays the given actions (no user input allowed other than abort).
      If None, user input controls spaceship.
    with_graphics: If False, nothing will be shown on screen. Used to compute outcome of recorded_actions.

  Returns:
    termination_reason: Reason for game end.
    actions_taken: List of all actions taken during the game.
  """
  if with_graphics:
    screen = sprites.setup_screen()
    clock = pg.time.Clock()

    fast_update_sprites = pg.sprite.Group()
    slow_update_sprites = pg.sprite.Group()
    all_sprites = [
      sprites.CelestialBodySprite(body=cast(cm.CelestialBody, body), sprite_groups=[fast_update_sprites])
      for body in state.other_objects
    ]
    all_sprites.extend([
      sprites.CelestialBodySprite(
        body=state.target,
        sprite_groups=[fast_update_sprites],
      ),
      sprites.SpaceshipSprite(
        spaceship=state.spaceship,
        sprite_groups=[fast_update_sprites],
      ),
      sprites.InfoboxSprite(
        state=state,
        sprite_groups=[slow_update_sprites],
      ),
    ])

  actions_taken = []
  while True:
    if with_graphics:
      for event in pg.event.get():
        if event.type == pg.QUIT:
          return cm.EpisodeTerminationReason.USER_ABORT, actions_taken
        if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
          return cm.EpisodeTerminationReason.USER_ABORT, actions_taken

    if state.time > cm.EPISODE_TIME:
      print(f'Episode ended after {cm.EPISODE_TIME}')
      return cm.EpisodeTerminationReason.REACHED_TIME_LIMIT, actions_taken

    if recorded_actions:
      action = recorded_actions[state.n_updates]
    else:
      keys_pressed = pg.key.get_pressed()
      action = cm.Action(left_thruster=keys_pressed[pg.K_LEFT], right_thruster=keys_pressed[pg.K_RIGHT])

    actions_taken.append(action)
    state.spaceship.apply_action(action, time_step=cm.SEC_PER_FRAME)

    celestial_exceptions = state.update_positions(time_step=cm.SEC_PER_FRAME)
    state.update_returns(action, celestial_exceptions, time_step=cm.SEC_PER_FRAME)
    for collision_ex in celestial_exceptions:
      print(collision_ex)
      if isinstance(collision_ex.obj, cm.Spaceship):
        if isinstance(collision_ex, cm.CollisionException):
          return cm.EpisodeTerminationReason.SPACESHIP_COLLISION, actions_taken
        elif isinstance(collision_ex, cm.LostInSpaceException):
          return cm.EpisodeTerminationReason.SPACESHIP_LOST, actions_taken
        else:
          collision_ex.add_note("Unknown exception type!")
          raise collision_ex

    if with_graphics:
      fast_update_sprites.update()
      if state.n_updates % 10 == 0:
        slow_update_sprites.update()
      screen.fill(sprites.BACKGROUND_COLOR)
      fast_update_sprites.draw(screen)
      slow_update_sprites.draw(screen)
      pg.display.flip()
      clock.tick(cm.SEC_PER_FRAME * 1000)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_to', action='store', help='JSON filename to which the game will be recorded')
  parser.add_argument('--load_from', action='store', help='JSON filename from which the game will be read')
  parser.add_argument('--no_graphics', action='store_true', help='If specified, computes outcome without showing anything')
  args = parser.parse_args()

  if args.load_from:
    print(f"Reading episode from {args.load_from}...")
    with open(args.load_from, 'rt') as f:
      json_dict = json.load(f)
      episode = cm.RecordedEpisode.from_json_dict(json_dict)
      print(f"Expected game duration: {episode.final_state.time}")
      state = episode.initial_state
  else:
    state = cm.build_initial_state()

  if args.save_to:
    initial_state = copy.deepcopy(state)

  start_time = datetime.datetime.now()
  termination_reason, actions_taken = game_loop(
    state, episode.actions_taken if args.load_from else None, with_graphics=not args.no_graphics
  )
  end_time = datetime.datetime.now()

  print(f"Game duration: {state.time}")
  print(f'Final return: {state.rl_return:.3f}')
  if args.no_graphics:
    print(f'Computation time: {end_time - start_time}')
  pg.quit()

  if args.save_to:
    print(f'Saving to {args.save_to}...')
    with open(args.save_to, 'w') as f:
      episode = cm.RecordedEpisode(
        initial_state=initial_state,
        final_state=state,
        actions_taken=actions_taken,
        termination_reason=termination_reason
      )
      f.write(json.dumps(episode.to_json_dict(), indent=2))


if __name__ == '__main__':
  main()
