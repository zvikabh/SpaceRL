import math
from typing import cast

import pygame as pg

import celestial_mechanics as cm
import sprites


SEC_PER_FRAME = 0.04


def build_state() -> cm.State:
  state = cm.State(
    spaceship=cm.Spaceship(
      mass=cm.SPACESHIP_MASS,
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height * 0.1),
      velocity=cm.Vector(0, 0),
      name='Spaceship',
      angle=-math.pi * 0.5,
      angular_velocity=0,
    ),
    target=cm.CelestialBody(
      mass=1.5e34,
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height / 2),
      velocity=cm.Vector(0, 0),
      name='Earth',
      radius=sprites.SCALE * 60,
      color=(0, 100, 255),
    ),
    other_objects=[
      cm.CelestialBody(
        mass=5e32,
        position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height * 0.2),
        velocity=cm.Vector(5e7, 0),
        name='Luna',
        radius=sprites.SCALE * 20,
        color=(128, 128, 128),
      ),
    ]
  )
  # Keep the center of mass stationary
  state.target.velocity = state.other_objects[0].velocity * (-state.other_objects[0].mass / state.target.mass)
  return state


def graphics_loop(state: cm.State) -> None:
  screen = sprites.setup_screen()
  rocket_img = sprites.setup_rocket_img()
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
      rocket_img=rocket_img,
      sprite_groups=[fast_update_sprites],
    ),
    sprites.InfoboxSprite(
      state=state,
      sprite_groups=[slow_update_sprites],
    ),
  ])

  while True:
    for event in pg.event.get():
      if event.type == pg.QUIT:
        return
      if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
        return

    keys_pressed = pg.key.get_pressed()
    fire_left = keys_pressed[pg.K_LEFT]
    fire_right = keys_pressed[pg.K_RIGHT]
    state.spaceship.fire_thrusters(
        left_thruster=fire_left, right_thruster=fire_right, time_step=SEC_PER_FRAME
    )

    celestial_exceptions = state.update_positions(time_step=SEC_PER_FRAME)
    state.update_returns(time_step=SEC_PER_FRAME)
    for collision_ex in celestial_exceptions:
      print(collision_ex)
      if isinstance(collision_ex.obj, cm.Spaceship):
        return
    fast_update_sprites.update()
    if state.n_updates % 10 == 0:
      slow_update_sprites.update()
    screen.fill(sprites.BACKGROUND_COLOR)
    fast_update_sprites.draw(screen)
    slow_update_sprites.draw(screen)
    pg.display.flip()
    clock.tick(SEC_PER_FRAME * 1000)


def main():
  state = build_state()
  graphics_loop(state)
  print(f"Final return: {state.rl_return:.3f}")


if __name__ == '__main__':
  main()
  pg.quit()
