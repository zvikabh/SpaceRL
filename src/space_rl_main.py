import math
import os
import time
from typing import cast, Sequence

import pygame as pg

import celestial_mechanics as cm

# Display constants
SCALE = 1e6  # meters per pixel
SCREEN_RECT = pg.Rect(0, 0, cm.UNIVERSE_RECT.width/SCALE, cm.UNIVERSE_RECT.height/SCALE)
BACKGROUND_COLOR = (0, 0, 0)
SEC_PER_FRAME = 0.04


class CelestialBodySprite(pg.sprite.Sprite):

  def __init__(self, body: cm.CelestialBody, sprite_groups: Sequence[pg.sprite.Group]):
    super().__init__(*sprite_groups)
    self.body = body
    diameter_pixels = int(self.body.radius * 2 / SCALE + 1)
    self.image = pg.Surface((diameter_pixels, diameter_pixels), pg.SRCALPHA)
    pg.draw.circle(
      self.image, self.body.color, (self.body.radius/SCALE, self.body.radius/SCALE), self.body.radius/SCALE
    )
    self.rect = self.image.get_rect()
    self.update()
  
  def update(self):
    self.rect.center = (self.body.position.x / SCALE, self.body.position.y / SCALE)


class SpaceshipSprite(pg.sprite.Sprite):

  def __init__(self, spaceship: cm.Spaceship, rocket_img: pg.Surface, sprite_groups: Sequence[pg.sprite.Group]):
    super().__init__(*sprite_groups)
    self.ship = spaceship
    self.orig_img = rocket_img
    self.image = self.orig_img.copy()
    self.rect = self.image.get_rect()
    self.update()

  def update(self):
    self.image = pg.transform.rotate(self.orig_img, -self.ship.angle*180/math.pi - 90)
    self.rect = self.image.get_rect()
    self.rect.center = (self.ship.position.x / SCALE, self.ship.position.y / SCALE)


class InfoboxSprite(pg.sprite.Sprite):

  NUM_LINES = 4
  LINE_HEIGHT = 20
  LINE_WIDTH = 250

  def __init__(self, state: cm.State, sprite_groups: Sequence[pg.sprite.Group]):
    super().__init__(*sprite_groups)
    self.state = state
    self.font = pg.font.Font(None, 20)
    self.color = (128, 128, 128)
    self.image = pg.Surface(
      (InfoboxSprite.LINE_WIDTH, InfoboxSprite.LINE_HEIGHT*InfoboxSprite.NUM_LINES), pg.SRCALPHA
    )
    self.update()
    self.rect = self.image.get_rect().move(10, 10)

  def update(self):
    self.image.fill((0,0,0))
    self.blit_text_line(f"Return: {self.state.rl_return:.3f}", 0)
    self.blit_text_line(f"Velocity: {self.state.spaceship.velocity.norm/1000:.0f} km/s", 1)
    dist_from_target = (self.state.spaceship.position - self.state.target.position).norm
    self.blit_text_line(f"Distance from Target: {dist_from_target/1000:.0f} km", 2)

  def blit_text_line(self, text: str, n_row: int):
    surf = self.font.render(text, True, self.color)
    self.image.blit(surf, (0, n_row*InfoboxSprite.LINE_HEIGHT))



def setup_screen() -> pg.Surface:
  pg.init()
  winstyle = 0  # | pg.FULLSCREEN
  best_depth = pg.display.mode_ok(SCREEN_RECT.size, winstyle, 32)
  return pg.display.set_mode(SCREEN_RECT.size, winstyle, best_depth)


def setup_rocket_img() -> pg.Surface:
  main_dir = os.path.split(os.path.abspath(__file__))[0]
  rocket_img = pg.image.load(os.path.join(main_dir, 'res', 'rocket.png')).convert()
  original_size = rocket_img.get_size()
  scale_factor = min(30 / original_size[0], 30 / original_size[1])
  new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
  return pg.transform.smoothscale(rocket_img, new_size)


def main():
  screen = setup_screen()
  rocket_img = setup_rocket_img()
  clock = pg.time.Clock()
  state = cm.State(
    spaceship=cm.Spaceship(
      mass=cm.SPACESHIP_MASS,
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height * 0.1),
      velocity=cm.Vector(0, 0),
      name='Spaceship',
      angle=-math.pi*0.5,
      angular_velocity=0,
    ),
    target=cm.CelestialBody(
      mass=1.5e34, 
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height / 2),
      velocity=cm.Vector(0, 0),
      name='Earth',
      radius=SCALE*60,
      color=(0, 100, 255),
    ),
    other_objects=[
      cm.CelestialBody(
        mass=3e31,
        position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height * 0.2),
        velocity=cm.Vector(5e7, 0),
        name='Luna',
        radius=SCALE*10,
        color=(128, 128, 128),
      ),
    ]
  )

  fast_update_sprites = pg.sprite.Group()
  slow_update_sprites = pg.sprite.Group()

  sprites = [
    CelestialBodySprite(body=cast(cm.CelestialBody, body), sprite_groups=[fast_update_sprites])
    for body in state.other_objects
  ]
  sprites.extend([
    CelestialBodySprite(
      body=state.target,
      sprite_groups=[fast_update_sprites],
    ),
    SpaceshipSprite(
      spaceship=state.spaceship,
      rocket_img=rocket_img,
      sprite_groups=[fast_update_sprites],
    ),
    InfoboxSprite(
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

    collided_objects = state.update_positions(time_step=SEC_PER_FRAME)
    state.update_returns(time_step=SEC_PER_FRAME)
    for collision_ex in collided_objects:
      if isinstance(collision_ex.smaller_obj, cm.Spaceship):
        if collision_ex.smaller_obj.velocity.norm < cm.MAX_LANDING_SPEED: 
          print(
            f"Victory! You have successfully landed on {collision_ex.larger_obj.name} "
            f"with landing velocity {collision_ex.smaller_obj.velocity.norm/1e3:.1f} m/s")
        else:
          print(f"Your spaceship has collided with {collision_ex.larger_obj.name} "
                f"with impact velocity {collision_ex.smaller_obj.velocity.norm/1e3:.1f} km/s")
        print(f"Final returns: {state.rl_return:.3f}")
        return
      else:
        print(collision_ex)
    fast_update_sprites.update()
    if state.n_updates % 10 == 0:
      slow_update_sprites.update()
    screen.fill(BACKGROUND_COLOR)
    fast_update_sprites.draw(screen)
    slow_update_sprites.draw(screen)
    pg.display.flip()
    clock.tick(SEC_PER_FRAME * 1000)


if __name__ == '__main__':
  main()
  pg.quit()
