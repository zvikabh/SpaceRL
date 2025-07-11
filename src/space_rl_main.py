import dataclasses
import math
import os
from typing import Sequence

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
    self.image = pg.transform.rotate(self.orig_img, -self.ship.orientation*180/math.pi)
    self.rect = self.image.get_rect()
    self.rect.center = (self.ship.position.x / SCALE, self.ship.position.y / SCALE)


def setup_screen() -> pg.Surface:
  pg.init()
  winstyle = 0  # | pg.FULLSCREEN
  best_depth = pg.display.mode_ok(SCREEN_RECT.size, winstyle, 32)
  return pg.display.set_mode(SCREEN_RECT.size, winstyle, best_depth)


def setup_rocket_img() -> pg.Surface:
  rocket_img = pg.image.load(os.path.join('src', 'res', 'rocket.png')).convert()
  original_size = rocket_img.get_size()
  scale_factor = min(30 / original_size[0], 30 / original_size[1])
  new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
  return pg.transform.smoothscale(rocket_img, new_size)


def main():
  screen = setup_screen()
  rocket_img = setup_rocket_img()
  clock = pg.time.Clock()
  all_sprites = pg.sprite.Group()
  space_objects = [
    cm.CelestialBody(
      mass=1.5e34, 
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height / 2),
      velocity=cm.Vector(0, 0),
      name='planet',
      radius=SCALE*30,
      color=(0, 100, 255),
    ),
    cm.CelestialBody(
      mass=3e31,
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height * 0.2),
      velocity=cm.Vector(5e7, 0),
      name='moon',
      radius=SCALE*10,
      color=(128, 128, 128),
    ),
    cm.Spaceship(
      mass=1e6,
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height * 0.1),
      velocity=cm.Vector(0, 0),
      name='spaceship',
      orientation=math.pi*0.5,
    ),
  ]
  planet = CelestialBodySprite(
    body=space_objects[0],
    sprite_groups=[all_sprites],
  )
  moon = CelestialBodySprite(
    body=space_objects[1],
    sprite_groups=[all_sprites],
  )
  ship = SpaceshipSprite(
    spaceship=space_objects[2],
    rocket_img=rocket_img,
    sprite_groups=[all_sprites],
  )

  while True:
    for event in pg.event.get():
      if event.type == pg.QUIT:
        return
      if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
        return
      
    cm.update_positions(space_objects, time_step=SEC_PER_FRAME)
    all_sprites.update()
    screen.fill(BACKGROUND_COLOR)
    all_sprites.draw(screen)
    pg.display.flip()
    clock.tick(SEC_PER_FRAME * 1000)



if __name__ == '__main__':
  main()
  pg.quit()
