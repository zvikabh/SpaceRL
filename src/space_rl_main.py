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
    # print(f"Placed {self.body.name} at {self.body.position}")
    self.rect.center = (self.body.position.x / SCALE, self.body.position.y / SCALE)



def setup_screen():
  pg.init()
  winstyle = 0  # | pg.FULLSCREEN
  best_depth = pg.display.mode_ok(SCREEN_RECT.size, winstyle, 32)
  screen = pg.display.set_mode(SCREEN_RECT.size, winstyle, best_depth)
  return screen


def main():
  screen = setup_screen()
  clock = pg.time.Clock()
  all_sprites = pg.sprite.Group()
  celestial_bodies = [
    cm.CelestialBody(
      mass=1.5e34, 
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height / 2),
      velocity=cm.Vector(0, 0.),
      name='planet',
      radius=SCALE*30,
      color=(0, 100, 255),
    ),
    cm.CelestialBody(
      mass=3e31,
      position=cm.Vector(cm.UNIVERSE_RECT.width / 2, cm.UNIVERSE_RECT.height * 0.2),
      velocity=cm.Vector(5e7, 0.),
      name='moon',
      radius=SCALE*10,
      color=(128, 128, 128),
    )
  ]
  planet = CelestialBodySprite(
    body=celestial_bodies[0],
    sprite_groups=[all_sprites],
  )
  moon = CelestialBodySprite(
    body=celestial_bodies[1],
    sprite_groups=[all_sprites],
  )

  while True:
    for event in pg.event.get():
      if event.type == pg.QUIT:
        return
      if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
        return
      
    cm.update_positions(celestial_bodies, time_step=SEC_PER_FRAME)
    all_sprites.update()
    screen.fill(BACKGROUND_COLOR)
    all_sprites.draw(screen)
    pg.display.flip()
    clock.tick(SEC_PER_FRAME * 1000)



if __name__ == '__main__':
  main()
  pg.quit()
