import math
import os
import time
from typing import Sequence

import pygame as pg

import celestial_mechanics as cm


# Display constants
SCALE = 1e6  # meters per pixel
SCREEN_RECT = pg.Rect(0, 0, cm.UNIVERSE_RECT.width/SCALE, cm.UNIVERSE_RECT.height/SCALE)
BACKGROUND_COLOR = (0, 0, 0)


class CelestialBodySprite(pg.sprite.Sprite):

  def __init__(self, body: cm.CelestialBody, sprite_groups: Sequence[pg.sprite.Group]):
    super().__init__(*sprite_groups)
    self.body = body
    diameter_pixels = int(self.body.radius * 2 / SCALE + 1)
    self.image = pg.Surface((diameter_pixels, diameter_pixels), pg.SRCALPHA)
    pg.draw.circle(
      self.image, self.body.color, (self.body.radius / SCALE, self.body.radius / SCALE), self.body.radius / SCALE
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
    self.image = pg.transform.rotate(self.orig_img, -self.ship.angle * 180 / math.pi - 90)
    self.rect = self.image.get_rect()
    self.rect.center = (self.ship.position.x / SCALE, self.ship.position.y / SCALE)


class InfoboxSprite(pg.sprite.Sprite):
  NUM_LINES = 4
  LINE_HEIGHT = 20
  LINE_WIDTH = 250

  def __init__(self, state: cm.State, sprite_groups: Sequence[pg.sprite.Group]):
    super().__init__(*sprite_groups)
    self.state = state
    self.last_update_time = None
    self.last_shown_frame = 0
    self.font = pg.font.Font(None, 20)
    self.color = (128, 128, 128)
    self.image = pg.Surface(
      (InfoboxSprite.LINE_WIDTH, InfoboxSprite.LINE_HEIGHT * InfoboxSprite.NUM_LINES), pg.SRCALPHA
    )
    self.update()
    self.rect = self.image.get_rect().move(10, 10)

  def update(self):
    self.image.fill((0, 0, 0))
    self.blit_text_line(f"Return: {self.state.rl_return:.3f}", 0)
    self.blit_text_line(f"Velocity: {self.state.spaceship.velocity.norm / 1000:.0f} km/s", 1)
    dist_from_target = (self.state.spaceship.position - self.state.target.position).norm
    self.blit_text_line(f"Distance from Target: {dist_from_target / 1000:.0f} km", 2)
    if self.last_update_time:
      cur_time = time.time()
      fps = (self.state.n_updates - self.last_shown_frame) / (cur_time - self.last_update_time)
      self.blit_text_line(f"FPS: {fps:.1f}", 3)
      self.last_update_time = cur_time
      self.last_shown_frame = self.state.n_updates
    else:
      self.last_update_time = time.time()

  def blit_text_line(self, text: str, n_row: int):
    surf = self.font.render(text, True, self.color)
    self.image.blit(surf, (0, n_row * InfoboxSprite.LINE_HEIGHT))


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
