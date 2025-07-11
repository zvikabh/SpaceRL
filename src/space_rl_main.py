import dataclasses
import math
from typing import Sequence

import pygame as pg

# Display constants
SCREEN_RECT = pg.Rect(0, 0, 1280, 1024)
SCALE = 1e6  # meters per pixel
BACKGROUND_COLOR = (0, 0, 0)
SEC_PER_FRAME = 0.04

# Physical constants
UNIVERSE_RECT = pg.Rect(0, 0, SCREEN_RECT.width * SCALE, SCREEN_RECT.height * SCALE)
GRAVITATIONAL_CONSTANT = 6e-11


@dataclasses.dataclass
class Vector:
  x: float
  y: float

  def __sub__(self, other: 'Vector') -> 'Vector':
    return Vector(self.x - other.x, self.y - other.y)
  
  def __add__(self, other: 'Vector') -> 'Vector':
    return Vector(self.x + other.x, self.y + other.y)

  def __mul__(self, scalar: float):
    return Vector(self.x * scalar, self.y * scalar)
  
  def __rmul__(self, scalar: float):
    return Vector(self.x * scalar, self.y * scalar)
  
  def __truediv__(self, scalar):
    return Vector(self.x / scalar, self.y / scalar)

  @property
  def squared_norm(self) -> float:
    return self.x * self.x + self.y * self.y
  
  @property
  def norm(self) -> float:
    return math.sqrt(self.squared_norm)


@dataclasses.dataclass
class SpaceObject:
  mass: float
  position: Vector  # meters
  velocity: Vector  # meters/sec


@dataclasses.dataclass
class CelestialBody(SpaceObject):
  name: str
  radius: float  # meters
  color: tuple[int, int, int]


def update_positions(objects: Sequence[SpaceObject]) -> None:
  for i, this_obj in enumerate(objects):
    total_accel = Vector(0., 0.)
    for j, other_obj in enumerate(objects):
      if i == j:
        continue
      delta: Vector = other_obj.position - this_obj.position
      cur_accel = (GRAVITATIONAL_CONSTANT * other_obj.mass / delta.norm / delta.squared_norm) * delta
      total_accel = total_accel + cur_accel
    this_obj.velocity = this_obj.velocity + total_accel * SEC_PER_FRAME
    this_obj.position = this_obj.position + this_obj.velocity * SEC_PER_FRAME


class CelestialBodySprite(pg.sprite.Sprite):

  def __init__(self, body: CelestialBody, sprite_groups: Sequence[pg.sprite.Group]):
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
    CelestialBody(
      mass=1.5e34, 
      position=Vector(UNIVERSE_RECT.width / 2, UNIVERSE_RECT.height / 2),
      velocity=Vector(0, 0.),
      name='planet',
      radius=SCALE*30,
      color=(0, 100, 255),
    ),
    CelestialBody(
      mass=3e31,
      position=Vector(UNIVERSE_RECT.width / 2, UNIVERSE_RECT.height * 0.2),
      velocity=Vector(5e7, 0.),
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
      
    update_positions(celestial_bodies)
    all_sprites.update()
    screen.fill(BACKGROUND_COLOR)
    all_sprites.draw(screen)
    pg.display.flip()
    clock.tick(SEC_PER_FRAME * 1000)



if __name__ == '__main__':
  main()
  pg.quit()
