"""Classes and functions for computing motion of celestial bodies."""

import dataclasses
import math
from typing import Sequence

import pygame as pg


# Physical constants
UNIVERSE_RECT = pg.Rect(0, 0, 1280*1e6, 1024*1e6)
GRAVITATIONAL_CONSTANT = 6e-11

# Spaceship charactersitics
MAX_LANDING_SPEED = 1000  # m/sec
SPACESHIP_MASS = 1e6  # kg
SPACESHIP_THRUST = 1e12  # Newton


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
  name: str


@dataclasses.dataclass
class CelestialBody(SpaceObject):
  radius: float  # meters
  color: tuple[int, int, int]


@dataclasses.dataclass
class Spaceship(SpaceObject):
  orientation: float  # Radians clockwise from x-axis

  def fire_thrusters(self, left_thruster: bool, right_thruster: bool, time_step: float) -> None:
    if left_thruster and not right_thruster:
      self.orientation -= 0.02
    elif right_thruster and not left_thruster:
      self.orientation += 0.02

    thrust = (left_thruster + right_thruster)/2 * SPACESHIP_THRUST
    acceleration = thrust / self.mass
    delta_v_norm = acceleration * time_step
    delta_v = Vector(delta_v_norm * math.cos(self.orientation), delta_v_norm * math.sin(self.orientation))
    self.velocity = self.velocity + delta_v

    # print(f"{left_thruster=}, {right_thruster=}, "
    #       f"Orientation: {self.orientation*180/math.pi:.0f} degrees, "
    #       f"delta_v=({delta_v.x:.0f}, {delta_v.y:.0f}), "
    #       f"v=({self.velocity.x:.0f}, {self.velocity.y:.0f})")


def update_positions(
    objects: Sequence[SpaceObject], time_step: float
) -> list[tuple[SpaceObject, SpaceObject]]:
  """
  Updates positions of all space objects.
  Returns a list of collided object tuples (smaller object first).
  """
  collided_objects = []
  for i, this_obj in enumerate(objects):
    total_accel = Vector(0., 0.)
    for j, other_obj in enumerate(objects):
      if i == j:
        continue
      delta: Vector = other_obj.position - this_obj.position
      if isinstance(other_obj, CelestialBody) and this_obj.mass < other_obj.mass:
        if isinstance(this_obj, CelestialBody):
          collision_dist = other_obj.radius + this_obj.radius
        else:
          collision_dist = other_obj.radius
        if delta.squared_norm <= collision_dist*collision_dist:
          collided_objects.append((this_obj, other_obj))
      if this_obj.mass > 1e6 * other_obj.mass:
        continue  # negligible gravitational effect
      cur_accel = (GRAVITATIONAL_CONSTANT * other_obj.mass / delta.norm / delta.squared_norm) * delta
      total_accel = total_accel + cur_accel
    this_obj.velocity = this_obj.velocity + total_accel * time_step
    this_obj.position = this_obj.position + this_obj.velocity * time_step
  return collided_objects
