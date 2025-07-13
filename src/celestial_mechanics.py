"""Classes and functions for computing motion of celestial bodies."""

import dataclasses
import math
from typing import Sequence

import pygame as pg


# Physical constants
UNIVERSE_RECT = pg.Rect(0, 0, 1280*1e6, 1024*1e6)
GRAVITATIONAL_CONSTANT = 6e-11

# Spaceship charactersitics
MAX_LANDING_SPEED = 10000  # m/sec
SPACESHIP_MASS = 1e6  # kg
SPACESHIP_THRUST = 1.2e14  # Newton


class CollisionException(Exception):
  """Raised by an object if it has collided with another obejct."""
  
  def __init__(self, smaller_obj: 'SpaceObject', larger_obj: 'SpaceObject') -> None:
    self.smaller_obj = smaller_obj
    self.larger_obj = larger_obj
  
  def __str__(self):
    return f"{self.smaller_obj.name} has collided with {self.larger_obj.name}"


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

  def update_self(self, all_objects: Sequence['SpaceObject'], time_step: float) -> None:
    """
    Updates the position of this object.
    Raises a CollisionException if the object has collided with a larger object.
    """
    total_accel = Vector(0., 0.)
    for other_obj in all_objects:
      if other_obj == self:
        continue
      delta: Vector = other_obj.position - self.position

      # Check for collision
      if isinstance(other_obj, CelestialBody) and self.mass < other_obj.mass:
        if isinstance(self, CelestialBody):
          collision_dist = other_obj.radius + self.radius
        else:
          collision_dist = other_obj.radius
        if delta.squared_norm <= collision_dist*collision_dist:
          raise CollisionException(self, other_obj)

      # Apply gravitational pull
      if self.mass > 1e6 * other_obj.mass:
        continue  # negligible gravitational effect
      cur_accel = (GRAVITATIONAL_CONSTANT * other_obj.mass / delta.norm / delta.squared_norm) * delta
      total_accel = total_accel + cur_accel

    self.velocity = self.velocity + total_accel * time_step
    self.position = self.position + self.velocity * time_step


@dataclasses.dataclass
class CelestialBody(SpaceObject):
  radius: float  # meters
  color: tuple[int, int, int]


@dataclasses.dataclass
class Spaceship(SpaceObject):
  angle: float  # Radians clockwise from x-axis
  angular_velocity: float  # Radians per second

  def update_self(self, all_objects, time_step):
    self.angle += self.angular_velocity * time_step
    super().update_self(all_objects, time_step)

  def fire_thrusters(self, left_thruster: bool, right_thruster: bool, time_step: float) -> None:
    if left_thruster and not right_thruster:
      self.angular_velocity -= 0.05
    elif right_thruster and not left_thruster:
      self.angular_velocity += 0.05

    thrust = (left_thruster + right_thruster)/2 * SPACESHIP_THRUST
    acceleration = thrust / self.mass
    delta_v_norm = acceleration * time_step
    delta_v = Vector(delta_v_norm * math.cos(self.angle), delta_v_norm * math.sin(self.angle))
    self.velocity = self.velocity + delta_v

    # print(f"{left_thruster=}, {right_thruster=}, "
    #       f"Angle: {self.angle*180/math.pi:.0f} degrees, "
    #       f"delta_v=({delta_v.x:.0f}, {delta_v.y:.0f}), "
    #       f"v=({self.velocity.x:.0f}, {self.velocity.y:.0f})")


def update_positions(
    objects: Sequence[SpaceObject], time_step: float
) -> list[CollisionException]:
  """
  Updates positions of all space objects.
  Returns a list of collided objects.
  """
  collided_objects = []
  for this_obj in objects:
    try:
      this_obj.update_self(objects, time_step)
    except CollisionException as ex:
      collided_objects.append(ex)
  return collided_objects
