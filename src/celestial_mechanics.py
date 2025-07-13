"""Classes and functions for computing motion of celestial bodies."""

import dataclasses
import datetime
import math
from typing import Optional, Sequence

import pygame as pg


# Physical constants
UNIVERSE_RECT = pg.Rect(0, 0, 1280*1e6, 1024*1e6)
GRAVITATIONAL_CONSTANT = 6e-11

# Spaceship charactersitics
MAX_LANDING_SPEED = 10000  # m/sec
SPACESHIP_MASS = 1e6  # kg
SPACESHIP_THRUST = 1.2e14  # Newton

# RL constants
DISCOUNT_FACTOR = 0.99  # TODO: Normalize per time_step


class CelestialException(Exception):
  """Catastrophic occurrence for one of the space objects."""

  def __init__(self, obj: 'SpaceObject') -> None:
    self.obj = obj


class CollisionException(CelestialException):
  """Raised by an object if it has collided with another obejct."""
  
  def __init__(self, smaller_obj: 'SpaceObject', larger_obj: 'SpaceObject') -> None:
    super().__init__(smaller_obj)
    self.larger_obj = larger_obj
  
  def __str__(self) -> str:
    return (
      f"{self.obj.name} has collided with {self.larger_obj.name} "
      f"with impact velocity {self.obj.velocity.norm/1e3:.0f} km/s."
    )


class LostInSpaceException(CelestialException):
  """Raised by an object if it has gone beyond the bounds of the known universe."""

  def __init__(self, obj: 'SpaceObject') -> None:
    super().__init__(obj)

  def __str__(self) -> str:
    return f"{self.obj.name} has been lost in space."


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

    if (self.position.x < 0 or self.position.y < 0 or
        self.position.x > UNIVERSE_RECT.width or self.position.y > UNIVERSE_RECT.height):
      raise LostInSpaceException(self)


@dataclasses.dataclass
class CelestialBody(SpaceObject):
  radius: float  # meters
  color: tuple[int, int, int]


@dataclasses.dataclass
class Action:
  left_thruster: bool
  right_thruster: bool


@dataclasses.dataclass
class Spaceship(SpaceObject):
  angle: float  # Radians clockwise from x-axis
  angular_velocity: float  # Radians per second
  last_action: Optional[Action] = None  # used for plotting the thruster flames

  def update_self(self, all_objects: Sequence[SpaceObject], time_step: float) -> None:
    self.angle += self.angular_velocity * time_step
    super().update_self(all_objects, time_step)

  def apply_action(self, action: Action, time_step: float) -> None:
    self.last_action = action
    if action.left_thruster and not action.right_thruster:
      self.angular_velocity -= 0.05
    elif action.right_thruster and not action.left_thruster:
      self.angular_velocity += 0.05

    thrust = (action.left_thruster + action.right_thruster)/2 * SPACESHIP_THRUST
    acceleration = thrust / self.mass
    delta_v_norm = acceleration * time_step
    delta_v = Vector(delta_v_norm * math.cos(self.angle), delta_v_norm * math.sin(self.angle))
    self.velocity = self.velocity + delta_v


@dataclasses.dataclass
class State:
  spaceship: Spaceship
  target: CelestialBody
  other_objects: list[SpaceObject]
  time: datetime.timedelta = datetime.timedelta(seconds=0)
  rl_return: float = 0.
  rl_discounted_return: float = 0.
  n_updates: int = 0

  @property
  def all_objects(self) -> Sequence[SpaceObject]:
    return [self.spaceship, self.target] + self.other_objects

  def update_positions(self, time_step: float) -> list[CelestialException]:
    """
    Updates positions of all space objects.
    Returns a list of celestial exceptions (objects lost in space or collided).
    """
    self.n_updates += 1
    self.time += datetime.timedelta(seconds=time_step)
    celestial_exceptions = []
    for this_obj in self.all_objects:
      try:
        this_obj.update_self(self.all_objects, time_step)
      except CelestialException as ex:
        celestial_exceptions.append(ex)
    return celestial_exceptions

  def update_returns(self, time_step: float) -> None:
    dist = (self.spaceship.position - self.target.position).norm
    reward_per_sec = 1e6 / dist
    reward = reward_per_sec * time_step
    self.rl_return += reward
    self.rl_discounted_return = self.rl_discounted_return * DISCOUNT_FACTOR + reward
