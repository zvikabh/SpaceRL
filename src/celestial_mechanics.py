"""Classes and functions for computing motion of celestial bodies."""

import dataclasses
import datetime
import enum
import math
from typing import Any, Optional, Sequence

import dacite
import pygame as pg


# Physical constants
UNIVERSE_RECT = pg.Rect(0, 0, 1280*1e6, 1024*1e6)
UNIVERSE_SIZE = max(UNIVERSE_RECT.width, UNIVERSE_RECT.height)
GRAVITATIONAL_CONSTANT = 6e-11
SCALE = 1e6  # meters per pixel

# Spaceship charactersitics
MAX_LANDING_SPEED = 10000  # m/sec
SPACESHIP_MASS = 1e6  # kg
SPACESHIP_THRUST = 1.2e14  # Newton
THRUSTER_COST_PER_SECOND = 0.03  # In reward units

# RL constants
DISCOUNT_FACTOR = 0.99  # TODO: Normalize per time_step
EPISODE_TIME = datetime.timedelta(seconds=60)
SEC_PER_FRAME = 0.04


class EpisodeTerminationReason(enum.Enum):
  USER_ABORT = 'user_abort'
  SPACESHIP_COLLISION = 'spaceship_collision'
  SPACESHIP_LOST = 'spaceship_lost'
  REACHED_TIME_LIMIT = 'reached_time_limit'


class CelestialException(Exception):
  """Catastrophic occurrence for one of the space objects."""

  def __init__(self, obj: 'SpaceObject') -> None:
    self.obj = obj

  def termination_reason(self):
    raise NotImplementedError()


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

  def termination_reason(self):
    return EpisodeTerminationReason.SPACESHIP_COLLISION


class LostInSpaceException(CelestialException):
  """Raised by an object if it has gone beyond the bounds of the known universe."""

  def __init__(self, obj: 'SpaceObject') -> None:
    super().__init__(obj)

  def __str__(self) -> str:
    return f"{self.obj.name} has been lost in space."

  def termination_reason(self):
    return EpisodeTerminationReason.SPACESHIP_LOST


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

  def to_vector(self) -> list[float]:
    return [self.x, self.y]

  @classmethod
  def from_vector(cls, vector: Sequence[float]) -> 'Vector':
    assert len(vector) == 2
    return cls(vector[0], vector[1])


@dataclasses.dataclass
class SpaceObject:
  mass: float
  position: Vector  # meters
  velocity: Vector  # meters/sec
  name: str

  @classmethod
  def state_vector_len(cls) -> int:
    return 4

  def to_vector(self) -> list[float]:
    normalized_pos = self.position / UNIVERSE_SIZE
    normalized_v = self.velocity / 1e9
    return normalized_pos.to_vector() + normalized_v.to_vector()

  def update_from_vector(self, vector: Sequence[float]) -> None:
    assert len(vector) == 4
    self.position = Vector.from_vector(vector[:2]) * UNIVERSE_SIZE
    self.velocity = Vector.from_vector(vector[2:]) * 1e9

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

  def to_char(self) -> str:
    return str(self.left_thruster * 2 + self.right_thruster)

  @classmethod
  def from_char(cls, s: str) -> 'Action':
    return cls.from_int(int(s))

  @classmethod
  def from_int(cls, n: int) -> 'Action':
    assert 0 <= n <= 3
    return cls(left_thruster=bool(n//2), right_thruster=bool(n%2))

  @property
  def cost_per_second(self) -> float:
    return (self.left_thruster + self.right_thruster) * THRUSTER_COST_PER_SECOND


@dataclasses.dataclass
class Spaceship(SpaceObject):
  angle: float  # Radians clockwise from x-axis
  angular_velocity: float  # Radians per second
  last_action: Optional[Action] = None  # used for plotting the thruster flames

  @classmethod
  def state_vector_len(cls) -> int:
    return super().state_vector_len() + 2

  def to_vector(self) -> list[float]:
    return super().to_vector() + [self.angle % (2*math.pi), self.angular_velocity]

  def update_from_vector(self, vector: Sequence[float]) -> None:
    assert len(vector) == self.state_vector_len()
    super().update_from_vector(vector[:4])
    self.angle = vector[4]
    self.angular_velocity = vector[5]

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
  other_objects: list[CelestialBody]
  time: datetime.timedelta = datetime.timedelta(seconds=0)
  fuel_cost: float = 0.
  rl_return: float = 0.
  rl_discounted_return: float = 0.
  n_updates: int = 0

  @property
  def all_objects(self) -> Sequence[SpaceObject]:
    return [self.spaceship, self.target] + self.other_objects

  def to_vector(self) -> list[float]:
    """Returns a vector containing normalized, RL-relevant portions of the state."""
    objects_state = sum((obj.to_vector() for obj in self.all_objects), start=[])
    return objects_state + [self.time.total_seconds() / 60]

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

  def update_returns(self, action: Action, celestial_exceptions: Sequence[CelestialException], time_step: float) -> float:
    dist = (self.spaceship.position - self.target.position).norm
    reward_per_sec = 1e6 / dist
    reward = reward_per_sec * time_step

    cur_fuel_cost = action.cost_per_second * time_step
    reward -= action.cost_per_second * time_step

    for collision_ex in celestial_exceptions:
      if isinstance(collision_ex.obj, Spaceship):
        reward -= 1  # Spaceship lost.

    self.fuel_cost += cur_fuel_cost
    self.rl_return += reward
    self.rl_discounted_return = self.rl_discounted_return * DISCOUNT_FACTOR + reward
    return reward


def build_initial_state() -> State:
  state = State(
    spaceship=Spaceship(
      mass=SPACESHIP_MASS,
      position=Vector(UNIVERSE_RECT.width / 2, UNIVERSE_RECT.height * 0.1),
      velocity=Vector(0, 0),
      name='Spaceship',
      angle=-math.pi * 0.5,
      angular_velocity=0,
    ),
    target=CelestialBody(
      mass=1.5e34,
      position=Vector(UNIVERSE_RECT.width / 2, UNIVERSE_RECT.height / 2),
      velocity=Vector(0, 0),
      name='Earth',
      radius=SCALE * 60,
      color=(0, 100, 255),
    ),
    other_objects=[
      CelestialBody(
        mass=1e33,
        position=Vector(UNIVERSE_RECT.width / 2, UNIVERSE_RECT.height * 0.2),
        velocity=Vector(5e7, 0),
        name='Luna',
        radius=SCALE * 20,
        color=(128, 128, 128),
      ),
    ]
  )
  # Keep the center of mass stationary
  state.target.velocity = state.other_objects[0].velocity * (-state.other_objects[0].mass / state.target.mass)
  return state


def _dict_factory(fields):
  """Custom dict factory that converts timedelta objects to seconds."""
  result = {}
  for key, value in fields:
    if isinstance(value, datetime.timedelta):
      result[key] = value.total_seconds()
    else:
      result[key] = value
  return result


@dataclasses.dataclass
class RecordedEpisode:
  initial_state: State
  final_state: Optional[State]
  actions_taken: list[Action]
  termination_reason: EpisodeTerminationReason

  def to_json_dict(self) -> dict[str, Any]:
    return {
      'initial_state': dataclasses.asdict(self.initial_state, dict_factory=_dict_factory),
      'final_state': dataclasses.asdict(self.final_state, dict_factory=_dict_factory),
      'actions_taken': ''.join(action.to_char() for action in self.actions_taken),
      'termination_reason': self.termination_reason.name
    }

  @classmethod
  def from_json_dict(cls, json_dict: dict[str, Any]) -> 'RecordedEpisode':
    dacite_config = dacite.Config(
      type_hooks={
        tuple[int, int, int]: tuple,
        datetime.timedelta: lambda s: datetime.timedelta(seconds=s),
      }
    )
    initial_state = dacite.from_dict(State, json_dict['initial_state'], config=dacite_config)
    if 'final_state' in json_dict:
      final_state = dacite.from_dict(State, json_dict['final_state'], config=dacite_config)
    else:
      final_state = None
    return RecordedEpisode(
      initial_state=initial_state,
      final_state=final_state,
      actions_taken=[Action.from_char(c) for c in json_dict['actions_taken']],
      termination_reason=EpisodeTerminationReason[json_dict['termination_reason']]
    )
