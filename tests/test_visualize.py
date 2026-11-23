"""Тесты визуализатора без открытия окна и без pygame."""

from __future__ import annotations

import pytest

from dino_master import perception, world
from dino_master.render import Pilot, parse_args, step_world


class FakeNetwork:
    def __init__(self, outputs: list[float]) -> None:
        self.outputs = outputs

    def activate(self, values: list[float]) -> list[float]:
        assert len(values) == perception.FEATURE_COUNT
        return self.outputs


def observation(*, obstacle: bool = False, jumping: bool = False) -> dict:
    empty = {
        "id": None,
        "distance": 1000.0,
        "type": None,
        "yPos": None,
        "width": 0.0,
        "speed": 6.0,
    }
    first = (
        {
            "id": 1,
            "distance": 100.0,
            "type": "CACTUS_SMALL",
            "yPos": 105.0,
            "width": 17.0,
            "speed": 6.0,
        }
        if obstacle
        else dict(empty)
    )
    return {
        "status": "OK",
        "first": first,
        "second": dict(empty),
        "speed": 6.0,
        "dino": {
            "jumping": jumping,
            "ducking": False,
            "yPos": 93.0,
            "speedDrop": False,
        },
    }


def test_pilot_jumps_and_respects_cooldown() -> None:
    pilot = Pilot(FakeNetwork([0.0, 2.0, 0.0]))

    assert pilot.act(observation(obstacle=True)) == 1
    assert pilot.jump_cooldown == perception.JUMP_COOLDOWN_STEPS

    # В воздухе прыжок запрещён маской, поэтому выбирается 0.
    assert pilot.act(observation(obstacle=True, jumping=True)) == 0


def test_pilot_holds_duck_for_minimum_steps() -> None:
    pilot = Pilot(FakeNetwork([0.0, 0.0, 1.0]))

    assert pilot.act(observation(obstacle=True)) == 2
    assert pilot.duck_remaining == perception.DUCK_HOLD_STEPS - 1

    pilot.network = FakeNetwork([0.0, 0.0, 0.0])
    assert pilot.act(observation(obstacle=True)) == 2
    assert pilot.duck_remaining == perception.DUCK_HOLD_STEPS - 2


def test_step_world_advances_and_stops_when_dead() -> None:
    world = world.SimWorld(
        world.SimConfig(dino_count=1, max_steps=50, speed=6.0, seed=1)
    )
    pilot = Pilot(FakeNetwork([0.0, 0.0, 0.0]))

    assert step_world(world, pilot) is True
    assert world.steps == [1]

    world.alive[0] = False
    assert step_world(world, pilot) is False


def test_parse_args_accepts_visualization_options() -> None:
    args = parse_args(["--scale", "3", "--speed", "6.0", "--max-frames", "10"])

    assert args.scale == 3
    assert args.speed == pytest.approx(6.0)
    assert args.max_frames == 10
