"""Unit-тесты симулятора Dino без браузера."""

from __future__ import annotations

import random

import pytest

from dino_master import perception, world as world_module


def make_obstacle(
    type_config: world_module.ObstacleType,
    *,
    x: float,
    y: int,
    size: int = 1,
) -> world_module.Obstacle:
    obstacle = world_module.Obstacle(type_config, random.Random(0), 6.0, 1)
    obstacle.x = x
    obstacle.y = y
    obstacle.size = size
    obstacle.width = type_config.width * size
    obstacle.collision_boxes = [
        world_module.CollisionBox(box.x, box.y, box.width, box.height)
        for box in type_config.collision_boxes
    ]
    if size > 1:
        obstacle.collision_boxes[1].width = (
            obstacle.width
            - obstacle.collision_boxes[0].width
            - obstacle.collision_boxes[2].width
        )
        obstacle.collision_boxes[2].x = (
            obstacle.width - obstacle.collision_boxes[2].width
        )
    return obstacle


def test_js_round_matches_javascript() -> None:
    assert world_module.js_round(0.5) == 1
    assert world_module.js_round(-0.5) == 0
    assert world_module.js_round(1.49) == 1
    assert world_module.js_round(-1.51) == -2


def test_jump_apex_and_landing() -> None:
    dino = world_module.TRex()
    dino.start_jump(6.0)
    min_y = dino.y
    for _ in range(120):
        if not dino.jumping:
            break
        dino.update_frame()
        min_y = min(min_y, dino.y)
    assert min_y >= 0.0
    assert min_y < 40.0
    assert dino.y == pytest.approx(world_module.GROUND_Y)
    assert dino.jumping is False
    assert dino.jump_count == 1


def test_speed_drop_falls_faster_and_lands() -> None:
    dino = world_module.TRex()
    dino.start_jump(6.0)
    dino.update_frame()
    normal_y = dino.y
    dino.set_speed_drop()
    dino.update_frame()
    assert dino.y >= normal_y

    for _ in range(120):
        if not dino.jumping:
            break
        dino.update_frame()
    assert dino.y == pytest.approx(world_module.GROUND_Y)
    assert dino.speed_drop is False


def test_held_speed_drop_does_not_reset_velocity() -> None:
    dino = world_module.TRex()
    dino.start_jump(6.0)
    dino.update_frame()
    dino.apply_action(2, 6.0)
    assert dino.vy == pytest.approx(2.5)
    dino.update_frame()
    velocity = dino.vy
    dino.apply_action(2, 6.0)
    assert dino.vy == pytest.approx(velocity)


def test_duck_does_not_get_cleared_on_ground() -> None:
    dino = world_module.TRex()
    dino.apply_action(2, 6.0)
    assert dino.ducking is True
    # update_frame относится только к физике в воздухе; присед на земле сохраняется.
    dino.apply_action(0, 6.0)
    assert dino.ducking is False


def test_standing_dino_hits_cactus() -> None:
    dino = world_module.TRex()
    obstacle = make_obstacle(world_module.CACTUS_SMALL, x=50, y=105)
    assert world_module.check_collision(obstacle, dino) is True


def test_ducking_dino_clears_high_bird() -> None:
    dino = world_module.TRex()
    dino.set_duck(True)
    obstacle = make_obstacle(world_module.PTERODACTYL, x=50, y=50)
    assert world_module.check_collision(obstacle, dino) is False


def test_world_spawns_first_obstacle_after_clear_time() -> None:
    early = world_module.SimWorld(
        world_module.SimConfig(dino_count=1, max_steps=200, speed=6.0, clear_time_ms=3000)
    )
    for _ in range(10):
        early.step([0])
    assert early.obstacles == []

    late = world_module.SimWorld(
        world_module.SimConfig(dino_count=1, max_steps=200, speed=6.0, clear_time_ms=0)
    )
    late.step([0])
    assert len(late.obstacles) == 1
    assert late.obstacles[0].x > world_module.WIDTH


def test_world_kills_dino_on_cactus_and_stops_steps() -> None:
    sim_world = world_module.SimWorld(
        world_module.SimConfig(dino_count=1, max_steps=200, speed=6.0, clear_time_ms=0)
    )
    obstacle = make_obstacle(world_module.CACTUS_SMALL, x=world_module.TREX_X, y=105)
    sim_world.obstacles = [obstacle]
    sim_world.step([0])
    assert sim_world.alive == [False]
    steps_at_death = sim_world.steps[0]
    sim_world.step([0])
    assert sim_world.steps[0] == steps_at_death


def test_world_observation_format() -> None:
    sim_world = world_module.SimWorld(
        world_module.SimConfig(dino_count=1, max_steps=200, speed=6.0, clear_time_ms=0)
    )
    sim_world.step([0])
    obs = sim_world.observation(0)
    assert obs["status"] == "OK"
    assert set(obs["first"]) == {"id", "distance", "type", "yPos", "width", "speed"}
    assert set(obs["second"]) == set(obs["first"])
    assert set(obs["dino"]) == {"jumping", "ducking", "yPos", "speedDrop"}


def test_world_clears_are_counted_when_obstacle_passes() -> None:
    sim_world = world_module.SimWorld(
        world_module.SimConfig(dino_count=1, max_steps=200, speed=6.0, clear_time_ms=0)
    )
    obstacle = make_obstacle(world_module.CACTUS_SMALL, x=world_module.TREX_X + 100, y=105)
    sim_world.obstacles = [obstacle]
    sim_world.last_first_id = [obstacle.id]
    obstacle.x = -1000
    sim_world.step([0])
    assert sim_world.clears[0] == 1


def test_observation_is_compatible_with_features() -> None:
    sim_world = world_module.SimWorld(
        world_module.SimConfig(dino_count=1, max_steps=200, speed=6.0, clear_time_ms=0)
    )
    sim_world.step([0])
    values = perception.observation_features(sim_world.observation(0))
    assert len(values) == perception.FEATURE_COUNT == 24


def test_dead_dino_observation_is_marked_dead() -> None:
    sim_world = world_module.SimWorld(
        world_module.SimConfig(dino_count=1, max_steps=200, speed=6.0, clear_time_ms=0)
    )
    sim_world.alive[0] = False
    obs = sim_world.observation(0)
    assert obs["status"] == "DEAD"
    assert obs["first"]["distance"] == 1000


def test_world_distance_matches_speed_per_frame() -> None:
    sim_world = world_module.SimWorld(
        world_module.SimConfig(dino_count=1, max_steps=200, speed=6.0, clear_time_ms=3000)
    )
    sim_world.step([0])
    assert sim_world.distance == pytest.approx(6.0)


def test_multiple_obstacle_collision_boxes_are_wide() -> None:
    obstacle = make_obstacle(world_module.CACTUS_SMALL, x=0, y=105, size=3)
    assert obstacle.width == world_module.CACTUS_SMALL.width * 3
    assert obstacle.collision_boxes[1].width > world_module.CACTUS_SMALL.collision_boxes[1].width
