"""Тесты признаков и действий."""

from __future__ import annotations

import pytest

from dino_master import perception


def obs(*, obstacle: bool = False, jumping: bool = False) -> dict:
    first = (
        {
            "id": 1,
            "distance": 100.0,
            "type": "CACTUS_SMALL",
            "yPos": 0.0,
            "width": 17.0,
            "speed": 6.0,
        }
        if obstacle
        else {
            "id": None,
            "distance": 1000.0,
            "type": None,
            "yPos": None,
            "width": 0.0,
            "speed": 6.0,
        }
    )
    return {
        "status": "OK",
        "first": first,
        "second": {
            "id": None,
            "distance": 1000.0,
            "type": None,
            "yPos": None,
            "width": 0.0,
            "speed": 6.0,
        },
        "speed": 6.0,
        "dino": {
            "jumping": jumping,
            "ducking": False,
            "yPos": 0.0,
            "speedDrop": False,
        },
    }


def test_valid_actions_mask() -> None:
    assert perception.valid_actions(obs()) == [0, 1]
    assert perception.valid_actions(obs(obstacle=True)) == [0, 1, 2]
    assert perception.valid_actions(obs(obstacle=True, jumping=True)) == [0, 2]


def test_resolve_action_holds_duck_and_jump_cancels() -> None:
    action, remaining = perception.resolve_action(2, 0)
    assert action == 2
    assert remaining == perception.DUCK_HOLD_STEPS - 1
    action, remaining = perception.resolve_action(0, remaining)
    assert action == 2
    assert perception.resolve_action(1, 3) == (1, 0)
    assert perception.resolve_action(0, 0) == (0, 0)


def test_curriculum_stages() -> None:
    assert perception.curriculum_speed(1, 5) == pytest.approx(6.0)
    assert perception.curriculum_speed(6, 5) == pytest.approx(8.0)
    assert perception.curriculum_speed(11, 5) == pytest.approx(10.0)
    assert perception.curriculum_speed(16, 5) == pytest.approx(13.0)
    assert perception.curriculum_speed(21, 5) is None


def test_features_length() -> None:
    assert len(perception.observation_features(obs())) == perception.FEATURE_COUNT == 24


def test_ttc_feature_has_useful_resolution_near_obstacles() -> None:
    near = obs(obstacle=True)
    near["first"]["distance"] = 60.0
    far = obs(obstacle=True)
    far["first"]["distance"] = 270.0
    near_ttc = perception.observation_features(near)[0]
    far_ttc = perception.observation_features(far)[0]
    assert near_ttc == pytest.approx(60.0 / 6.0 / 45.0)
    assert far_ttc == pytest.approx(1.0)
    assert far_ttc > near_ttc
