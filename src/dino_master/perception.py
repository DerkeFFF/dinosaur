"""Общие признаки наблюдения и действия для агентов Dino."""

from __future__ import annotations

from typing import Any

FEATURE_COUNT = 24
ACTION_COUNT = 3
NONE_DISTANCE = 1000.0
TTC_CLIP = 150.0
DUCK_HOLD_STEPS = 5
JUMP_COOLDOWN_STEPS = 3
TERMINAL_BONUS = 1000.0
CURRICULUM_SPEEDS = (6.0, 8.0, 10.0, 13.0)

TYPE_CODES = {
    "none": 0,
    "CACTUS_SMALL": 1,
    "CACTUS_LARGE": 2,
    "PTERODACTYL_LOW": 3,
    "PTERODACTYL_MID": 4,
    "PTERODACTYL_HIGH": 5,
}

# Порядок 24 признаков, которые видит сеть:
#  0      TTC первого препятствия
#  1..6   one-hot типа первого (none + 5 типов)
#  7      высота первого
#  8      ширина первого
#  9..17  то же для второго препятствия
#  18     gap между первым и вторым
#  19     текущая скорость / 13
#  20     dino.jumping
#  21     dino.ducking
#  22     высота динозавра / 150
#  23     dino.speedDrop


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def type_code(item: dict[str, Any] | None) -> int:
    if not item or item.get("type") is None:
        return TYPE_CODES["none"]
    obstacle_type = str(item.get("type"))
    if obstacle_type == "CACTUS_SMALL":
        return TYPE_CODES["CACTUS_SMALL"]
    if obstacle_type == "CACTUS_LARGE":
        return TYPE_CODES["CACTUS_LARGE"]
    if obstacle_type == "PTERODACTYL":
        raw_height = item.get("yPos")
        height = float(raw_height) if isinstance(raw_height, (int, float)) else 0.0
        # Чем выше птица, тем больше code: low=3, mid=4, high=5.
        if height >= 90:
            return TYPE_CODES["PTERODACTYL_LOW"]
        if height >= 65:
            return TYPE_CODES["PTERODACTYL_MID"]
        return TYPE_CODES["PTERODACTYL_HIGH"]
    raise ValueError(f"неизвестный тип препятствия: {obstacle_type!r}")


def _has_obstacle(item: dict[str, Any] | None) -> bool:
    if not item or item.get("type") is None:
        return False
    distance = item.get("distance")
    return not isinstance(distance, (int, float)) or float(distance) < NONE_DISTANCE


def _one_hot(code: int) -> list[float]:
    row = [0.0] * len(TYPE_CODES)
    row[code] = 1.0
    return row


def _ttc_feature(item: dict[str, Any] | None) -> float:
    if not _has_obstacle(item):
        return _clip(NONE_DISTANCE / 6.0 / TTC_CLIP)
    assert item is not None
    distance = float(item.get("distance") or 0.0)
    speed = float(item.get("speed") or 0.0)
    if speed <= 0:
        return 0.0
    return _clip(distance / speed / TTC_CLIP)


def _y_feature(item: dict[str, Any] | None) -> float:
    if not item:
        return 0.0
    value = item.get("yPos")
    if not isinstance(value, (int, float)):
        return 0.0
    return _clip(float(value) / 150.0)


def _width_feature(item: dict[str, Any] | None) -> float:
    if not item:
        return 0.0
    value = item.get("width")
    if not isinstance(value, (int, float)):
        return 0.0
    return _clip(float(value) / 100.0)


def observation_features(obs: dict[str, Any]) -> list[float]:
    if obs.get("status") != "OK":
        raise ValueError(f"статус наблюдения должен быть OK, получен {obs.get('status')!r}")

    first = obs.get("first")
    second = obs.get("second")
    speed = float(obs.get("speed") or 0.0)
    dino = obs.get("dino") or {}

    features: list[float] = [_ttc_feature(first)]
    features.extend(_one_hot(type_code(first)))
    features.append(_y_feature(first))
    features.append(_width_feature(first))

    # Второе препятствие важно для птиц и групповых кактусов.
    features.append(_ttc_feature(second))
    features.extend(_one_hot(type_code(second)))
    features.append(_y_feature(second))
    features.append(_width_feature(second))

    # Если второе препятствие далеко/отсутствует, gap считается «свободным».
    if _has_obstacle(second):
        first_distance = float((first or {}).get("distance") or 0.0)
        second_distance = float((second or {}).get("distance") or 0.0)
        features.append(_clip((second_distance - first_distance) / 500.0))
    else:
        features.append(1.0)

    # Состояние самого динозавра: скорость, прыжок, присед, высота, speedDrop.
    features.append(_clip(speed / 13.0))
    features.append(1.0 if dino.get("jumping") else 0.0)
    features.append(1.0 if dino.get("ducking") else 0.0)
    dino_y = dino.get("yPos")
    features.append(
        _clip(float(dino_y) / 150.0) if isinstance(dino_y, (int, float)) else 0.0
    )
    features.append(1.0 if dino.get("speedDrop") else 0.0)

    if len(features) != FEATURE_COUNT:
        raise AssertionError(f"ожидалось {FEATURE_COUNT} признаков, получено {len(features)}")
    return features


def valid_actions(obs: dict[str, Any]) -> list[int]:
    """Маска: присед без препятствия и прыжок в воздухе запрещены."""
    # Без препятствия duck бессмыслен; в воздухе нельзя начинать новый прыжок.
    actions = [0, 1, 2] if _has_obstacle(obs.get("first")) else [0, 1]
    if (obs.get("dino") or {}).get("jumping"):
        actions = [action for action in actions if action != 1]
    return actions or [0]


def resolve_action(action: int, duck_remaining: int) -> tuple[int, int]:
    """Удерживать присед несколько кадров; прыжок отменяет удержание."""
    if action == 1:
        # Прыжок всегда сбрасывает удержание duck.
        return 1, 0
    if action == 2:
        # Начали присед — держим его DUCK_HOLD_STEPS кадров.
        return 2, DUCK_HOLD_STEPS - 1
    if duck_remaining > 0:
        # Сеть выбрала 0, но удержание ещё не истекло.
        return 2, duck_remaining - 1
    return 0, 0


def curriculum_speed(generation: int, stage_gens: int = 5) -> float | None:
    """Фиксированная скорость для поколения (с 1); ``None`` — естественный разгон."""
    if stage_gens < 1:
        raise ValueError(f"stage_gens должно быть положительным, получено {stage_gens}")
    # 1..5 → 6, 6..10 → 8, 11..15 → 10, 16..20 → 13, дальше natural.
    stage = max(0, (generation - 1) // stage_gens)
    if stage < len(CURRICULUM_SPEEDS):
        return CURRICULUM_SPEEDS[stage]
    return None


def episode_fitness(steps: int, clears: int, max_steps: int) -> float:
    # Выживание важнее всего; препятствия дают небольшой бонус,
    # а полный горизонт — большой терминальный бонус.
    return steps + 5 * clears + (TERMINAL_BONUS if steps >= max_steps else 0.0)
