"""Чистый Python-порт физики Chrome Dino, спавна и коллизий.

Модуль повторяет ``vendor/t-rex-runner/index.js`` достаточно точно, чтобы
обучать и проверять агентов без браузера. Без Playwright, NumPy и torch;
вся случайность идёт через ``random.Random``, поэтому запуск детерминирован
при заданном seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Final

FPS: Final[int] = 60
DT_MS: Final[float] = 1000.0 / FPS
# Координаты игры: (0,0) — левый верхний угол canvas 600×150.
GROUND_Y: Final[int] = 93
TREX_X: Final[int] = 50

SPEED: Final[float] = 6.0
ACCELERATION: Final[float] = 0.001
MAX_SPEED: Final[float] = 13.0

GRAVITY: Final[float] = 0.6
JUMP_V: Final[float] = -10.0
DROP_V: Final[float] = -5.0
MIN_JUMP_HEIGHT: Final[float] = 30.0
MAX_JUMP_HEIGHT: Final[float] = 30.0
SPEED_DROP_COEFFICIENT: Final[float] = 3.0

TREX_WIDTH: Final[int] = 44
TREX_HEIGHT: Final[int] = 47
TREX_WIDTH_DUCK: Final[int] = 59

CLEAR_TIME_MS: Final[int] = 3000
WIDTH: Final[int] = 600
HEIGHT: Final[int] = 150
MAX_GAP_COEFFICIENT: Final[float] = 1.5
MAX_OBSTACLE_DUPLICATION: Final[int] = 2
MAX_OBSTACLE_LENGTH: Final[int] = 3
GAP_COEFFICIENT: Final[float] = 0.6


def js_round(value: float) -> int:
    """Аналог JavaScript ``Math.round`` для конечных чисел."""

    return math.floor(value + 0.5)


@dataclass
class CollisionBox:
    """Выровненный по осям collision box."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class ObstacleType:
    """Описание препятствия, скопированное из исходника игры."""

    name: str
    width: int
    height: int
    y_pos: int | tuple[int, ...]
    min_gap: int
    min_speed: float
    multiple_speed: float
    collision_boxes: tuple[CollisionBox, ...]
    speed_offset: float = 0.0
    num_frames: int = 0
    frame_rate: float = 0.0


CACTUS_SMALL: Final[ObstacleType] = ObstacleType(
    name="CACTUS_SMALL",
    width=17,
    height=35,
    y_pos=105,
    min_gap=120,
    min_speed=0.0,
    multiple_speed=4.0,
    collision_boxes=(
        CollisionBox(0, 7, 5, 27),
        CollisionBox(4, 0, 6, 34),
        CollisionBox(10, 4, 7, 14),
    ),
)

CACTUS_LARGE: Final[ObstacleType] = ObstacleType(
    name="CACTUS_LARGE",
    width=25,
    height=50,
    y_pos=90,
    min_gap=120,
    min_speed=0.0,
    multiple_speed=7.0,
    collision_boxes=(
        CollisionBox(0, 12, 7, 38),
        CollisionBox(8, 0, 7, 49),
        CollisionBox(13, 10, 10, 38),
    ),
)

PTERODACTYL: Final[ObstacleType] = ObstacleType(
    name="PTERODACTYL",
    width=46,
    height=40,
    y_pos=(100, 75, 50),
    min_gap=150,
    min_speed=8.5,
    multiple_speed=999.0,
    collision_boxes=(
        CollisionBox(15, 15, 16, 5),
        CollisionBox(18, 21, 24, 6),
        CollisionBox(2, 14, 4, 3),
        CollisionBox(6, 10, 4, 7),
        CollisionBox(10, 8, 6, 9),
    ),
    speed_offset=0.8,
    num_frames=2,
    frame_rate=1000.0 / 6.0,
)

OBSTACLE_TYPES: Final[tuple[ObstacleType, ...]] = (
    CACTUS_SMALL,
    CACTUS_LARGE,
    PTERODACTYL,
)


_RUNNING_BOXES: Final[tuple[CollisionBox, ...]] = (
    CollisionBox(22, 0, 17, 16),
    CollisionBox(1, 18, 30, 9),
    CollisionBox(10, 35, 14, 8),
    CollisionBox(1, 24, 29, 5),
    CollisionBox(5, 30, 21, 4),
    CollisionBox(9, 34, 15, 4),
)
_DUCKING_BOXES: Final[tuple[CollisionBox, ...]] = (
    CollisionBox(1, 18, 55, 25),
)


@dataclass
class TRex:
    """Состояние динозавра и физика прыжка."""

    x: int = TREX_X
    y: float = GROUND_Y
    vy: float = 0.0
    jumping: bool = False
    ducking: bool = False
    speed_drop: bool = False
    reached_min_height: bool = False
    jump_count: int = 0
    ground: float = GROUND_Y

    def start_jump(self, speed: float) -> None:
        if not self.jumping:
            self.vy = JUMP_V - speed / 10.0
            self.jumping = True
            self.reached_min_height = False
            self.speed_drop = False

    def set_duck(self, ducking: bool) -> None:
        self.ducking = ducking

    def set_speed_drop(self) -> None:
        self.speed_drop = True
        self.vy = 1.0

    def reset(self) -> None:
        self.x = TREX_X
        self.y = self.ground
        self.vy = 0.0
        self.jumping = False
        self.ducking = False
        self.speed_drop = False
        self.reached_min_height = False
        self.jump_count = 0

    def update_frame(self) -> None:
        """Один кадр ``updateJump`` (``framesElapsed`` == 1)."""

        # Как в оригинале: при speedDrop падение в 3 раза быстрее,
        # а высота округляется как Math.round в JS.
        if self.speed_drop:
            self.y += js_round(self.vy * SPEED_DROP_COEFFICIENT)
        else:
            self.y += js_round(self.vy)
        self.vy += GRAVITY

        # reachedMinHeight и endJump ограничивают скорость наверху дуги.
        if self.y < self.ground - MIN_JUMP_HEIGHT or self.speed_drop:
            self.reached_min_height = True

        if (
            (self.y < MAX_JUMP_HEIGHT or self.speed_drop)
            and self.reached_min_height
            and self.vy < DROP_V
        ):
            self.vy = DROP_V

        # Приземление: сброс всех флагов и счётчик прыжка.
        if self.y > self.ground:
            was_jumping = self.jumping
            self.y = self.ground
            self.jumping = False
            self.vy = 0.0
            self.ducking = False
            self.speed_drop = False
            self.reached_min_height = False
            if was_jumping:
                self.jump_count = 0
                self.jump_count += 1

    def apply_action(self, action: int, speed: float) -> None:
        # 1 — прыжок (из duck сначала отпускаем присед),
        # 2 — присед на земле или ускоренное падение в воздухе,
        # 0 — отпустить duck/speedDrop.
        if action == 1:
            if self.ducking:
                self.set_duck(False)
            self.start_jump(speed)
        elif action == 2:
            if self.jumping:
                if not self.speed_drop:
                    self.set_speed_drop()
            else:
                self.set_duck(True)
        elif action == 0:
            if self.ducking:
                self.set_duck(False)
            if self.speed_drop:
                self.speed_drop = False
        else:
            raise ValueError(f"неизвестное действие: {action!r}")

    def t_rex_box(self) -> CollisionBox:
        return CollisionBox(
            x=self.x + 1,
            y=int(self.y) + 1,
            width=TREX_WIDTH - 2,
            height=TREX_HEIGHT - 2,
        )

    def ducking_box(self) -> tuple[CollisionBox, ...]:
        return _DUCKING_BOXES if self.ducking else _RUNNING_BOXES


class Obstacle:
    """Одно препятствие: динамические collision box и зазор."""

    def __init__(
        self,
        type_config: ObstacleType,
        rng: random.Random,
        speed: float,
        obstacle_id: int,
    ) -> None:
        self.type = type_config
        self._rng = rng
        self.x = WIDTH + type_config.width
        self.size = rng.randint(1, MAX_OBSTACLE_LENGTH)
        if self.size > 1 and type_config.multiple_speed > speed:
            self.size = 1
        self.width = type_config.width * self.size

        if isinstance(type_config.y_pos, tuple):
            self.y = rng.choice(type_config.y_pos)
        else:
            self.y = type_config.y_pos

        self.collision_boxes = [
            CollisionBox(box.x, box.y, box.width, box.height)
            for box in type_config.collision_boxes
        ]
        if self.size > 1:
            self.collision_boxes[1].width = (
                self.width
                - self.collision_boxes[0].width
                - self.collision_boxes[2].width
            )
            self.collision_boxes[2].x = self.width - self.collision_boxes[2].width

        self.speed_offset = 0.0
        if type_config.speed_offset:
            self.speed_offset = (
                type_config.speed_offset if rng.random() > 0.5 else -type_config.speed_offset
            )

        self.gap = self.get_gap(speed, rng)
        self.current_frame = 0
        self.timer = 0.0
        self.remove = False
        self.id = obstacle_id
        self.following = False

    def update_frame(self, speed: float) -> None:
        # Птицы летят быстрее/медленнее горизонта на speedOffset.
        move_speed = speed + self.speed_offset if self.type.speed_offset else speed
        self.x -= math.floor(move_speed)

        # Анимация птицы (2 кадра); для кактусов num_frames == 0.
        if self.type.num_frames:
            self.timer += DT_MS
            if self.timer >= self.type.frame_rate:
                self.current_frame = (self.current_frame + 1) % self.type.num_frames
                self.timer = 0.0

        # Как только препятствие ушло за левый край — удаляем.
        if self.x + self.width <= 0:
            self.remove = True

    def get_gap(self, speed: float, rng: random.Random | None = None) -> int:
        source = self._rng if rng is None else rng
        min_gap = js_round(self.width * speed + self.type.min_gap * GAP_COEFFICIENT)
        max_gap = js_round(min_gap * MAX_GAP_COEFFICIENT)
        return source.randint(min_gap, max_gap)


@dataclass
class SimConfig:
    """Конфигурация мира."""

    dino_count: int
    max_steps: int
    speed: float | None = None
    clear_time_ms: int = CLEAR_TIME_MS
    seed: int = 0


def box_compare(left: CollisionBox, right: CollisionBox) -> bool:
    """Строгое пересечение AABB."""

    return (
        left.x < right.x + right.width
        and left.x + left.width > right.x
        and left.y < right.y + right.height
        and left.height + left.y > right.y
    )


def _adjusted_box(box: CollisionBox, adjustment: CollisionBox) -> CollisionBox:
    return CollisionBox(
        x=box.x + adjustment.x,
        y=box.y + adjustment.y,
        width=box.width,
        height=box.height,
    )


def check_collision(obstacle: Obstacle, t_rex: TRex) -> bool:
    """Аналог ``checkForCollision``: внешний и детальные боксы."""

    obstacle_box = CollisionBox(
        x=obstacle.x + 1,
        y=obstacle.y + 1,
        width=obstacle.type.width * obstacle.size - 2,
        height=obstacle.type.height - 2,
    )
    t_rex_box = t_rex.t_rex_box()

    if not box_compare(t_rex_box, obstacle_box):
        return False

    for t_rex_collision_box in t_rex.ducking_box():
        adjusted_t_rex = _adjusted_box(t_rex_collision_box, t_rex_box)
        for obstacle_collision_box in obstacle.collision_boxes:
            adjusted_obstacle = _adjusted_box(obstacle_collision_box, obstacle_box)
            if box_compare(adjusted_t_rex, adjusted_obstacle):
                return True
    return False


class SimWorld:
    """Заменитель ``Runner`` и ``Horizon`` без браузера."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.dinos: list[TRex] = []
        self.obstacles: list[Obstacle] = []
        self.obstacle_history: list[str] = []
        self.running_time_ms = 0.0
        self.current_speed = SPEED
        self.acceleration = ACCELERATION
        self.steps: list[int] = []
        self.clears: list[int] = []
        self.alive: list[bool] = []
        self.distance = 0.0
        self.last_first_id: list[int | None] = []
        self._next_obstacle_id = 0
        self.reset(config.seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(self.config.seed if seed is None else seed)
        self.dinos = [TRex() for _ in range(self.config.dino_count)]
        self.obstacles = []
        self.obstacle_history = []
        self.running_time_ms = 0.0
        if self.config.speed is None:
            self.current_speed = SPEED
            self.acceleration = ACCELERATION
        else:
            self.current_speed = self.config.speed
            self.acceleration = 0.0
        self.steps = [0] * self.config.dino_count
        self.clears = [0] * self.config.dino_count
        self.alive = [True] * self.config.dino_count
        self.distance = 0.0
        self.last_first_id = [None] * self.config.dino_count
        self._next_obstacle_id = 0

    @property
    def alive_count(self) -> int:
        return sum(self.alive)

    @property
    def all_dead(self) -> bool:
        return not any(self.alive)

    def step(self, actions: list[int]) -> None:
        if len(actions) != self.config.dino_count:
            raise ValueError(
                f"ожидалось {self.config.dino_count} действий, получено {len(actions)}"
            )
        if self.all_dead:
            return

        # 1. Действия и физика каждого живого динозавра.
        for index, dino in enumerate(self.dinos):
            if not self.alive[index]:
                continue
            dino.apply_action(actions[index], self.current_speed)
            if dino.jumping:
                dino.update_frame()
            self.steps[index] += 1

        # 2. Горизонт: задержка старта, движение и спавн препятствий.
        self.running_time_ms += DT_MS
        has_obstacles = self.running_time_ms > self.config.clear_time_ms

        if has_obstacles:
            self._update_obstacles()

        # 3. Коллизия только с первым препятствием, как в оригинале.
        if has_obstacles and self.obstacles:
            first_obstacle = self.obstacles[0]
            for index, dino in enumerate(self.dinos):
                if not self.alive[index]:
                    continue
                if check_collision(first_obstacle, dino):
                    self.alive[index] = False

        # 4. Пройденные препятствия: id исчез из first/second → +1 clear.
        ahead = self._ahead_obstacles()
        first_id = ahead[0].id if ahead else None
        second_id = ahead[1].id if len(ahead) > 1 else None
        for index in range(self.config.dino_count):
            if not self.alive[index]:
                continue
            previous_id = self.last_first_id[index]
            if previous_id is not None and previous_id not in {first_id, second_id}:
                self.clears[index] += 1
            self.last_first_id[index] = first_id

        # 5. Дистанция и естественный разгон (при fixed speed acceleration=0).
        if self.alive_count:
            self.distance += self.current_speed
            if self.acceleration and self.current_speed < MAX_SPEED:
                self.current_speed += self.acceleration

    def observation(self, index: int) -> dict[str, Any]:
        dino = self.dinos[index]
        empty = self._empty_obstacle()
        dino_observation: dict[str, Any] = {
            "jumping": dino.jumping,
            "ducking": dino.ducking,
            "yPos": dino.y,
            "speedDrop": dino.speed_drop,
        }
        if not self.alive[index]:
            return {
                "status": "DEAD",
                "first": empty,
                "second": dict(empty),
                "speed": self.current_speed,
                "dino": dino_observation,
            }

        ahead = self._ahead_obstacles()
        return {
            "status": "OK",
            "first": self._describe_obstacle(ahead[0]) if ahead else empty,
            "second": self._describe_obstacle(ahead[1]) if len(ahead) > 1 else dict(empty),
            "speed": self.current_speed,
            "dino": dino_observation,
        }

    def _empty_obstacle(self) -> dict[str, Any]:
        return {
            "id": None,
            "distance": 1000,
            "type": None,
            "yPos": None,
            "width": 0,
            "speed": self.current_speed,
        }

    def _ahead_obstacles(self) -> list[Obstacle]:
        ahead = [
            obstacle
            for obstacle in self.obstacles
            if obstacle.x + obstacle.width > TREX_X
        ]
        ahead.sort(key=lambda obstacle: obstacle.x)
        return ahead

    def _describe_obstacle(self, obstacle: Obstacle) -> dict[str, Any]:
        return {
            "id": obstacle.id,
            "distance": obstacle.x - TREX_X,
            "type": obstacle.type.name,
            "yPos": obstacle.y,
            "width": obstacle.width,
            "speed": self.current_speed + obstacle.speed_offset,
        }

    def _update_obstacles(self) -> None:
        for obstacle in self.obstacles:
            obstacle.update_frame(self.current_speed)
        self.obstacles = [obstacle for obstacle in self.obstacles if not obstacle.remove]

        # Новое препятствие появляется, когда последнее достаточно уехало влево
        # с учётом его gap. Если препятствий нет — создаём сразу.
        if self.obstacles:
            last_obstacle = self.obstacles[-1]
            if (
                not last_obstacle.following
                and last_obstacle.x + last_obstacle.width > 0
                and last_obstacle.x + last_obstacle.width + last_obstacle.gap < WIDTH
            ):
                self._add_new_obstacle()
                last_obstacle.following = True
        else:
            self._add_new_obstacle()

    def _add_new_obstacle(self) -> None:
        # Повторяем выбор, пока тип недоступен по скорости
        # или пока не нарушено правило двух одинаковых подряд.
        while True:
            type_config = self.rng.choice(OBSTACLE_TYPES)
            if (
                self._duplicate_obstacle_check(type_config.name)
                or self.current_speed < type_config.min_speed
            ):
                continue
            break

        obstacle = Obstacle(
            type_config,
            self.rng,
            self.current_speed,
            self._next_obstacle_id,
        )
        self._next_obstacle_id += 1
        self.obstacles.append(obstacle)
        self.obstacle_history.insert(0, type_config.name)
        del self.obstacle_history[MAX_OBSTACLE_DUPLICATION:]

    def _duplicate_obstacle_check(self, next_obstacle_type: str) -> bool:
        duplicate_count = 0
        for obstacle_type in self.obstacle_history:
            duplicate_count = (
                duplicate_count + 1 if obstacle_type == next_obstacle_type else 0
            )
        return duplicate_count >= MAX_OBSTACLE_DUPLICATION
