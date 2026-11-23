"""Окно pygame: один геном играет в чистом Python-симуляторе."""

from __future__ import annotations

import argparse
import contextlib
import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import neat

from . import features, sim

DEFAULT_GENOME = Path("models/champion-7553.pkl")
DEFAULT_CONFIG = Path(__file__).with_name("neat.ini")

BACKGROUND = (247, 247, 247)
INK = (83, 83, 83)
CACTUS = (76, 126, 78)
CRASH = (190, 55, 55)
GROUND = (165, 165, 165)
HUD_BG = (255, 255, 255)
HUD_LINE = (225, 225, 225)
HUD_HEIGHT = 78
GROUND_BOTTOM = 140
DUCK_HEIGHT = 25
RESTART_DELAY = 1.0
SCREENSHOT_FRAMES = 240


class Pilot:
    """Повторяет цикл принятия решений из ``train.run_sim_episode``."""

    def __init__(self, network: Any) -> None:
        self.network = network
        self.duck_remaining = 0
        self.jump_cooldown = 0

    def reset(self) -> None:
        self.duck_remaining = 0
        self.jump_cooldown = 0

    def act(self, obs: dict[str, Any]) -> int:
        values = features.observation_features(obs)
        valid = features.valid_actions(obs)
        if self.jump_cooldown > 0:
            valid = [action for action in valid if action != 1] or [0]
        outputs = list(self.network.activate(values))
        raw = max(valid, key=lambda action: float(outputs[action]))
        action, self.duck_remaining = features.resolve_action(
            raw, self.duck_remaining
        )
        self.jump_cooldown = (
            features.JUMP_COOLDOWN_STEPS
            if action == 1
            else max(0, self.jump_cooldown - 1)
        )
        return action


def step_world(world: sim.SimWorld, pilot: Pilot) -> bool:
    """Один кадр мира. ``False`` — динозавр погиб или наблюдение не OK."""

    if world.all_dead:
        return False
    obs = world.observation(0)
    if obs.get("status") != "OK":
        return False
    world.step([pilot.act(obs)])
    return not world.all_dead


def load_network(
    genome_path: str | Path, config_path: str | Path
) -> tuple[Any, Any]:
    """Загрузить доверенный локальный pickle и собрать FeedForwardNetwork."""

    path = Path(genome_path)
    if not path.is_file():
        raise SystemExit(f"геном не найден: {path}")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    with path.open("rb") as handle:
        genome = pickle.load(handle)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return genome, network


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dino-visualize",
        description="Показать один геном в симуляторе Chrome Dino",
    )
    parser.add_argument("--genome", default=str(DEFAULT_GENOME))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Фиксированная скорость; по умолчанию естественный разгон 6 → 13",
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Шагов на эпизод; 0 — без ограничения",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Кадров окна; 0 — без ограничения (для тестов)",
    )
    parser.add_argument(
        "--screenshot",
        default=None,
        help="Сохранить кадр в PNG и выйти (по умолчанию после 240 кадров)",
    )
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error("--scale должно быть положительным")
    if args.fps < 1:
        parser.error("--fps должно быть положительным")
    if args.speed is not None and args.speed <= 0:
        parser.error("--speed должно быть положительным")
    if args.max_steps < 0:
        parser.error("--max-steps не может быть отрицательным")
    if args.max_frames < 0:
        parser.error("--max-frames не может быть отрицательным")
    return args


def _font(pygame: Any, size: int) -> Any:
    for name in ("dejavusans", "liberationsans", "arial"):
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


class Renderer:
    def __init__(self, pygame: Any, screen: Any, scale: int) -> None:
        self.pygame = pygame
        self.screen = screen
        self.scale = scale
        self.title_font = _font(pygame, 20)
        self.small_font = _font(pygame, 16)

    def _px(self, value: float) -> int:
        return int(round(value * self.scale))

    def _py(self, value: float) -> int:
        return int(round(value * self.scale)) + HUD_HEIGHT

    def _rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: tuple[int, int, int],
        radius: int = 0,
    ) -> None:
        rect = self.pygame.Rect(
            self._px(x),
            self._py(y),
            max(1, self._px(width)),
            max(1, self._px(height)),
        )
        self.pygame.draw.rect(
            self.screen, color, rect, border_radius=max(0, radius * self.scale)
        )

    def _polygon(
        self, points: Sequence[tuple[float, float]], color: tuple[int, int, int]
    ) -> None:
        self.pygame.draw.polygon(
            self.screen,
            color,
            [(self._px(x), self._py(y)) for x, y in points],
        )

    def _ground(self, distance: float) -> None:
        pygame = self.pygame
        y = self._py(GROUND_BOTTOM)
        pygame.draw.line(
            self.screen,
            GROUND,
            (0, y),
            (self.screen.get_width(), y),
            max(1, self.scale),
        )
        dash = max(24, 48 * self.scale)
        offset = int(distance * self.scale) % dash
        for start in range(-offset, self.screen.get_width(), dash):
            end = min(start + 20 * self.scale, self.screen.get_width())
            pygame.draw.line(
                self.screen,
                GROUND,
                (start, y - 4 * self.scale),
                (end, y - 4 * self.scale),
                max(1, self.scale // 2),
            )

    def _dino(self, dino: sim.TRex, crashed: bool) -> None:
        color = CRASH if crashed else INK
        x, y = dino.x, dino.y
        if dino.ducking:
            top = y + sim.TREX_HEIGHT - DUCK_HEIGHT
            self._rect(x + 2, top, 51, 20, color, 3)
            self._rect(x + 42, top - 5, 15, 13, color, 3)
            self._rect(x + 51, top + 6, 7, 4, color)
            self._rect(x + 8, top + 19, 8, 6, color)
            self._rect(x + 27, top + 19, 8, 6, color)
            self._rect(x + 48, top - 2, 3, 3, BACKGROUND)
            return

        self._polygon(
            [(x, y + 24), (x + 4, y + 13), (x + 14, y + 22), (x + 14, y + 34)],
            color,
        )
        self._rect(x + 7, y + 13, 25, 27, color, 2)
        self._rect(x + 25, y, 19, 18, color, 3)
        self._rect(x + 25, y + 13, 15, 5, color)
        self._rect(x + 10, y + 37, 7, 10, color)
        self._rect(x + 24, y + 37, 7, 10, color)
        self._rect(x + 29, y + 29, 7, 4, color)
        self._rect(x + 38, y + 5, 3, 3, BACKGROUND)

    def _cactus(self, obstacle: sim.Obstacle) -> None:
        stem_width = obstacle.type.width
        height = obstacle.type.height
        for index in range(obstacle.size):
            base_x = obstacle.x + index * stem_width
            trunk_width = max(3, stem_width * 0.34)
            trunk_x = base_x + (stem_width - trunk_width) / 2
            self._rect(trunk_x, obstacle.y, trunk_width, height, CACTUS, 2)

            arm_width = max(3, stem_width * 0.24)
            arm_height = max(5, height * 0.22)
            connector = max(2, height * 0.07)

            left_x = base_x + stem_width * 0.02
            left_y = obstacle.y + height * 0.42
            self._rect(left_x, left_y, arm_width, arm_height, CACTUS, 2)
            self._rect(
                left_x + arm_width,
                left_y + arm_height - connector,
                trunk_x - (left_x + arm_width),
                connector,
                CACTUS,
            )

            right_x = base_x + stem_width * 0.98 - arm_width
            right_y = obstacle.y + height * 0.30
            self._rect(right_x, right_y, arm_width, arm_height, CACTUS, 2)
            self._rect(
                trunk_x + trunk_width,
                right_y + arm_height - connector,
                right_x - (trunk_x + trunk_width),
                connector,
                CACTUS,
            )

    def _bird(self, obstacle: sim.Obstacle) -> None:
        x, y = obstacle.x, obstacle.y
        self._rect(x + 10, y + 18, 22, 10, INK, 4)
        self._rect(x + 28, y + 12, 11, 10, INK, 3)
        self._polygon(
            [(x + 39, y + 16), (x + 46, y + 19), (x + 39, y + 22)], INK
        )
        if obstacle.current_frame == 0:
            wing = [(x + 20, y + 18), (x + 7, y + 7), (x + 28, y + 14)]
        else:
            wing = [(x + 20, y + 20), (x + 7, y + 33), (x + 28, y + 26)]
        self._polygon(wing, INK)
        self._rect(x + 34, y + 14, 2, 2, BACKGROUND)

    def _obstacle(self, obstacle: sim.Obstacle) -> None:
        if obstacle.type.name.startswith("CACTUS"):
            self._cactus(obstacle)
        else:
            self._bird(obstacle)

    def _hud(
        self,
        world: sim.SimWorld,
        *,
        genome_key: Any,
        fitness: float | None,
        episode: int,
        best_steps: int,
        paused: bool,
    ) -> None:
        pygame = self.pygame
        width = self.screen.get_width()
        pygame.draw.rect(self.screen, HUD_BG, (0, 0, width, HUD_HEIGHT))
        pygame.draw.line(
            self.screen, HUD_LINE, (0, HUD_HEIGHT - 1), (width, HUD_HEIGHT - 1)
        )

        fitness_text = "—" if fitness is None else f"{fitness:.0f}"
        first = f"Genome {genome_key} · fitness {fitness_text} · эпизод {episode}"
        second = (
            f"шаг {world.steps[0]} · препятствий {world.clears[0]} · "
            f"скорость {world.current_speed:.1f} · "
            f"рекорд {max(best_steps, world.steps[0])}"
        )
        hint = "Esc/Q — выход · R — заново · Space — пауза"
        title_surface = self.title_font.render(first, True, INK)
        hint_surface = self.small_font.render(hint, True, (130, 130, 130))
        self.screen.blit(title_surface, (14, 8))
        self.screen.blit(self.small_font.render(second, True, INK), (14, 34))
        hint_x = width - hint_surface.get_width() - 14
        hint_y = 8
        if hint_x < title_surface.get_width() + 28:
            hint_x = 14
            hint_y = 56
        self.screen.blit(hint_surface, (hint_x, hint_y))
        if paused:
            pause = self.title_font.render("Пауза", True, CRASH)
            self.screen.blit(
                pause,
                (width - pause.get_width() - 14, 38),
            )

    def _message(self, text: str, steps: int) -> None:
        pygame = self.pygame
        width = self.screen.get_width()
        height = self.screen.get_height() - HUD_HEIGHT
        veil = pygame.Surface((width, height), pygame.SRCALPHA)
        veil.fill((255, 255, 255, 190))
        self.screen.blit(veil, (0, HUD_HEIGHT))

        color = CRASH if text == "Крах" else INK
        title = self.title_font.render(f"{text} · шаг {steps}", True, color)
        self.screen.blit(
            title,
            title.get_rect(center=(width // 2, HUD_HEIGHT + height // 2 - 10)),
        )
        subtitle = self.small_font.render("следующий эпизод...", True, INK)
        self.screen.blit(
            subtitle,
            subtitle.get_rect(center=(width // 2, HUD_HEIGHT + height // 2 + 20)),
        )

    def draw(
        self,
        world: sim.SimWorld,
        *,
        genome_key: Any,
        fitness: float | None,
        episode: int,
        best_steps: int,
        paused: bool,
        message: str,
    ) -> None:
        self.screen.fill(BACKGROUND)
        self._ground(world.distance)
        for obstacle in world.obstacles:
            self._obstacle(obstacle)
        self._dino(world.dinos[0], crashed=world.all_dead)
        self._hud(
            world,
            genome_key=genome_key,
            fitness=fitness,
            episode=episode,
            best_steps=best_steps,
            paused=paused,
        )
        if message:
            self._message(message, world.steps[0])


def run(args: argparse.Namespace) -> None:
    try:
        import pygame
    except ImportError as error:
        raise SystemExit(
            "Для dino-visualize нужен pygame.\n"
            "  uv sync --extra viz\n"
            "  uv run dino-visualize"
        ) from error

    genome, network = load_network(args.genome, args.config)
    world = sim.SimWorld(
        sim.SimConfig(
            dino_count=1,
            max_steps=args.max_steps or 1_000_000,
            speed=args.speed,
            seed=args.seed,
        )
    )
    pilot = Pilot(network)

    pygame.init()
    width = sim.WIDTH * args.scale
    height = sim.HEIGHT * args.scale + HUD_HEIGHT
    try:
        screen = pygame.display.set_mode((width, height))
    except pygame.error as error:
        pygame.quit()
        raise SystemExit(
            f"Не удалось открыть окно: {error}\n"
            "Нужно графическое окружение и доступный DISPLAY."
        ) from error

    pygame.display.set_caption("Dino NEAT — визуализация")
    clock = pygame.time.Clock()
    renderer = Renderer(pygame, screen, args.scale)

    raw_fitness = getattr(genome, "fitness", None)
    fitness = float(raw_fitness) if isinstance(raw_fitness, (int, float)) else None
    genome_key = getattr(genome, "key", "?")

    episode = 1
    best_steps = 0
    paused = False
    message = ""
    restart_at: float | None = None
    frames = 0
    max_frames = args.max_frames
    if args.screenshot and max_frames == 0:
        max_frames = SCREENSHOT_FRAMES

    print(
        f"Геном {genome_key} · {args.genome}\n"
        f"Окно {width}×{height}; Esc/Q — выход, R — заново, Space — пауза"
    )

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if (
                        event.key == pygame.K_ESCAPE
                        or event.key == pygame.K_q
                        or event.scancode == pygame.KSCAN_Q
                    ):
                        running = False
                    elif event.key == pygame.K_r or event.scancode == pygame.KSCAN_R:
                        best_steps = max(best_steps, world.steps[0])
                        episode += 1
                        pilot.reset()
                        world.reset(args.seed + episode)
                        restart_at = None
                        message = ""
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            now = time.monotonic()
            finished = world.all_dead or (
                args.max_steps > 0 and world.steps[0] >= args.max_steps
            )
            if not paused:
                if finished:
                    if restart_at is None:
                        best_steps = max(best_steps, world.steps[0])
                        restart_at = now + RESTART_DELAY
                        message = "Крах" if world.all_dead else "Горизонт достигнут"
                    elif now >= restart_at:
                        episode += 1
                        pilot.reset()
                        world.reset(args.seed + episode)
                        restart_at = None
                        message = ""
                else:
                    step_world(world, pilot)

            renderer.draw(
                world,
                genome_key=genome_key,
                fitness=fitness,
                episode=episode,
                best_steps=best_steps,
                paused=paused,
                message=message,
            )
            pygame.display.flip()
            frames += 1
            if max_frames and frames >= max_frames:
                running = False
            clock.tick(args.fps)

        if args.screenshot:
            path = Path(args.screenshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(path))
            print(f"кадр сохранён: {path}")
    finally:
        pygame.quit()


def cli() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        run(parse_args())


if __name__ == "__main__":
    cli()
