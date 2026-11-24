"""Обучение NEAT в симуляторе Dino без браузера."""

from __future__ import annotations

import argparse
import pickle
import statistics
import time
from pathlib import Path
from typing import Any

import neat

from . import features, sim

FEATURE_COUNT = features.FEATURE_COUNT
DEFAULT_CONFIG = Path(__file__).with_name("neat.ini")

EPISODE_STATS: dict[int, tuple[int, int]] = {}
LAST_STATS: dict[str, float] = {}


def choose_action(outputs: list[float], valid: list[int]) -> int:
    if not valid:
        raise ValueError("valid_actions не должен быть пустым")
    return max(valid, key=lambda action: float(outputs[action]))


def run_sim_episode(
    genomes: list[tuple[int, Any]],
    config: Any,
    *,
    max_steps: int,
    speed: float | None,
    seed: int,
) -> tuple[list[float], list[tuple[int, int]]]:
    count = len(genomes)
    if not count:
        return [], []
    world = sim.SimWorld(
        sim.SimConfig(dino_count=count, max_steps=max_steps, speed=speed, seed=seed)
    )
    networks = [
        neat.nn.FeedForwardNetwork.create(genome, config)
        for _key, genome in genomes
    ]
    duck_remaining = [0] * count
    jump_cooldown = [0] * count

    # Один кадр = одно решение для каждого живого генома.
    for _step in range(max_steps):
        actions = [0] * count
        for index in range(count):
            if not world.alive[index]:
                continue
            obs = world.observation(index)
            if obs.get("status") != "OK":
                continue
            values = features.observation_features(obs)
            valid = features.valid_actions(obs)
            if jump_cooldown[index] > 0:
                valid = [action for action in valid if action != 1] or [0]
            raw = choose_action(list(networks[index].activate(values)), valid)
            action, duck_remaining[index] = features.resolve_action(
                raw, duck_remaining[index]
            )
            jump_cooldown[index] = (
                features.JUMP_COOLDOWN_STEPS
                if action == 1
                else max(0, jump_cooldown[index] - 1)
            )
            actions[index] = action
        world.step(actions)
        if world.all_dead:
            break

    # Fitness: выживание + бонус за препятствия + бонус за полный горизонт.
    fitness = [
        features.episode_fitness(world.steps[i], world.clears[i], max_steps)
        for i in range(count)
    ]
    stats = [(world.steps[i], world.clears[i]) for i in range(count)]
    return fitness, stats


def _record_stats(
    genomes: list[tuple[int, Any]],
    steps: list[int],
    clears: list[int],
) -> None:
    fitnesses = [float(genome.fitness or 0.0) for _key, genome in genomes]
    LAST_STATS.clear()
    LAST_STATS.update(
        {
            "best": max(fitnesses) if fitnesses else 0.0,
            "median": float(statistics.median(fitnesses)) if fitnesses else 0.0,
            "mean": float(statistics.fmean(fitnesses)) if fitnesses else 0.0,
            "steps": float(sum(steps)),
            "clears": float(sum(clears)),
        }
    )


def evaluate_genomes(
    genomes: list[tuple[int, Any]],
    config: Any,
    *,
    max_steps: int,
    episodes: int,
    speed: float | None,
    seed: int,
) -> None:
    if episodes < 1:
        raise ValueError(f"episodes должно быть положительным, получено {episodes}")
    if not genomes:
        return
    count = len(genomes)
    fitness_total = [0.0] * count
    steps_total = [0] * count
    clears_total = [0] * count

    # Несколько разных seed снижают удачу на одном потоке препятствий.
    for episode in range(1, episodes + 1):
        fitnesses, stats = run_sim_episode(
            genomes, config, max_steps=max_steps, speed=speed, seed=seed + episode
        )
        for index in range(count):
            fitness_total[index] += fitnesses[index]
            steps_total[index] += stats[index][0]
            clears_total[index] += stats[index][1]

    steps = [round(value / episodes) for value in steps_total]
    clears = [round(value / episodes) for value in clears_total]
    for index, (_key, genome) in enumerate(genomes):
        genome.fitness = fitness_total[index] / episodes
        EPISODE_STATS[genome.key] = (steps[index], clears[index])
    _record_stats(genomes, steps, clears)


class BestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.best_fitness: float | None = None
        self.generation = 0

    def post_evaluate(
        self, config: Any, population: Any, species: Any, best_genome: Any
    ) -> None:
        del config, population, species
        self.generation += 1
        if best_genome is not None and best_genome.fitness is not None:
            fitness = float(best_genome.fitness)
            if self.best_fitness is None or fitness > self.best_fitness:
                self.best_fitness = fitness
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("wb") as handle:
                    pickle.dump(best_genome, handle)
        stats = LAST_STATS
        print(
            f"Поколение {self.generation}: best={stats.get('best', 0):.0f} "
            f"median={stats.get('median', 0):.0f} "
            f"mean={stats.get('mean', 0):.0f} "
            f"steps={stats.get('steps', 0):.0f} "
            f"clears={stats.get('clears', 0):.0f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dino-train",
        description="Обучение NEAT в симуляторе Chrome Dino",
    )
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--episodes-per-gen", type=int, default=4)
    parser.add_argument("--curriculum-gens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--best", default="models/best.pkl")
    args = parser.parse_args(argv)
    for name in ("generations", "population", "max_steps", "episodes_per_gen", "curriculum_gens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} должно быть положительным")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )
    config.pop_size = args.population
    population = neat.Population(config, seed=args.seed)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(BestGenomeReporter(args.best))

    def eval_genomes(genomes: list[tuple[int, Any]], cfg: Any) -> None:
        # population.generation ещё не увеличен внутри NEAT — передаём 1-based.
        generation = population.generation + 1
        speed = features.curriculum_speed(generation, args.curriculum_gens)
        evaluate_genomes(
            genomes,
            cfg,
            max_steps=args.max_steps,
            episodes=args.episodes_per_gen,
            speed=speed,
            seed=args.seed,
        )

    started = time.monotonic()
    best = population.run(eval_genomes, args.generations)
    elapsed = time.monotonic() - started
    target = Path(args.best)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(best, handle)
    print(
        f"готово: всего={elapsed:.1f}с "
        f"среднее_поколение={elapsed / args.generations:.2f}с"
    )


if __name__ == "__main__":
    main()
