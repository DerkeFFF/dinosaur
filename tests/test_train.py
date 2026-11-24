"""Unit-тесты обучения NEAT без браузера."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import neat
import pytest

from dino_master import perception, train

CONFIG_PATH = Path(__file__).resolve().parents[1] / "src" / "dino_master" / "neat.ini"


def make_config() -> neat.Config:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )
    config.genome_config.innovation_tracker = neat.InnovationTracker()
    return config


def make_genomes(config: neat.Config, count: int = 2) -> list[tuple[int, neat.DefaultGenome]]:
    genomes = []
    for key in range(1, count + 1):
        genome = neat.DefaultGenome(key)
        genome.configure_new(config.genome_config)
        genomes.append((key, genome))
    return genomes


def test_parse_args_defaults() -> None:
    args = train.parse_args([])
    assert args.generations == 50
    assert args.population == 500
    assert args.max_steps == 30000
    assert args.episodes_per_gen == 4
    assert args.curriculum_gens == 5
    assert args.seed == 1


def test_parse_args_rejects_non_positive_population() -> None:
    with pytest.raises(SystemExit):
        train.parse_args(["--population", "0"])


def test_choose_action_picks_best_valid() -> None:
    assert train.choose_action([0.1, 0.9, 0.2], [0, 1]) == 1
    assert train.choose_action([0.1, 0.2, 2.0], [0, 1]) == 1
    with pytest.raises(ValueError, match="valid_actions"):
        train.choose_action([1.0, 2.0, 3.0], [])


def test_run_sim_episode_short_horizon() -> None:
    config = make_config()
    genomes = make_genomes(config, 2)
    fitnesses, stats = train.run_sim_episode(
        genomes, config, max_steps=5, speed=6.0, seed=1
    )
    assert len(fitnesses) == 2
    assert stats == [(5, 0), (5, 0)]
    assert all(value >= 5.0 for value in fitnesses)


def test_evaluate_genomes_averages_episodes() -> None:
    config = make_config()
    genomes = make_genomes(config, 2)
    train.EPISODE_STATS.clear()
    train.LAST_STATS.clear()
    train.evaluate_genomes(
        genomes, config, max_steps=5, episodes=2, speed=6.0, seed=7
    )
    for key, genome in genomes:
        assert genome.fitness is not None
        assert genome.fitness > 0.0
        assert train.EPISODE_STATS[key] == (5, 0)
    assert train.LAST_STATS["steps"] == 10.0
    assert train.LAST_STATS["best"] == pytest.approx(
        max(genome.fitness for _key, genome in genomes if genome.fitness is not None)
    )


def test_reporter_persists_only_improvements(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = make_config()
    species = SimpleNamespace(species=[])
    best_path = tmp_path / "best.pkl"
    reporter = train.BestGenomeReporter(best_path)
    genome = make_genomes(config, 1)[0][1]
    genome.fitness = 42.0
    train.LAST_STATS.clear()
    train.LAST_STATS.update(
        {"best": 42.0, "median": 21.0, "mean": 21.0, "steps": 5.0, "clears": 0.0}
    )

    reporter.post_evaluate(config, None, species, genome)
    assert best_path.is_file()
    assert "Поколение 1" in capsys.readouterr().out
    with best_path.open("rb") as handle:
        assert pickle.load(handle).fitness == pytest.approx(42.0)

    worse = make_genomes(config, 1)[0][1]
    worse.fitness = 10.0
    reporter.post_evaluate(config, None, species, worse)
    with best_path.open("rb") as handle:
        assert pickle.load(handle).fitness == pytest.approx(42.0)


def test_features_shape() -> None:
    obs = {
        "status": "OK",
        "first": {
            "id": 1,
            "distance": 100.0,
            "type": None,
            "yPos": 0.0,
            "width": 0.0,
            "speed": 6.0,
        },
        "second": {
            "id": None,
            "distance": 1000.0,
            "type": None,
            "yPos": None,
            "width": 0.0,
            "speed": 6.0,
        },
        "speed": 6.0,
        "dino": {"jumping": False, "ducking": False, "yPos": 93.0, "speedDrop": False},
    }
    assert len(perception.observation_features(obs)) == perception.FEATURE_COUNT == 24
