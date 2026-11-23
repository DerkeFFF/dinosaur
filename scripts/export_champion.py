"""Export the champion NEAT genome to JSON for the static GitHub Pages demo."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import neat

DEFAULT_GENOME = "models/champion-7553.pkl"
DEFAULT_CONFIG = "src/dino_master/neat.ini"
DEFAULT_OUT = "docs/champion.json"


def _activation(name: str) -> str:
    if name in {"tanh", "relu", "sigmoid"}:
        return name
    raise ValueError(f"unsupported activation: {name!r}")


def export(genome: Any, config: neat.Config) -> dict[str, Any]:
    # В JSON кладём «сырые» веса из генома. Масштабирование активаций
    # (tanh×2.5, sigmoid×5) выполняет уже JS/Python при вычислении.
    inputs = list(config.genome_config.input_keys)
    outputs = list(config.genome_config.output_keys)
    weights: list[list[float]] = []
    biases: list[float] = []
    activations: list[str] = []
    for output in outputs:
        row = []
        for input_key in inputs:
            connection = genome.connections.get((input_key, output))
            row.append(float(connection.weight) if connection and connection.enabled else 0.0)
        weights.append(row)
        node = genome.nodes[output]
        biases.append(float(node.bias))
        activations.append(_activation(node.activation))
    return {
        "key": genome.key,
        "fitness": genome.fitness,
        "inputs": len(inputs),
        "outputs": len(outputs),
        "weights": weights,
        "biases": biases,
        "activations": activations,
    }


def verify(payload: dict[str, Any], genome: Any, config: neat.Config) -> None:
    import math

    network = neat.nn.FeedForwardNetwork.create(genome, config)
    rng = random.Random(0)
    for _ in range(20):
        values = [rng.uniform(-1.0, 1.0) for _ in range(payload["inputs"])]
        expected = list(network.activate(values))
        actual = []
        for row, bias, activation in zip(
            payload["weights"], payload["biases"], payload["activations"], strict=True
        ):
            total = bias + sum(
                weight * value for weight, value in zip(row, values, strict=True)
            )
            if activation == "tanh":
                actual.append(math.tanh(max(-60.0, min(60.0, 2.5 * total))))
            elif activation == "relu":
                actual.append(max(0.0, total))
            else:
                z = max(-60.0, min(60.0, 5.0 * total))
                actual.append(1.0 / (1.0 + math.exp(-z)))
        for left, right in zip(expected, actual, strict=True):
            if abs(left - right) > 1e-6:
                raise AssertionError(f"export mismatch: {left} != {right}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="export-champion")
    parser.add_argument("--genome", default=DEFAULT_GENOME)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    with Path(args.genome).open("rb") as handle:
        genome = pickle.load(handle)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )
    payload = export(genome, config)
    verify(payload, genome, config)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"exported {args.genome} -> {target}")


if __name__ == "__main__":
    main()
