"""Показ текущего лучшего генома в видимом окне Chrome."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import pickle
from pathlib import Path
from typing import Any

import neat
from playwright.async_api import async_playwright

from . import real_game

DEFAULT_GENOME = "models/champion-7553.pkl"


async def _badge(page: Any, genome: Any) -> None:
    label = f"Геном {getattr(genome, 'key', '?')}"
    with contextlib.suppress(Exception):
        await page.evaluate(
            """(label) => {
              let el = document.getElementById('genome-badge');
              if (!el) {
                el = document.createElement('div');
                el.id = 'genome-badge';
                el.style.cssText = 'position:fixed;left:8px;top:8px;z-index:2147483647;' +
                  'background:rgba(255,255,255,.9);color:#222;font:12px monospace;' +
                  'padding:4px 8px;border:1px solid #bbb;border-radius:3px;';
                document.body.appendChild(el);
              }
              el.textContent = label;
            }""",
            label,
        )


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dino-spectate",
        description="Показ лучшего генома NEAT в реальной игре",
    )
    parser.add_argument("--genome", default=DEFAULT_GENOME)
    parser.add_argument("--url", default=real_game.DEFAULT_URL)
    parser.add_argument("--config", default=str(real_game.DEFAULT_CONFIG))
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--step-delay", type=float, default=0.016)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )
    best_path = Path(args.genome)
    async with async_playwright() as playwright:
        context = await real_game._launch(
            playwright, headless=bool(args.headless or real_game.HEADLESS)
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
            await real_game._wait_runner(page)
            await real_game._focus(context, page)
            await real_game._style(page)
            while True:
                if not best_path.is_file():
                    print(f"ожидание {best_path}")
                    await real_game._sleep(2.0)
                    continue
                try:
                    # Перечитываем pkl каждый эпизод: если обучение обновило
                    # чемпиона, следующий запуск будет уже с новой сетью.
                    with best_path.open("rb") as handle:
                        genome = pickle.load(handle)
                    network = neat.nn.FeedForwardNetwork.create(genome, config)
                except Exception as error:
                    print(f"не удалось загрузить {best_path}: {error}")
                    await real_game._sleep(2.0)
                    continue
                await _badge(page, genome)
                steps, status = await real_game.run_episode(
                    page,
                    genome,
                    network,
                    max_steps=args.max_steps,
                    step_delay=args.step_delay,
                )
                print(
                    f"геном={getattr(genome, 'key', '?')} "
                    f"шаги={steps} статус={status}"
                )
                await real_game._sleep(0.5)
        finally:
            await context.close()


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
