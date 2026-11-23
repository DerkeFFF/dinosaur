"""Проверка сохранённого генома NEAT на реальной игре Chrome Dino."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import neat
from playwright.async_api import async_playwright

from . import features

DEFAULT_URL = "https://wayou.github.io/t-rex-runner/"
DEFAULT_CONFIG = Path(__file__).with_name("neat.ini")
DEFAULT_GENOME = "models/champion-7553.pkl"
PROFILE_DIR = Path(".chrome_profiles_eval")
HEADLESS = os.environ.get("DINO_HEADLESS") == "1"


def _chrome() -> str:
    env = os.environ.get("DINO_CHROME_EXECUTABLE")
    if env:
        return env
    cached = Path.home() / ".cache" / "dino-chrome" / "chrome"
    if cached.exists():
        return str(cached)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("Chrome не найден; задайте DINO_CHROME_EXECUTABLE")


START_GAME_JS = """() => {
  const Runner = window.Runner;
  const r = Runner && Runner.instance_;
  if (!r) return false;
  const message = document.getElementById("messageBox");
  if (message) message.style.display = "none";
  if (typeof r.stop === "function") r.stop();
  r.paused = false;
  r.crashed = false;
  r.restart();
  r.activated = true;
  r.playingIntro = false;
  if (r.tRex) {
    r.tRex.playingIntro = false;
    r.tRex.xPos = 50;
  }
  if (r.containerEl) {
    r.containerEl.style.width = r.dimensions.WIDTH + "px";
    r.containerEl.style.height = r.dimensions.HEIGHT + "px";
  }
  if (!r.playing && typeof r.play === "function") r.play();
  r.update();
  return Boolean(r.playing);
}"""

GET_OBSERVATION_JS = """() => {
  const runner = window.Runner && window.Runner.instance_;
  if (!runner) return {status: "NO_RUNNER"};
  if (runner.crashed) return {status: "DEAD"};
  if (!runner.playing) return {status: "NOT_PLAYING"};
  const dino = runner.tRex;
  const currentSpeed = runner.currentSpeed || 0;
  const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
  const ahead = obstacles.filter(
    (obstacle) => obstacle.xPos + (obstacle.width || 0) > dino.xPos
  );
  ahead.sort((left, right) => left.xPos - right.xPos);
  const describe = (obstacle) => {
    const config = obstacle.typeConfig || null;
    const offset = typeof obstacle.speedOffset === "number" ? obstacle.speedOffset : 0;
    if (!obstacle.__dinoId) {
      obstacle.__dinoId = (window.__dinoObstacleId || 0) + 1;
      window.__dinoObstacleId = obstacle.__dinoId;
    }
    return {
      id: obstacle.__dinoId,
      distance: obstacle.xPos - dino.xPos,
      type: config ? config.type : null,
      yPos: typeof obstacle.yPos === "number" ? obstacle.yPos : null,
      width: typeof obstacle.width === "number" ? obstacle.width : 0,
      speed: currentSpeed + offset
    };
  };
  const empty = {id: null, distance: 1000, type: null, yPos: null, width: 0, speed: currentSpeed};
  return {
    status: "OK",
    first: ahead.length > 0 ? describe(ahead[0]) : empty,
    second: ahead.length > 1 ? describe(ahead[1]) : empty,
    speed: currentSpeed,
    dino: {
      jumping: Boolean(dino.jumping),
      ducking: Boolean(dino.ducking),
      yPos: typeof dino.yPos === "number" ? dino.yPos : 0,
      speedDrop: Boolean(dino.speedDrop)
    }
  };
}"""


async def _launch(playwright: Any, *, headless: bool) -> Any:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    # Эти флаги не дают Chrome затормозить/остановить rAF в неактивном окне.
    args = [
        "--window-size=960,540",
        "--disable-gpu",
        "--mute-audio",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling",
    ]
    kwargs: dict[str, Any] = {
        "user_data_dir": str(PROFILE_DIR),
        "headless": headless,
        "no_viewport": True,
        "args": args,
    }
    kwargs["executable_path"] = _chrome()
    return await playwright.chromium.launch_persistent_context(**kwargs)


async def _wait_runner(page: Any) -> None:
    await page.wait_for_function(
        "() => {"
        "  const r = window.Runner && window.Runner.instance_;"
        "  return !!(r && r.containerEl);"
        "}",
        timeout=30_000,
    )


async def _focus(context: Any, page: Any) -> None:
    with contextlib.suppress(Exception):
        cdp = await context.new_cdp_session(page)
        await cdp.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
        await cdp.send("Page.setWebLifecycleState", {"state": "active"})


async def _style(page: Any) -> None:
    with contextlib.suppress(Exception):
        await page.evaluate(
            "document.body.style.zoom = '1.5';"
            "document.body.style.overflow = 'hidden';"
        )


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def start_game(page: Any) -> bool:
    started = bool(await page.evaluate(START_GAME_JS))
    if started:
        await _sleep(0.3)
        return True
    await page.keyboard.press("Space")
    await _sleep(1.0)
    started = bool(await page.evaluate(START_GAME_JS))
    if started:
        return True
    await page.locator("canvas").first.click(timeout=1000)
    await page.keyboard.press("Space")
    await _sleep(1.0)
    return bool(await page.evaluate(START_GAME_JS))


async def get_observation(page: Any) -> tuple[dict[str, Any] | None, str]:
    try:
        obs = await page.evaluate(GET_OBSERVATION_JS)
    except Exception:
        return None, "ERROR"
    if not isinstance(obs, dict):
        return None, "ERROR"
    status = str(obs.get("status", "ERROR"))
    return (obs if status == "OK" else None), status


async def apply_action(page: Any, action: int, duck_held: bool) -> bool:
    if action == 2:
        if not duck_held:
            await page.keyboard.down("ArrowDown")
        return True
    if duck_held:
        await page.keyboard.up("ArrowDown")
    if action == 1:
        await page.keyboard.press("Space")
    return False


async def run_episode(
    page: Any,
    genome: Any,
    network: Any,
    *,
    max_steps: int,
    step_delay: float,
) -> tuple[int, str]:
    if not await start_game(page):
        return 0, "NOT_PLAYING"
    obs, status = await get_observation(page)
    if obs is None:
        return 0, status
    steps = 0
    duck_remaining = 0
    duck_held = False
    jump_cooldown = 0
    try:
        # Один шаг = observation → features → network → action.
        for _ in range(max_steps):
            values = features.observation_features(obs)
            valid = features.valid_actions(obs)
            if jump_cooldown > 0:
                valid = [action for action in valid if action != 1] or [0]
            outputs = list(network.activate(values))
            raw = max(valid, key=lambda action: float(outputs[action]))
            action, duck_remaining = features.resolve_action(raw, duck_remaining)
            jump_cooldown = (
                features.JUMP_COOLDOWN_STEPS
                if action == 1
                else max(0, jump_cooldown - 1)
            )
            duck_held = await apply_action(page, action, duck_held)
            await _sleep(step_delay)
            next_obs, next_status = await get_observation(page)
            steps += 1
            if next_obs is None:
                return steps, next_status
            obs = next_obs
    finally:
        # Не оставляем ArrowDown зажатой после смерти/исключения.
        if duck_held:
            with contextlib.suppress(Exception):
                await apply_action(page, 0, True)
    return steps, "MAX_STEPS"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dino-eval",
        description="Проверка генома NEAT на реальной игре Chrome Dino",
    )
    parser.add_argument("--genome", default=DEFAULT_GENOME)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--step-delay", type=float, default=0.016)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out", default="eval.json")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes должно быть положительным")
    if args.max_steps < 1:
        parser.error("--max-steps должно быть положительным")
    return args


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )
    with Path(args.genome).open("rb") as handle:
        genome = pickle.load(handle)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    results: list[dict[str, Any]] = []

    async with async_playwright() as playwright:
        context = await _launch(playwright, headless=bool(args.headless or HEADLESS))
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
            await _wait_runner(page)
            await _focus(context, page)
            await _style(page)
            for episode in range(1, args.episodes + 1):
                steps, status = await run_episode(
                    page,
                    genome,
                    network,
                    max_steps=args.max_steps,
                    step_delay=args.step_delay,
                )
                result = {
                    "episode": episode,
                    "genome": getattr(genome, "key", None),
                    "steps": steps,
                    "status": status,
                }
                results.append(result)
                print(json.dumps(result))
        finally:
            await context.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = [item for item in results if item["status"] not in {"DEAD", "MAX_STEPS"}]
    if failed:
        raise SystemExit(1)


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
