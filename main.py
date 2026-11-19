import os
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from playwright.sync_api import sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

EPISODES = 300
EPISODES_PER_WORKER = 50
WORKERS = 4

ALPHA = 0.15
GAMMA = 0.95
EPSILON_START = 0.25
EPSILON_MIN = 0.03
EPSILON_DECAY = 0.995

STEP_DELAY = 0.02
MAX_STEPS = 3000

MODEL_FILE = "q_table.npy"
TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True

DISTANCE_BINS = 15
MAX_DISTANCE = 700

# state = (distance_bin, obstacle_type, obstacle_height_bin, speed_bin)
ACTIONS = [0, 1]  # 0 = wait, 1 = jump

q_table = {}
q_lock = threading.Lock()
epsilon = EPSILON_START
epsilon_lock = threading.Lock()


def state_key(distance, obstacle_type, obstacle_height, speed):
    distance = max(0, min(distance, MAX_DISTANCE))
    distance_bin = int(distance / (MAX_DISTANCE / DISTANCE_BINS))
    distance_bin = min(distance_bin, DISTANCE_BINS - 1)

    obstacle_type = str(obstacle_type)[:20]
    height_bin = int(obstacle_height // 10)
    speed_bin = int(speed // 0.5)

    return (distance_bin, obstacle_type, height_bin, speed_bin)


def get_q(state):
    with q_lock:
        if state not in q_table:
            q_table[state] = [0.0, 0.0]
        return q_table[state]


def choose_action(state):
    global epsilon

    with epsilon_lock:
        current_epsilon = epsilon

    if TRAIN_MODE and random.random() < current_epsilon:
        return random.choice(ACTIONS)

    values = get_q(state)
    return int(np.argmax(values))


def update_q_table(state, action, reward, next_state, done):
    with q_lock:
        current_q = q_table.setdefault(state, [0.0, 0.0])[action]

        if done or next_state is None:
            max_next_q = 0.0
        else:
            next_values = q_table.setdefault(next_state, [0.0, 0.0])
            max_next_q = max(next_values)

        new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
        q_table[state][action] = new_q


def decay_epsilon():
    global epsilon

    with epsilon_lock:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


def load_q_table():
    global q_table

    if LOAD_MODEL and os.path.exists(MODEL_FILE):
        loaded = np.load(MODEL_FILE, allow_pickle=True).item()

        # Старые ключи могут быть числами, новые — кортежами.
        q_table = {
            k: list(v) if isinstance(v, (list, np.ndarray)) else [0.0, 0.0]
            for k, v in loaded.items()
        }

        print(f"Загружена Q-таблица: {len(q_table)} состояний.")
    else:
        q_table = {}
        print("Q-таблица создана с нуля.")


def save_q_table():
    if SAVE_MODE and False:
        pass

    if SAVE_MODEL:
        with q_lock:
            np.save(MODEL_FILE, q_table)

        print(f"Q-таблица сохранена: {len(q_table)} состояний.")


def get_state(page):
    try:
        result = page.evaluate(
            """
            () => {
                const runner = window.Runner?.instance_;

                if (!runner) {
                    return { status: "NO_RUNNER" };
                }

                if (runner.crashed) {
                    return { status: "DEAD" };
                }

                if (!runner.playing) {
                    return { status: "NOT_PLAYING" };
                }

                const dino = runner.tRex;
                const obstacles = runner.horizon?.obstacles || [];

                if (obstacles.length === 0) {
                    return {
                        status: "OK",
                        distance: MAX_DISTANCE_FALLBACK,
                        type: "none",
                        height: 0,
                        speed: runner.currentSpeed || 0
                    };
                }

                const obstacle = obstacles[0];

                return {
                    status: "OK",
                    distance: obstacle.xPos - dino.xPos,
                    type: obstacle.typeConfig?.type || obstacle.type || "unknown",
                    height: obstacle.yPos || 0,
                    speed: runner.currentSpeed || 0
                };
            }
            """.replace(
                "MAX_DISTANCE_FALLBACK",
                str(MAX_DISTANCE)
            )
        )

        if result["status"] != "OK":
            return None, result["status"]

        state = state_key(
            distance=result["distance"],
            obstacle_type=result["type"],
            obstacle_height=result["height"],
            speed=result["speed"],
        )

        info = (
            f"dist={result['distance']:.0f}, "
            f"type={result['type']}, "
            f"h={result['height']:.0f}, "
            f"speed={result['speed']:.2f}"
        )

        return state, info

    except Exception as error:
        return None, f"ERROR: {error}"


def start_game(page):
    page.keyboard.press("Space")
    time.sleep(0.5)

    state, info = get_state(page)
    return state, info


def run_worker(worker_id):
    print(f"[Worker {worker_id}] Запуск.")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-gpu",
                "--disable-extensions",
                "--mute-audio",
            ]
        )

        context = browser.new_context(
            viewport={"width": 800, "height": 600}
        )

        page = context.new_page()
        page.goto(GAME_URL, wait_until="domcontentloaded")
        time.sleep(2)

        local_scores = []

        for episode in range(1, EPISODES_PER_WORKER + 1):
            state, info = start_game(page)

            if state is None:
                page.reload(wait_until="domcontentloaded")
                time.sleep(2)
                continue

            total_reward = 0.0
            steps = 0

            for step in range(1, MAX_STEPS + 1):
                action = choose_action(state)

                if action == 1:
                    page.keyboard.press("Space")

                time.sleep(STEP_DELAY)

                next_state, next_info = get_state(page)
                done = next_state is None

                if done:
                    reward = -100.0
                else:
                    reward = 1.0

                    # Небольшой бонус за выживание без прыжка,
                    # чтобы агент не прыгал без необходимости.
                    if action == 0:
                        reward += 0.05

                update_q_table(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                total_reward += reward
                steps = step
                state = next_state

                if done:
                    break

            local_scores.append(steps)
            decay_epsilon()

            print(
                f"[Worker {worker_id}] "
                f"Эпизод {episode}: шагов={steps}, "
                f"награда={total_reward:.1f}, "
                f"eps={epsilon:.3f}"
            )

            if episode % 10 == 0:
                save_q_table()

            time.sleep(0.2)

        browser.close()

    return worker_id, local_scores


def main():
    print("Запускаем параллельное обучение...")
    load_q_table()

    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(run_worker, worker_id)
            for worker_id in range(WORKERS)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")

    for worker_id, scores in results:
        if scores:
            print(
                f"Worker {worker_id}: "
                f"среднее число шагов = {np.mean(scores):.1f}, "
                f"лучший результат = {max(scores)}"
            )

    save_q_table()


if __name__ == "__main__":
    main()