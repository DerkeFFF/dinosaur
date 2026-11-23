const MODEL_URL = "./champion.json";
const MAX_STEPS = 40000;

const frame = document.getElementById("game");
const ui = {
  episode: document.getElementById("episode"),
  steps: document.getElementById("steps"),
  best: document.getElementById("best"),
  speed: document.getElementById("speed"),
  status: document.getElementById("status"),
  restart: document.getElementById("restart"),
  model: document.getElementById("model"),
};

const state = {
  model: null,
  runner: null,
  win: null,
  running: false,
  pending: false,
  steps: 0,
  best: 0,
  episode: 1,
  duckRemaining: 0,
  jumpCooldown: 0,
  drawTick: 0,
};

const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, value));
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function typeCode(item) {
  if (!item || !item.type) return 0;
  if (item.type === "CACTUS_SMALL") return 1;
  if (item.type === "CACTUS_LARGE") return 2;
  if (item.type === "PTERODACTYL") {
    const y = Number(item.yPos || 0);
    if (y >= 90) return 3;
    if (y >= 65) return 4;
    return 5;
  }
  return 0;
}

const oneHot = (code) => [0, 1, 2, 3, 4, 5].map((index) => (index === code ? 1 : 0));
const hasObstacle = (item) => Boolean(item && item.type && item.distance < 1000);

function ttc(item) {
  if (!hasObstacle(item)) return clamp(1000 / 6 / 150);
  if (!item.speed) return 0;
  return clamp(item.distance / item.speed / 150);
}

// 24 признака идут ровно в том же порядке, что в Python `perception.py`.
// Иначе JSON-сеть получит не те числа и политика развалится.
function features(obs) {
  const first = obs.first;
  const second = obs.second;
  const dino = obs.dino;
  const values = [ttc(first), ...oneHot(typeCode(first)), clamp((first.yPos || 0) / 150), clamp((first.width || 0) / 100)];
  values.push(ttc(second), ...oneHot(typeCode(second)), clamp((second.yPos || 0) / 150), clamp((second.width || 0) / 100));
  values.push(
    hasObstacle(second)
      ? clamp((second.distance - first.distance) / 500)
      : 1
  );
  values.push(clamp(obs.speed / 13));
  values.push(dino.jumping ? 1 : 0);
  values.push(dino.ducking ? 1 : 0);
  values.push(clamp(dino.yPos / 150));
  values.push(dino.speedDrop ? 1 : 0);
  return values;
}

function validActions(obs) {
  let actions = hasObstacle(obs.first) ? [0, 1, 2] : [0, 1];
  if (obs.dino.jumping) actions = actions.filter((action) => action !== 1);
  return actions.length ? actions : [0];
}

function resolveAction(action, remaining) {
  if (action === 1) return [1, 0];
  if (action === 2) return [2, 4];
  if (remaining > 0) return [2, remaining - 1];
  return [0, 0];
}

function activation(kind, value) {
  // ВАЖНО: NEAT 2.0 масштабирует tanh на 2.5, sigmoid на 5.
  // Это не обычные math.tanh/math.sigmoid; иначе выходы не совпадут с Python.
  if (kind === "tanh") {
    return Math.tanh(Math.max(-60, Math.min(60, 2.5 * value)));
  }
  if (kind === "sigmoid") {
    const scaled = Math.max(-60, Math.min(60, 5 * value));
    return 1 / (1 + Math.exp(-scaled));
  }
  return Math.max(0, value);
}

function evaluate(values) {
  const { weights, biases, activations } = state.model;
  return weights.map((row, output) => {
    let total = biases[output];
    for (let index = 0; index < row.length; index += 1) {
      total += row[index] * values[index];
    }
    return activation(activations[output], total);
  });
}

function describe(obstacle, dino, speed) {
  return {
    distance: obstacle.xPos - dino.xPos,
    type: obstacle.typeConfig ? obstacle.typeConfig.type : null,
    yPos: typeof obstacle.yPos === "number" ? obstacle.yPos : null,
    width: typeof obstacle.width === "number" ? obstacle.width : 0,
    speed: speed + (typeof obstacle.speedOffset === "number" ? obstacle.speedOffset : 0),
  };
}

function observation(runner) {
  // Собираем те же first/second/speed/dino, что отдаёт Python-симулятор.
  const dino = runner.tRex;
  const speed = runner.currentSpeed || 0;
  const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
  const ahead = obstacles
    .filter((obstacle) => obstacle.xPos + (obstacle.width || 0) > dino.xPos)
    .sort((left, right) => left.xPos - right.xPos);
  const empty = { distance: 1000, type: null, yPos: null, width: 0, speed };
  return {
    first: ahead.length ? describe(ahead[0], dino, speed) : empty,
    second: ahead.length > 1 ? describe(ahead[1], dino, speed) : { ...empty },
    speed,
    dino: {
      jumping: Boolean(dino.jumping),
      ducking: Boolean(dino.ducking),
      yPos: typeof dino.yPos === "number" ? dino.yPos : 0,
      speedDrop: Boolean(dino.speedDrop),
    },
  };
}

function applyAction(runner, action) {
  const dino = runner.tRex;
  if (action === 1) {
    if (dino.jumping) return;
    if (dino.ducking) dino.setDuck(false);
    dino.startJump(runner.currentSpeed);
    return;
  }
  if (action === 2) {
    if (dino.jumping) {
      if (!dino.speedDrop) dino.setSpeedDrop();
    } else if (!dino.ducking) {
      dino.setDuck(true);
    }
    return;
  }
  if (dino.ducking) dino.setDuck(false);
  if (dino.speedDrop) dino.speedDrop = false;
}

function setStatus(text, error = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle("error", error);
}

function drawStats() {
  ui.episode.textContent = String(state.episode);
  ui.steps.textContent = String(state.steps);
  ui.best.textContent = String(state.best);
  ui.speed.textContent = state.runner ? state.runner.currentSpeed.toFixed(1) : "—";
}

function prepareFrame() {
  // Оригинальная игра создаёт canvas внутри #main-frame-error.
  // Поэтому контейнер нельзя скрывать display:none — только чистим offline-контент.
  const doc = frame.contentDocument;
  const body = doc.body;
  body.style.margin = "0";
  body.style.width = "600px";
  body.style.height = "150px";
  body.style.overflow = "hidden";
  body.style.background = "#f7f7f7";
  const message = doc.getElementById("messageBox");
  if (message) message.style.display = "none";
  const wrapper = doc.getElementById("main-frame-error");
  if (wrapper) {
    wrapper.style.cssText =
      "display:block;position:relative;width:600px;height:150px;" +
      "margin:0;padding:0;overflow:hidden;visibility:visible;pointer-events:none;";
  }
  for (const id of ["main-content", "offline-resources"]) {
    const element = doc.getElementById(id);
    if (element) element.style.display = "none";
  }
}

function restartGame() {
  // stop() отменяет старый rAF, restart() запускает новую игру.
  // xPos=50 — та же стартовая точка, что в Python-симуляторе.
  const runner = state.runner;
  if (runner.outerContainerEl) {
    runner.outerContainerEl.style.width = "600px";
    runner.outerContainerEl.style.height = "150px";
    runner.outerContainerEl.style.padding = "0";
  }
  if (runner.containerEl) {
    runner.containerEl.style.width = "600px";
    runner.containerEl.style.height = "150px";
  }
  runner.adjustDimensions();
  runner.stop();
  runner.paused = false;
  runner.crashed = false;
  runner.playing = true;
  runner.restart();
  runner.activated = true;
  runner.playingIntro = false;
  if (runner.tRex) {
    runner.tRex.playingIntro = false;
    runner.tRex.xPos = 50;
  }
  state.steps = 0;
  state.duckRemaining = 0;
  state.jumpCooldown = 0;
  state.running = true;
  state.pending = false;
  setStatus("Играет");
  drawStats();
}

function scheduleRestart(message) {
  if (state.pending) return;
  state.running = false;
  state.pending = true;
  state.best = Math.max(state.best, state.steps);
  setStatus(message);
  drawStats();
  setTimeout(() => {
    state.episode += 1;
    restartGame();
    requestAnimationFrame(tick);
  }, 900);
}

function tick() {
  if (!state.running) return;
  const runner = state.runner;
  if (!runner) return;
  if (runner.crashed || !runner.playing) {
    scheduleRestart(`Крах на ${state.steps} шагах — перезапуск`);
    return;
  }

  // Главный цикл браузерной версии:
  // observation → features → network → valid action → apply → game step.
  const obs = observation(runner);
  const values = features(obs);
  const actions = validActions(obs);
  if (state.jumpCooldown > 0) {
    const filtered = actions.filter((action) => action !== 1);
    actions.length = 0;
    actions.push(...(filtered.length ? filtered : [0]));
  }
  let action = actions[0];
  let best = -Infinity;
  const outputs = evaluate(values);
  for (const candidate of actions) {
    if (outputs[candidate] > best) {
      best = outputs[candidate];
      action = candidate;
    }
  }
  const resolved = resolveAction(action, state.duckRemaining);
  state.duckRemaining = resolved[1];
  state.jumpCooldown = action === 1 ? 3 : Math.max(0, state.jumpCooldown - 1);
  applyAction(runner, resolved[0]);

  state.steps += 1;
  if (state.steps % 5 === 0) drawStats();
  if (state.steps >= MAX_STEPS) {
    scheduleRestart(`${MAX_STEPS} шагов — новый эпизод`);
    return;
  }
  requestAnimationFrame(tick);
}

async function waitForGame() {
  state.win = frame.contentWindow;
  for (let attempt = 0; attempt < 300; attempt += 1) {
    const runner = state.win.Runner && state.win.Runner.instance_;
    if (runner && runner.containerEl && state.win.Runner.imageSprite) {
      return runner;
    }
    await delay(50);
  }
  throw new Error("игра не загрузилась");
}

async function boot() {
  try {
    setStatus("Загрузка игры...");
    prepareFrame();
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`champion.json: ${response.status}`);
    state.model = await response.json();
    ui.model.textContent = `Genome ${state.model.key} · ${state.model.fitness}`;
    state.runner = await waitForGame();
    restartGame();
    setStatus("Играет");
    requestAnimationFrame(tick);
  } catch (error) {
    console.error(error);
    setStatus(`Ошибка: ${error.message}`, true);
  }
}

ui.restart.addEventListener("click", () => {
  if (!state.runner) return;
  state.running = false;
  state.pending = false;
  restartGame();
  requestAnimationFrame(tick);
});

frame.addEventListener("load", boot, { once: true });
