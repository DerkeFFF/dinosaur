# dino-master

агент для игры chrome dino. обучение проходит в python-симуляторе с использованием neat, а результат можно проверить в браузере.

## установка

```bash
uv sync
```

## запуск

обучение:

```bash
uv run dino-train --generations 50 --population 500 --max-steps 30000 --episodes-per-gen 4 --best models/best.pkl
```

проверка в chrome:

```bash
uv sync --extra real
uv run dino-eval --genome models/champion-7553.pkl --episodes 3 --max-steps 40000 --out eval.json
```

просмотр игры в браузере:

```bash
uv run dino-spectate --genome models/champion-7553.pkl
```

визуализация симулятора:

```bash
uv sync --extra viz
uv run dino-visualize --genome models/champion-7553.pkl
```

## основные файлы

- `world.py` - симулятор игры и физика
- `perception.py` - обработка наблюдений и признаки
- `learn.py` - обучение агента
- `browser_eval.py` - проверка в браузере
- `watch.py` - просмотр игры в браузере
- `render.py` - визуализация симулятора
- `models/` - сохранённые модели

для проверки в браузере нужны google chrome и доступ к сети. если chrome не найден автоматически, укажите путь через переменную `DINO_CHROME_EXECUTABLE`.
