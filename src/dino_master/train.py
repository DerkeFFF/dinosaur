"""Точка входа обучения и совместимость со старым именем модуля."""

from .learn import *  # noqa: F401,F403
from .learn import main

if __name__ == "__main__":
    main()
