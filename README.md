# YouTube Content Factory

Система, яка щодня автоматично робить відео для YouTube-каналу про стоїцизм і
практичну філософію (англійською, 25–35 хв) і публікує їх.

**Поточний стан:** тільки каркас. Усі етапи pipeline — заглушки.

## Запуск

```
python run.py            # створює запис у БД і проходить усі етапи
python run.py --dry-run  # лише показує етапи, БД не чіпає
```

Потрібен Python 3.11+. Зовнішніх залежностей поки немає.

## Структура

- `config.json` — налаштування каналу та відео
- `run.py` — оркестратор (помилка етапу → статус `failed`, процес не падає)
- `core/` — config, db (SQLite), logger, retry, validate
- `pipeline/` — етапи: topic, script, images, tts, render, thumbnail, seo, upload
- `assets/` — музика і шрифти
- `data/factory.db` — база (у .gitignore)
- `output/` — результати (у .gitignore)
- `logs/factory.log` — лог (у .gitignore)
