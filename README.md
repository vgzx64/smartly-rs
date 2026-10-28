# smartly-rs

Диагностика SSD на Rust: **состояние диска → риск потери данных (ONNX ML-модель)
→ возможность восстановления → рекомендации → отдельный блок «можно ли удалить
данные без восстановления»**.

Программа **только читает** информацию (lsblk/smartctl/nvme) и **никогда не
выполняет** удаление/стирание — команды стирания выводятся справочно.

## Возможности

1. **Состояние SSD** — износ, запас резервных блоков, температура, ошибки,
   critical_warning, вердикт OK / WARNING / CRITICAL с причинами.
2. **Риск потери данных** — вероятность 0–100% от встроенной ML-модели
   (градиентный бустинг HGB, ONNX, `ort`), с интерпретацией вклада факторов.
3. **Возможность восстановления** — ВЫСОКАЯ / СРЕДНЯЯ / НИЗКАЯ + объяснение.
4. **Рекомендации** — пошаговый план (ddrescue, PhotoRec/R-Studio, лаборатория).
5. **Безвозвратное удаление** — ДА / ЧАСТИЧНО / НЕТ + методы
   (`nvme sanitize/format`, `hdparm --security-erase`, `blkdiscard`, уничтожение
   ключа LUKS/SED) и предупреждения.

## Сборка и запуск

```bash
cargo build --release

# Список устройств (lsblk)
cargo run -- --list

# Реальная диагностика (для SMART обычно нужны root и smartmontools)
sudo apt install smartmontools nvme-cli
cargo run --release -- --disk nvme0n1      # или /dev/sda, --disk /dev/sda

# Демо без доступа к диску (healthy | worn | critical)
cargo run -- --demo healthy
cargo run -- --demo critical --explain     # полная интерпретация модели
cargo run -- --demo worn --json            # JSON-отчёт (для интеграций)
```

## Что опрашивается (только чтение)

```bash
lsblk -J -b -o NAME,PATH,TYPE,MODEL,TRAN,SIZE,ROTA,SERIAL
smartctl -j -a /dev/nvmeXn1        # NVMe и SATA через один интерфейс
nvme id-ctrl /dev/nvmeXn1 -o json  # capabilities sanitize (опционально)
```

Если `smartctl`/`nvme` или права отсутствуют, программа не падает: выводит
идентификацию из `lsblk`, список примечаний и строит отчёт с ограниченной
точностью (это видно по полю `source` и `notes` в JSON).

## ML-модель (интеграция AI)

- **Что предсказывает**: вероятность отказа с потерей данных в ближайшие
  30 дней, 8 нормированных входных признаков → одно число 0..1.
- **На чём обучена**: реальный датасет **Alibaba ssd_smart_logs** (Tianchi
  `dataId=95044`, Xu et al., DSN 2021): ежедневные SMART-срезы ~400 тыс.
  SATA-SSD (2018–2019). Подробности — `docs/dataset.md`.
- **Выбор модели** (`assets/risk/train_risk_model.py`): сравниваются логистическая
  регрессия, её balanced-вариант и histogram gradient boosting; тест —
  сгруппированный по `model#disk_id` (срезы одного диска не разрываются).
  Победитель по PR-AUC экспортируется в `assets/risk/risk_model.onnx`
  (sklearn + skl2onnx, opset 15):

| модель | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| logistic | 0.163 | 0.759 | 0.037 |
| logistic_balanced | 0.182 | 0.789 | 0.190 |
| **HGB** ← экспортирована | **0.390** | **0.848** | **0.031** |

  Базовая частота в тесте — 0.041, т.е. случайный PR-AUC = 0.041.
- **Инференс в Rust**: крейт `ort` (ONNX Runtime), модель вшита в бинарник
  через `include_bytes!` — работает офлайн. Если ONNX-рантайм недоступен,
  применяется логистическая формула по коэффициентам
  (`assets/risk/risk_coefficients.json`).
- **Уровни риска**: пороги не зашиты в код, а выводятся на тестовой выборке
  (`assets/risk/train_metrics.json`) как наименьший скор, при котором точность
  срабатываний достигает цели, и читаются из JSON при старте:

| уровень | порог | precision | recall | lift |
|---|---|---|---|---|
| средний | ≥ 0.19 | 0.54 | 0.27 | 13× |
| высокий | ≥ 0.31 | 0.71 | 0.23 | 17× |
| критический | ≥ 0.62 | 0.86 | 0.18 | 21× |

  precision здесь — доля дисков, которые действительно отказали в течение
  30 дней среди помеченных моделью; базовый уровень 0.041.
- **Интерпретируемость**: `--explain` печатает вклад каждого признака
  (weight × value в log-odds) по коэффициентам линейной регрессии. Когда
  экспортирована HGB, вклады помечаются как условные: рядом выводится скор
  самого линейного приближения (`proxy_probability` в JSON), так что
  расхождение видно. Более надёжный ориентир — **глобальная значимость**
  (permutation importance на тесте, печатается в `--explain`): `poh` 0.169,
  `spare_deficit` 0.147, `realloc` 0.126, `crit_warn` 0.058,
  `pct_used` 0.049 (в единицах падения PR-AUC).
- **Признаки** (порядок = порядок входа модели): `pct_used, spare_deficit,
  media_err, temp, poh, crit_warn, realloc, smart_fail` — все нормированы
  в 0..1 (см. `src/features.rs`). Признаки `unsafe_shutdown` и `wear_cycles`
  исключены из модели: в обучающих данных первый был индикатором конкретной
  (самой «здоровой») модели дисков, а второй дублировал `pct_used`
  (подробнее — `docs/dataset.md`).

Переобучение модели:

```bash
.venv/bin/python assets/train_risk_model.py --data assets/train.csv   # v2, реальные данные
.venv/bin/python assets/train_risk_model.py                           # синтетика (только демо)
cargo build
```

**Ограничение честности**: обучение — на SATA-SSD дата-центра Alibaba,
инференс часто на NVMe (перенос только через документированный маппинг
полей); горизонт — «отказ в ближайшие 30 дней», а не «прямо сейчас»;
поведение бустинга вне обучающего распределения (например, когда все
признаки сразу выкручены в максимум) не гарантировано — на демо-сценарии
«critical» модель даёт ~23%, и уровень берётся из этого скора; линейные
коэффициенты как объяснение почти бесполезны (`media_err`, `temp`,
`crit_warn` в датасете почти константны, а «вес» `media_err` ≈ +10 —
артефакт), поэтому ориентируйтесь на permutation-значимость и поле
`state`. Модель — статистический риск-скор, а не приговор диску; движок
виден в отчёте (`engine`), а при отказе ONNX указывается `error`.

## Структура

```
src/
  main.rs      CLI (clap): --disk/--demo/--list/--json/--explain
  collect.rs   lsblk + smartctl -j + nvme id-ctrl (read-only)
  model.rs     единая структура Health, парсер SMART NVMe/SATA
  features.rs  8 нормализованных признаков
  infer.rs     ONNX-инференс (ort) + fallback + вклады факторов
  advise.rs    правила: состояние, восстановление, вердикт удаления
  report.rs    текстовый и JSON отчёт
  demo.rs      сценарии healthy/worn/critical
assets/        risk_model.onnx, risk_coefficients.json, train_metrics.json,
               eda_alibaba.py, build_train_alibaba.py, train_risk_model.py,
               analyze_train_marginals.py, make_features_fixture.py,
               train.csv (артефакт сборки)
tests/fixtures smartctl -j примеры (NVMe/SATA) + features_real.json (held-out)
docs/theory.md теория (SSD/NAND/контроллер/SMART/SATA vs NVMe) — этап 1
docs/dataset.md EDA датасета Alibaba DSN'21, маппинг полей и результаты v2
```

## Обучение на реальных данных (v2)

Датасет: **Alibaba ssd_smart_logs** (Tianchi `dataId=95044`, Xu et al., DSN 2021)
— ежедневные SMART-срезы ~400 тыс. SATA-SSD, 6 моделей, 2018–2019,
16 305 меток отказов. Скачан в `downloads/` (zip, ~9 ГБ).

Пайплайн:

```bash
# 1) EDA → docs/dataset.md (покрытие атрибутов, распределения, sanity-чеки)
.venv/bin/python assets/eda_alibaba.py

# 2) сборка обучающей выборки → assets/train.csv
#    позитив: срез за 1–30 дней до failure_time (горизонт 30 дней)
#    негатив: ~500 случайных «здоровых» дисков в день (~1:20)
.venv/bin/python assets/risk/build_train_alibaba.py

# 2b) диагностика: какие признаки вообще различают отказы (маргиналы,
#     коллинеарность) — вход для раздела про интерпретацию в docs/dataset.md
.venv/bin/python assets/risk/analyze_train_marginals.py

# 3) обучение v2 (кандидаты: logistic / logistic_balanced / HGB;
#    split по model#disk_id без утечек; метрики → assets/risk/train_metrics.json)
.venv/bin/python assets/risk/train_risk_model.py --data assets/risk/train.csv

# 4) выборка реальных held-out сэмплов для Rust-тестов + прогон тестов
.venv/bin/python assets/risk/make_features_fixture.py
cargo test
```

Собранная выборка: **372 930 строк / 14 930 позитивов (4.0%) / 148 175 дисков**;
позитивы — срез за 1–30 дней до отказа (все 16 305 меток нашли свой снимок),
негативы — 500 случайных «никогда не отказавших» дисков в день.

Контракт входа/выхода ONNX не менялся по форме (8 признаков → `probability`),
поэтому Rust-код перекомпилировать достаточно один раз — модель подхватится
из `include_bytes!`.

Особенности маппинга vendor-SMART на признаки (подробно — `docs/dataset.md`):
износ у разных вендоров в `n_173`/`n_233` (у MB1/MB2 отсутствует →
`wear_unknown=1`), host-write счётчики `r_241/r_242` немонотонны, поэтому
не использовались; `smart_fail = 0` (в датасете нет вердикта самотеста).

Переобучить синтетическую модель (для сравнения/демо):

```bash
.venv/bin/python assets/risk/train_risk_model.py   # без --data
```

## Результаты v2 и ограничения

Тест (20% дисков, никогда не виденные моделью) — PR-AUC **0.390** у HGB против
0.163 у логистической регрессии при базовой частоте 0.041. Разбивка по моделям
дисков (AP на тесте, `assets/risk/train_metrics.json`):

| MA1 | MC2 | MB1 | MC1 | MB2 | MA2 |
|---|---|---|---|---|---|
| 0.766 | 0.548 | 0.390 | 0.339 | 0.034 | 0.016 |

Слабые MA2/MB2 — это модели без нормализованного износа (`n_173`/`n_233`) и с
иными наборами атрибутов: по одним host-write счётчикам отказ предсказывается
плохо, что честно отражает данные, а не ошибку пайплайна. (У MA2 добавочный
фактор: её единственный «сигнал» `r_192`/unsafe_shutdowns был исключён из
признаков как индикатор модели, а не физический фактор риска — см.
`docs/dataset.md`.)

- Обучение — на **SATA-SSD** дата-центра Alibaba, инференс часто на **NVMe**:
  перенос только через документированный маппинг полей (см. `docs/dataset.md`).
- Горизонт предсказания — «отказ в ближайшие 30 дней», не «сейчас».
- Бустинг немонотонен: векторы вне обучающего распределения (все признаки
  разом в максимум) не обязаны получать высокий скор — Rust-тесты поэтому
  проверяются на реальных held-out сэмплах (`tests/fixtures/features_real.json`),
  а не на синтетических крайностях.
- Объяснения всегда строятся по коэффициентам логистической регрессии; для
  экспортированной HGB это приближение (`approx_explain: true`), и его
  собственный скор печатается рядом (`proxy_probability`). Диагностика
  маргиналов (`assets/risk/analyze_train_marginals.py`) показывает, почему этим
  вкладам нельзя верить буквально: `media_err`, `temp`, `crit_warn` в
  выборке почти константны.
- Уровни риска (`низкий/средний/высокий/критический`) — не константы, а
  пороги 0.19 / 0.31 / 0.62, выведенные на тестовой выборке под precision
  0.54 / 0.71 / 0.86 и вшитые в `assets/risk/risk_coefficients.json`.



## Модель нормальности (неконтролируемая, SMART)

Пока в `notebooks/SMART` нет меток отказа за 30 дней, contrôle риска
осуществляется через **неконтролируемую модель нормальности**: изучается
профиль "здорового" SSD, а значительные отклонения трактуются как тревога.

- **Данные**: `notebooks/SMART/parsed_smart_records.csv` (168 787 записей,
  233 столбца) — парсинг через `assets/smart/extract_smart_features.py`,
  покрытие признаков — `assets/smart/coverage_analysis.py`.
- **Обучение**: `assets/normality/train_normal_model.py` — IsolationForest
  обучается на "здоровых" записях (без `self_test == FAILED`,
  `media_err > 0`, `crit_warn > 0`, `pct_used >= 0.90`).
- **Вывод**: `assets/normality/anomaly_config.json` — порог тревоги
  (1-й процентиль здоровых ≈ 1% ложных срабатываний),
  `assets/normality/normal_model.joblib` — fitted IsolationForest.
- **Инференс**: `assets/normality/detect_anomaly.py` — scores a device's
  8-feature vector, печатает `NORMAL` / `ALARM`.

Валидация на обучающей выборке:healthy — 0.98% тревог (цель ≈1%),
устройства с признаками неисправности — 66% тревог (слабый сигнал,
не ground truth).

## Тесты

```bash
cargo test          # парсинг SMART, признаки, ML, правила, JSON
cargo clippy        # без предупреждений
```
