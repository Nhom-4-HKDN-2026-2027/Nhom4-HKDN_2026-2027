# Data Quality — OpenAQ & Open-Meteo: Hanoi and HCMC

**Sprint 1 · Verified 2026-11-16 · Python 3.11.18.** UTC evaluation window: `[2024-11-14, 2026-11-14)`, **17,520 hours**. Manifest/checksums were verified; summaries, gaps and both samples were reproduced from this backfill.

**Sources and scope.** The inventory contains **62 locations**: 36 returned **112,673 hourly PM2.5 rows**, 7 returned no data in the window, and 19 were skipped because their last-observation metadata preceded the window. No `fetch_failed` or `not_fetched` statuses remain. Missing records do not prove station closure. Each location uses one PM2.5 sensor; all locations in this inventory had one candidate. ERA5 weather was requested at the coordinates of **5 selected stations**; it is gridded data rather than weather measurements at those stations.

**Definitions and proposed screening.** A valid hour is a unique hourly timestamp with a finite, nonnegative PM2.5 value; completeness = valid hours / 17,520. Raw NaN/negative values are retained; no cross-station imputation is applied. Deduplication prioritizes valid values. Values >500 µg/m³ are retained and flagged for review, not assumed to be sensor errors. The current demo rule requires <60% missing hours over two years, a valid observation on or after 2026-11-01, and no values >500. This is a **proposal, not a mentor-approved threshold or sensor-quality standard**. `is_monitor` is not mandatory. Gap counts below exclude boundary gaps, which are reported separately; stations without records appear in lifecycle counts rather than fabricated measurement rows.

| Station (ID; original names retained) | City | Valid hours | Two-year completeness | Longest internal gap (h) |
|---|---|---:|---:|---:|
| CMT8 (3276359) | HCMC | 16,048 | 91.60% | 690 |
| Công viên Nhân Chính (4946812) | Hanoi | 11,531 | 65.82% | 86 |
| Số 46 Lưu Quang Vũ (2161292) | Hanoi | 11,392 | 65.02% | 859 |
| Số 1 Giải Phóng (4946813) | Hanoi | 11,098 | 63.34% | 136 |
| Dragon (6123215) | Hanoi | 8,646 | 49.35% | 119 |

**Missing data and joins.** There are **1,150 internal gaps** across stations with records and zero duplicate station/time keys in both raw datasets. Weather contains **87,600 rows**, but all five variables are missing for **72 hours per station**, from `2026-11-11T00:00:00Z` through `2026-11-13T23:00:00Z`, under the ERA5 archive-delay policy applied by the pipeline: **1,800 NaN cells**. A complete timestamp grid does not mean complete values. Join-key coverage for selected stations is **100%**; **99.49%** of their PM2.5 rows have all five weather values.

**API and rate limits.** OpenAQ v3 uses `/locations` and `/sensors/{id}/hours`, pagination and bounded retries for timeout/connection errors, 429 and 500/502/503/504; 400/401/403 are not retried. Waiting follows `Retry-After` or `x-ratelimit-reset` ([OpenAQ documentation](https://docs.openaq.org/using-the-api/rate-limits)). Response headers were not retained, so the account quota is **unverified**. [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) uses `era5`, UTC and units °C, %, m/s, ° and mm; ERA5 has an approximately five-day delay. Missing values are retained without automatic model substitution or interpolation.

**Samples and reproducibility.** `data/sample/` contains [PM2.5 — 815 rows](../data/sample/openaq_pm25_hanoi_hcmc.csv) and [weather — 840 rows](../data/sample/openmeteo_weather_hanoi_hcmc.csv), covering `[2025-11-15, 2025-11-22)` UTC. Samples are extracted from the two-year raw data, not fetched separately. Fetch/quality code is in `ingest/`; all 7 notebook code cells ran on Python 3.11, **117 tests pass**, and Ruff checks pass. Follow the [setup guide](ingest_setup.md) to backfill using your own API key. Full raw data is not included in Git. The same raw snapshot reproduces the results; a new API fetch may differ as the source changes.

**Decision.** The team has agreed to use **Beijing Multi-Site for experiments and Vietnam data for the demo**. The [Beijing Multi-Site sample](../data/sample/beijing_multisite_features.csv) is included: 2,016 rows (168 hours × 12 stations), mapped to the seven shared feature columns. Humidity is derived from TEMP/DEWP and wind directions are mapped to degrees. It retains 26 missing PM2.5 values and 14 missing directions. Source timestamps are interpreted as Asia/Shanghai before UTC conversion; this is a documented assumption because the UCI page omits timezone. See [source, conversions and reproduction](ingest_setup.md#6-beijing-multi-site-source-and-reproduction).

The shared adapter in `ingest/feature_contract.py` joins Vietnam PM2.5/weather and gives both sources the same in-memory types: string station IDs, UTC timestamps and five float columns. Missing values are preserved; duplicate keys and malformed schema are rejected. CSV formats and the shared contract remain unchanged.
