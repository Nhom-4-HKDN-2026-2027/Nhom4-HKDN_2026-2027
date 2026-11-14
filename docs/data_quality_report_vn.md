# Data Quality Report: OpenAQ & Open-Meteo for Hanoi and Ho Chi Minh City

- **Project:** Adaptive Model Updating in Automated Data Pipelines (Sprint 1: Foundations & Data Contracts)
- **Author:** Nguyen Quoc Tuong (`@ngynqt`)
- **Date:** 2026-11-14
- **2-Year Evaluation Window (UTC):** `2024-11-14T00:00:00Z` → `2026-11-14T00:00:00Z` (**17,520 hours**)

---

## 1. Executive Summary & Architectural Decision

> **Decision:** Vietnam data (OpenAQ + Open-Meteo) will be used for the **Live Demo & Streaming Pipeline**, while **Offline Drift & Adaptive Model Updating Experiments** will use the **Beijing Multi-Site Air-Quality** benchmark dataset (supplemented by validation on active Vietnam stations).

### Evidence from the 2-Year Assessment (`2024-11` – `2026-11`)
1. **Historical Continuity in Vietnam.** Out of 62 PM2.5 stations listed across Hanoi (43) and Ho Chi Minh City (19), 26 stations ceased reporting before `2024-11`. Among the 36 stations with records in the last 2 years, the two US Diplomatic Post reference stations (`7441` in Hanoi and `7440` in HCMC) stopped reporting in March–April 2025 (notably, station `7440` in HCMC returned 2,652 hourly rows that are **100% `NaN`**). Meanwhile, Hanoi's newer municipal reference stations (`4946812`, `4946813`) only began operating on `2025-07-03` (~16.5 months), which is insufficient for standalone multi-year seasonal drift benchmarking.
2. **Suitability for Live Demo & Streaming Pipeline.** Currently, **5 active stations (live through `2026-11`)** achieve **65.10% to 97.16%** completeness during their active spans (`CMT8` in HCMC; `Công viên Nhân Chính`, `Số 1 Giải Phóng`, `Dragon`, and `Số 46 Lưu Quang Vũ` in Hanoi). Paired with **Open-Meteo's 100% complete hourly meteorological data**, this dataset fully supports the end-to-end real-time streaming pipeline (Redpanda → MinIO → dbt → FastAPI).

---

## 2. API Access & Rate Limits

### OpenAQ API v3
- **Endpoints** · `GET /v3/locations`, `GET /v3/sensors/{id}/hours`
- **Authentication** · Header `X-API-Key`
- **Verified Rate Limits** · **60 requests / minute** (`x-ratelimit-limit: 60`, `x-ratelimit-reset: 60` seconds; maximum `1,000` records per page)
- **Pipeline Operations** · Historical backfills across 36 stations require a `~0.9s–1.0s` delay per request and HTTP `429` retry handling. For the Live Demo (5 stations polled hourly), consumption is only 5 requests/hour (< 10% of the per-minute limit).

### Open-Meteo API
- **Endpoints** · `GET /v1/archive`, `GET /v1/forecast`
- **Authentication** · No API key required (Free Tier)
- **Verified Capabilities** · Retrieves the entire 2-year hourly history (17,520 hours) for a station in a **single HTTP request**
- **Pipeline Operations** · Supports direct `wind_speed_unit=ms` and `UTC` timestamps, matching `docs/CONTRACTS.md`.

---

## 3. Usable PM2.5 Stations & 2-Year Completeness (`2024-11-14` → `2026-11-14`)

After auditing all 62 locations (36 stations with records during the evaluation period, yielding **112,661 hourly rows**), the following **5 usable stations** remain actively reporting through `2026-11`:

### 3.1. Overview of the 5 Usable Stations

- **CMT8** (HCMC, `3276359`) — Primary HCMC station. **16,048** / 17,520 valid hours. Missing rate is **8.40%** over the full 2-year window and **7.72%** during its active span. Longest gap: `690h`.
- **Công viên Nhân Chính - Khuất Duy Tiến** (Hanoi, `4946812`) — Primary Hanoi station. **11,531** / 17,520 valid hours. Missing rate is **34.18%** over 2 years and **3.60%** during its active span. Longest gap: `86h`.
- **Số 46, phố Lưu Quang Vũ** (Hanoi, `2161292`) — Hanoi station with records spanning nearly the full evaluation window. **11,392** / 17,520 valid hours. Missing rate is **34.98%** over 2 years and **34.90%** during its active span. Longest gap: `859h`.
- **Số 1 đường Giải Phóng - Bạch Mai - ĐHBK** (Hanoi, `4946813`) — Primary Hanoi station. **11,098** / 17,520 valid hours. Missing rate is **36.66%** over 2 years and **7.22%** during its active span. Longest gap: `136h`.
- **Dragon** (Hanoi, `6123215`) — Supplementary Hanoi station. **8,646** / 17,520 valid hours. Missing rate is **50.65%** over 2 years and **2.84%** during its active span. Longest gap: `119h`.

### 3.2. Detailed Station Profiles

#### 1. [HCMC] CMT8 (`station_id: 3276359`, `sensor_id: 11357424`)
- **Provider & Type** · AirGradient (`is_monitor: False`)
- **Coordinates** · `(10.78533, 106.67029)`
- **Valid Data Span** · `2024-11-19 10:00` → `2026-11-13 23:00 UTC` (Active span: `17,390 hours`)
- **Completeness** · `16,048 / 17,520` valid hours. Missing **8.40%** over 2 years (**7.72%** during active span)
- **Gaps** · `62` gaps in total (`5` gaps `>= 24h`, longest gap: **690h / 28.8 days**).

#### 2. [Hanoi] Công viên Nhân Chính - Khuất Duy Tiến (`station_id: 4946812`, `sensor_id: 13502151`)
- **Provider & Type** · Hanoi Air Quality Monitoring Network (`is_monitor: True`)
- **Coordinates** · `(21.0031, 105.7947)`
- **Valid Data Span** · `2025-07-03 15:00` → `2026-11-13 23:00 UTC` (Active span: `11,961 hours`)
- **Completeness** · `11,531 / 17,520` valid hours. Missing **34.18%** over 2 years (only **3.60%** since deployment)
- **Gaps** · `63` gaps in total (`4` gaps `>= 24h`, longest gap: **86h / 3.6 days**).

#### 3. [Hanoi] Số 46, phố Lưu Quang Vũ (`station_id: 2161292`, `sensor_id: 7772032`)
- **Provider & Type** · Hanoi Air Quality Monitoring Network (`is_monitor: True`)
- **Coordinates** · `(21.0152, 105.7999)`
- **Valid Data Span** · `2024-11-14 00:00` → `2026-11-13 01:00 UTC` (Active span: `17,498 hours`)
- **Completeness** · `11,392 / 17,520` valid hours (`57` `NaN` rows). Missing **34.98%** over 2 years (**34.90%** during active span)
- **Gaps** · `228` gaps in total (`61` gaps `>= 24h`, longest gap: **859h / 35.8 days**).

#### 4. [Hanoi] Số 1 đường Giải Phóng - Bạch Mai - ĐHBK (`station_id: 4946813`, `sensor_id: 13502154`)
- **Provider & Type** · Hanoi Air Quality Monitoring Network (`is_monitor: True`)
- **Coordinates** · `(21.0052, 105.8418)`
- **Valid Data Span** · `2025-07-03 15:00` → `2026-11-13 23:00 UTC` (Active span: `11,961 hours`)
- **Completeness** · `11,098 / 17,520` valid hours. Missing **36.66%** over 2 years (**7.22%** since deployment)
- **Gaps** · `189` gaps in total (`8` gaps `>= 24h`, longest gap: **136h / 5.7 days**).

#### 5. [Hanoi] Dragon (`station_id: 6123215`, `sensor_id: 14581375`)
- **Provider & Type** · AirGradient (`is_monitor: False`)
- **Coordinates** · `(20.9933, 105.9441)`
- **Valid Data Span** · `2025-11-08 05:00` → `2026-11-13 23:00 UTC` (Active span: `8,899 hours`)
- **Completeness** · `8,646 / 17,520` valid hours. Missing **50.65%** over 2 years (only **2.84%** since deployment)
- **Gaps** · `42` gaps in total (`2` gaps `>= 24h`, longest gap: **119h / 5.0 days**).

### 3.3. Inactive or Excluded Reference Stations
- **556 Nguyễn Văn Cừ (`4946811` - Hanoi):** `7,829` valid hours (`2025-07-03` → `2026-07-15`), missing `55.31%` over 2 years (`13.43%` during active span, max gap `625h`). Offline since July 2026.
- **Minh Khai - Bắc Từ Liêm (`2161306` - Hanoi):** `4,292` valid hours (`2024-11-14` → `2026-11-13`), missing `75.50%`, longest outage **3,849h (~160 days)**, and contains **13 extreme outliers `> 500 µg/m³`** (up to `872 µg/m³`).
- **US Diplomatic Post Hanoi (`7441`) & HCMC (`7440`):** Hanoi station has `2,905` valid hours (`585` `NaN` rows) and stopped on `2025-04-09`; HCMC station returned `2,652` rows that are **100% `NaN`** and stopped on `2025-03-04`.

---

## 4. Data Gaps, Anomalies & Handling Policy

Following the team's data engineering policy (**no cross-station imputation**; preserve missing intervals and explicitly log outage windows):
1. **Longest Outage Intervals (`Gaps >= 24h`) Across the 5 Usable Stations:**
   - **Số 46, phố Lưu Quang Vũ (`2161292`):** 61 outages `>= 24h`; longest intervals: `2026-06-15 17:00` → `2026-07-21 11:00` (**859h / 35.8 days**), `2026-08-21 04:00` → `2026-09-12 06:00` (**531h / 22.1 days**), and `2025-07-23 17:00` → `2025-08-08 08:00` (**376h / 15.7 days**), plus 57 `NaN` records.
   - **CMT8 (`3276359` - HCMC):** 5 outages `>= 24h`; longest intervals: `2025-06-16 09:00` → `2025-07-15 02:00` (**690h / 28.8 days**) and `2025-05-03 19:00` → `2025-05-14 02:00` (**248h / 10.3 days**).
   - **Số 1 Giải Phóng (`4946813`):** 8 outages `>= 24h`; longest interval: `2025-07-19 10:00` → `2025-07-25 01:00` (**136h / 5.7 days**).
   - **Dragon (`6123215`):** 2 outages `>= 24h`; longest interval: `2026-04-02 03:00` → `2026-04-07 01:00` (**119h / 5.0 days**).
   - **Công viên Nhân Chính (`4946812`):** 4 outages `>= 24h`; longest interval: `2026-07-15 11:00` → `2026-07-19 00:00` (**86h / 3.6 days**).
2. **Open-Meteo Meteorological Data (5 Stations, `2024-11-14` → `2026-11-13`):**
   - Achieves **17,520 / 17,520 hours (100% completeness, 0 `NaN`, 0 gaps)** at the exact `(latitude, longitude)` of all 5 stations across all required variables: `temperature_c`, `relative_humidity_pct`, `wind_speed_m_s` (in `m/s`), `wind_direction_deg`, and `precipitation_mm`.

---

## 5. 1-Week Sample Dataset in `data/sample/`

A **1-week sample (`2025-11-15T00:00:00Z` → `2025-11-21T23:00:00Z`, 168 hours/station)** for the **5 usable stations** (preserving natural OpenAQ missing hours) has been saved as **2 combined CSV files** in `data/sample/`:

**1. OpenAQ PM2.5 (`data/sample/openaq_pm25_hanoi_hcmc.csv` — 815 rows, 108 KB)**
- **[HCMC] CMT8 (`3276359`)** · `164 / 168` rows (4 missing hours)
- **[Hanoi] Công viên Nhân Chính (`4946812`)** · `168 / 168` rows (100% complete)
- **[Hanoi] Số 46 Lưu Quang Vũ (`2161292`)** · `157 / 168` rows (11 missing hours)
- **[Hanoi] Số 1 Giải Phóng (`4946813`)** · `162 / 168` rows (6 missing hours)
- **[Hanoi] Dragon (`6123215`)** · `164 / 168` rows (4 missing hours)

**2. Open-Meteo Weather (`data/sample/openmeteo_weather_hanoi_hcmc.csv` — 840 rows, 85 KB)**
- Contains all `168 / 168` hourly rows per station (`5 × 168 = 840` rows, 0 missing values) at the exact `(latitude, longitude)` of all 5 stations, following the `docs/CONTRACTS.md` column schema: `station_id`, `event_time`, `temperature_c`, `relative_humidity_pct`, `wind_speed_m_s`, `wind_direction_deg`, `precipitation_mm`.
