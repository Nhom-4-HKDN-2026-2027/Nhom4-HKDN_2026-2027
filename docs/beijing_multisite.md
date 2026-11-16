# Beijing Multi-Site sample: source and contract mapping

## Dataset identity and attribution

Use **UCI dataset 501, Beijing Multi-Site Air Quality**, DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), licensed under **CC BY 4.0**. Attribution: Chen, S. (2017), Beijing Multi-Site Air Quality, UCI Machine Learning Repository. The [official dataset page](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) describes 12 sites and hourly observations from March 2013 through February 2017, including missing values. Air-quality and weather observations come from different monitoring networks; weather is matched to nearby stations.

The official download contains `PRSA2017_Data_20130301-20170228.zip` plus unrelated files. Ingestion reads only the 12 `PRSA_Data_<station>_20130301-20170228.csv` members of that nested archive. It does **not** use the older single-site Beijing PM2.5 dataset (UCI 381, 2010–2014). The name `PRSA` in a filename alone does not distinguish these datasets.

## Reproduce the sample

After installing the environment in [ingest_setup_vn.md](ingest_setup_vn.md), run from the repository root:

```sh
.venv/py311/bin/python -m ingest.beijing_fetcher --output data/raw/beijing_multisite/reproduced_features.csv --metadata data/raw/beijing_multisite/reproduced_manifest.json
```

No API key is required. The CLI downloads the public archive to `data/raw/beijing_multisite/uci_501.zip` if absent, otherwise reuses it. Outputs must not already exist. Defaults select `[2015-11-15T00:00:00Z, 2015-11-22T00:00:00Z)` and all 12 sites. Pass `--archive`, `--start`, `--end`, `--source-timezone`, `--output` and `--metadata` to select another source file or seven-day window. Calendar bounds must be timezone-aware whole hours; exactly seven days are required for this sample workflow.

The handoff file is [data/sample/beijing_multisite_features.csv](../data/sample/beijing_multisite_features.csv). Only this small derived CSV and ingestion code/docs/tests belong in the PR; do not commit the complete archive. Local metadata records source/member hashes, row counts, conversions, assumptions, missing-value counts and sample checksum. Run the CLI against the same source to regenerate it; no shared contract file was changed.

## Contract mapping

The CSV uses exactly the seven feature columns in `docs/CONTRACTS.md` §3.2:

| Contract column | Source / transformation | Unit |
|---|---|---|
| `station_id` | `beijing_` + original station name | String |
| `event_time` | `year/month/day/hour`, interpreted in source timezone, converted to UTC | ISO 8601 with `Z` |
| `pm25_ug_m3` | `PM2.5` | µg/m³ |
| `temperature_c` | `TEMP` | °C |
| `relative_humidity_pct` | Derived from `TEMP` and `DEWP` | % |
| `wind_speed_m_s` | `WSPM`, unchanged | m/s |
| `wind_direction_deg` | `wd`, 16-point compass mapping | Degrees clockwise from north |

**Timezone assumption:** the UCI page does not explicitly state a timezone. The default interprets calendar fields as local Beijing time, `Asia/Shanghai` (UTC+8 for the dataset period), then converts to UTC. Thus source `2015-11-15 08:00` becomes `2015-11-15T00:00:00Z`. The default UTC sample corresponds to local `[2015-11-15 08:00, 2015-11-22 08:00)`. This assumption is recorded in metadata and must be confirmed before cross-source temporal comparisons; use `--source-timezone` if authoritative clarification differs.

Humidity uses the liquid-water Magnus approximation, as used in [this AMS research article, Eq. 5](https://journals.ametsoc.org/view/journals/apme/61/8/JAMC-D-21-0091.1.xml):

```text
RH = 100 × exp(17.625 × Td / (243.04 + Td) − 17.625 × T / (243.04 + T))
```

`T=TEMP` and `Td=DEWP` are in Celsius. This is a derived approximation, not directly observed RH or an ice-relative calculation below freezing. Missing/nonfinite inputs, `Td > T`, singular/nonpositive denominators or invalid results produce NaN; humidity is not clipped or imputed. The raw inputs remain in the source archive. The formula is intended for meteorological temperatures, not arbitrary extreme values.

Wind uses the standard [16-point compass bearings](https://www.weather.gov/media/mqt/FireWx/MQT%20Fire%20Wx%20AOP%202020%20-%20Final%20Version.pdf): N=0, NNE=22.5, NE=45, …, NNW=337.5. These represent the direction **from which** wind blows. Missing directions stay NaN; unknown nonmissing tokens raise an error. `RAIN` is not added because it is not in the seven-column feature contract; raw precipitation remains available in the archive.

## Measured sample quality

The generated sample contains **2,016 rows (12 × 168 hours)**, unique station/time keys and a complete timestamp grid per station. It retains **26 missing PM2.5 values** and **14 missing wind directions**; temperature, derived RH and wind speed have no missing values in this week. There are no negative PM2.5 or wind-speed values in the sample. Invalid/negative source measurements are not silently dropped or corrected by normalization. Duplicate keys, malformed timestamps, unknown direction tokens and missing source-hour rows cause explicit errors.

These figures describe this week, not quality of the full experiment dataset. No cross-station filling, train/test split or model evaluation is performed. The existing Vietnam sample schemas and CSVs remain unchanged; both sources can supply the shared feature names, with source-specific ingestion before downstream processing.

## Shared downstream adapter

CSV timestamp values are serialized as ISO 8601 strings; they become UTC timestamps when loaded for downstream processing. Vietnam's existing split CSVs retain numeric OpenAQ IDs for compatibility. `ingest/feature_schema.py` provides the common in-memory interface:

```python
import pandas as pd
from ingest.feature_schema import build_vietnam_features, normalize_feature_table

vn_features = build_vietnam_features(
    pd.read_csv("data/sample/openaq_pm25_hanoi_hcmc.csv"),
    pd.read_csv("data/sample/openmeteo_weather_hanoi_hcmc.csv"),
)
beijing_features = normalize_feature_table(
    pd.read_csv("data/sample/beijing_multisite_features.csv")
)
```

Both return exactly seven columns: `station_id` as pandas string, `event_time` as timezone-aware UTC datetime, and five float columns. The Vietnam adapter performs a one-to-one left join on normalized station/time keys. Missing weather remains NaN with a warning; PM2.5 rows are preserved. Naive/invalid/non-hourly timestamps, missing IDs, duplicate keys, nonnumeric values and infinities are rejected. Numeric NaN and finite source values, including negatives, remain available for quality assessment; this adapter does not define experiment acceptance or imputation policies. Contract names and units are unchanged.

Verified source archive SHA-256:
`b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`

Handoff CSV SHA-256:
`525ba52fbde3001bd2f80f901911f77dd3d003f32a8d8a0d40698301545ed162`
