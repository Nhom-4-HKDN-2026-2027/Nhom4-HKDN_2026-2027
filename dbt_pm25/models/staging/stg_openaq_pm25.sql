SELECT
    CAST(TRIM(city) AS VARCHAR) AS city,
    CAST(TRIM(station_id) AS VARCHAR) AS station_id,
    CAST(TRIM(station_name) AS VARCHAR) AS station_name,

    TRY_CAST(event_time AS TIMESTAMPTZ) AS event_time,
    TRY_CAST(pm25_ug_m3 AS DOUBLE) AS pm25_ug_m3,
    TRY_CAST(percent_coverage AS DOUBLE) AS percent_coverage

FROM read_csv_auto(
    '../data/sample/openaq_pm25_hanoi_hcmc.csv',
    header = true,
    all_varchar = true
)

WHERE TRY_CAST(event_time AS TIMESTAMPTZ) IS NOT NULL
  AND TRY_CAST(pm25_ug_m3 AS DOUBLE) IS NOT NULL