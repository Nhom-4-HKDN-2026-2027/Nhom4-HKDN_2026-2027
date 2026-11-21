SELECT
    CAST(TRIM(city) AS VARCHAR) AS city,
    CAST(TRIM(station_id) AS VARCHAR) AS station_id,

    TRY_CAST(event_time AS TIMESTAMPTZ) AS event_time,

    TRY_CAST(temperature_c AS DOUBLE) AS temperature_c,

    TRY_CAST(relative_humidity_pct AS DOUBLE)
        AS relative_humidity_pct,

    TRY_CAST(wind_speed_m_s AS DOUBLE)
        AS wind_speed_m_s,

    TRY_CAST(wind_direction_deg AS DOUBLE)
        AS wind_direction_deg,

    TRY_CAST(precipitation_mm AS DOUBLE)
        AS precipitation_mm

FROM read_csv_auto(
    '../data/sample/openmeteo_weather_hanoi_hcmc.csv',
    header = true,
    all_varchar = true
)

WHERE TRY_CAST(event_time AS TIMESTAMPTZ) IS NOT NULL