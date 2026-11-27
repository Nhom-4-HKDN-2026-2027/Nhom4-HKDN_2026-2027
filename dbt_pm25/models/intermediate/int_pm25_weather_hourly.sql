
SELECT
    city,
    station_id,
    event_time,
    pm25_ug_m3,

    temperature_c,
    relative_humidity_pct,
    wind_speed_m_s,
    wind_direction_deg,
    precipitation_mm

FROM {{ ref('stg_beijing_pm25_weather') }}
