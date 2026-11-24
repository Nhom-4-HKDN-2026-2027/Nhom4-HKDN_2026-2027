SELECT
    p.city,
    p.station_id,
    p.event_time,
    p.pm25_ug_m3,

    w.temperature_c,
    w.relative_humidity_pct,
    w.wind_speed_m_s,
    w.wind_direction_deg,
    w.precipitation_mm

FROM {{ ref('stg_openaq_pm25') }} AS p

LEFT JOIN {{ ref('stg_openmeteo_weather') }} AS w

    ON p.city = w.city
    AND p.station_id = w.station_id
    AND p.event_time = w.event_time