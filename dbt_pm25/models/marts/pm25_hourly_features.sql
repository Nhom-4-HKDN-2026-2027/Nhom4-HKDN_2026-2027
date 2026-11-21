{{ config(materialized='table') }}

WITH base AS (

    SELECT *
    FROM {{ ref('int_pm25_weather_hourly') }}

),

rolling_features AS (

    SELECT
        *,

        AVG(pm25_ug_m3) OVER (
            PARTITION BY city, station_id
            ORDER BY event_time
            RANGE BETWEEN INTERVAL 3 HOURS PRECEDING
            AND INTERVAL 1 HOUR PRECEDING
        ) AS pm25_rolling_mean_3h,

        AVG(pm25_ug_m3) OVER (
            PARTITION BY city, station_id
            ORDER BY event_time
            RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
            AND INTERVAL 1 HOUR PRECEDING
        ) AS pm25_rolling_mean_24h

    FROM base

),

lag_features AS (

    SELECT
        b.*,

        p1.pm25_ug_m3 AS pm25_lag_1h,
        p3.pm25_ug_m3 AS pm25_lag_3h,
        p24.pm25_ug_m3 AS pm25_lag_24h

    FROM rolling_features AS b

    LEFT JOIN base AS p1
        ON b.city = p1.city
        AND b.station_id = p1.station_id
        AND p1.event_time = b.event_time - INTERVAL 1 HOUR

    LEFT JOIN base AS p3
        ON b.city = p3.city
        AND b.station_id = p3.station_id
        AND p3.event_time = b.event_time - INTERVAL 3 HOURS

    LEFT JOIN base AS p24
        ON b.city = p24.city
        AND b.station_id = p24.station_id
        AND p24.event_time = b.event_time - INTERVAL 24 HOURS

),

local_features AS (

    SELECT
        *,
        event_time AT TIME ZONE 'Asia/Ho_Chi_Minh'
            AS local_event_time

    FROM lag_features

)

SELECT
    city,
    station_id,
    event_time,

    pm25_ug_m3,

    pm25_lag_1h,
    pm25_lag_3h,
    pm25_lag_24h,

    pm25_rolling_mean_3h,
    pm25_rolling_mean_24h,

    temperature_c,
    relative_humidity_pct,
    wind_speed_m_s,
    wind_direction_deg,
    precipitation_mm,

    CAST(EXTRACT(HOUR FROM local_event_time) AS TINYINT)
        AS hour,

    CAST(EXTRACT(ISODOW FROM local_event_time) AS TINYINT)
        AS weekday,

    EXTRACT(ISODOW FROM local_event_time) IN (6, 7)
        AS is_weekend,

    EXISTS (
    SELECT 1
    FROM {{ ref('vn_holidays') }} AS h
    WHERE h.holiday_date = CAST(f.local_event_time AS DATE)
) AS is_holiday

FROM local_features AS f