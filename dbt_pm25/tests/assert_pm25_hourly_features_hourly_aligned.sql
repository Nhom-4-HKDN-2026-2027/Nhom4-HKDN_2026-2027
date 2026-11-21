-- Ensure event_time is aligned to whole hours per docs/CONTRACTS.md Section 1.2 & 1.3.

SELECT
    station_id,
    event_time
FROM {{ ref('pm25_hourly_features') }}
WHERE (event_time AT TIME ZONE 'UTC') != DATE_TRUNC('hour', event_time AT TIME ZONE 'UTC')
