-- Ensure composite primary key (station_id, event_time) has no duplicates
-- per docs/CONTRACTS.md Section 1.3.

SELECT
    station_id,
    event_time,
    COUNT(*) AS record_count
FROM {{ ref('pm25_hourly_features') }}
GROUP BY station_id, event_time
HAVING COUNT(*) > 1
