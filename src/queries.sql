-- Feature engineering via temporary SQL views.
-- These views are connection-scoped: they persist for the lifetime of the
-- SQLite connection and are automatically dropped when the connection closes.

-- user_stats: per-user aggregate behavioural features
CREATE TEMP VIEW user_stats AS
SELECT
    user_id,
    COUNT(*)      AS tx_count,
    AVG(amount)   AS avg_amount,
    SUM(amount)   AS total_amount
FROM transactions
GROUP BY user_id;

-- daily_user: per-user, per-day activity features
CREATE TEMP VIEW daily_user AS
SELECT
    user_id,
    date,
    COUNT(*)    AS daily_tx,
    SUM(amount) AS daily_amount
FROM transactions
GROUP BY user_id, date;

-- Final SELECT: one row per transaction, joining in both feature sets.
-- COALESCE guards against NULLs if a join ever returns no match.
SELECT
    t.tx_id,
    t.user_id,
    t.date,
    t.region,
    t.merchant,
    t.amount,
    us.tx_count,
    us.avg_amount,
    us.total_amount,
    COALESCE(du.daily_tx,     0)   AS daily_tx,
    COALESCE(du.daily_amount, 0.0) AS daily_amount
FROM transactions t
LEFT JOIN user_stats us ON t.user_id = us.user_id
LEFT JOIN daily_user du ON t.user_id = du.user_id AND t.date = du.date