"""SQLite database schema definitions."""

EPISODES_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    winner TEXT,
    score INTEGER,
    steps INTEGER,
    pellets_eaten INTEGER,
    ghosts_eaten INTEGER,
    fruits_eaten INTEGER,
    lives_remaining INTEGER,
    level_cleared INTEGER
)
"""

AGENT_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    total_reward REAL,
    avg_q_value REAL,
    epsilon REAL,
    loss REAL,
    actions_taken INTEGER,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
)
"""

TRAINING_CONFIG_TABLE = """
CREATE TABLE IF NOT EXISTS training_config (
    key TEXT PRIMARY KEY,
    value TEXT
)
"""

ALL_TABLES = [EPISODES_TABLE, AGENT_METRICS_TABLE, TRAINING_CONFIG_TABLE]
