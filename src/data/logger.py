"""Data logger — SQLite writer for per-episode and per-agent metrics."""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.schemas import ALL_TABLES


class DataLogger:
    """Logs training metrics to SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        for table_sql in ALL_TABLES:
            cursor.execute(table_sql)
        self.conn.commit()

    def log_episode(self, episode_id: int, winner: str | None, score: int,
                    steps: int, pellets_eaten: int, ghosts_eaten: int,
                    fruits_eaten: int, lives_remaining: int, level_cleared: bool):
        """Log a completed episode."""
        self.conn.execute(
            """INSERT INTO episodes
               (episode_id, timestamp, winner, score, steps, pellets_eaten,
                ghosts_eaten, fruits_eaten, lives_remaining, level_cleared)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (episode_id, datetime.now(timezone.utc).isoformat(), winner, score, steps,
             pellets_eaten, ghosts_eaten, fruits_eaten, lives_remaining,
             1 if level_cleared else 0)
        )
        self.conn.commit()

    def log_agent_metrics(self, episode_id: int, agent_name: str,
                          total_reward: float, avg_q_value: float,
                          epsilon: float, loss: float | None,
                          actions_taken: int):
        """Log per-agent metrics for an episode."""
        self.conn.execute(
            """INSERT INTO agent_metrics
               (episode_id, agent_name, total_reward, avg_q_value, epsilon, loss, actions_taken)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (episode_id, agent_name, total_reward, avg_q_value, epsilon,
             loss if loss is not None else 0.0, actions_taken)
        )
        self.conn.commit()

    def log_config(self, config: dict):
        """Log training configuration as key-value pairs."""
        flat = self._flatten_dict(config)
        for key, value in flat.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO training_config (key, value) VALUES (?, ?)",
                (key, json.dumps(value))
            )
        self.conn.commit()

    def _flatten_dict(self, d: dict, prefix: str = "") -> dict:
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, key))
            else:
                items[key] = v
        return items

    def get_episodes(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Query recent episodes."""
        cursor = self.conn.execute(
            "SELECT * FROM episodes ORDER BY episode_id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_agent_metrics(self, agent_name: str, limit: int = 100) -> list[dict]:
        """Query recent metrics for a specific agent."""
        cursor = self.conn.execute(
            """SELECT * FROM agent_metrics WHERE agent_name = ?
               ORDER BY episode_id DESC LIMIT ?""",
            (agent_name, limit)
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_win_rates(self, window: int = 100) -> dict:
        """Calculate rolling win rates."""
        cursor = self.conn.execute(
            "SELECT winner FROM episodes ORDER BY episode_id DESC LIMIT ?",
            (window,)
        )
        winners = [row[0] for row in cursor.fetchall()]
        total = len(winners)
        if total == 0:
            return {"pacman": 0.0, "ghosts": 0.0, "total": 0}
        pac_wins = sum(1 for w in winners if w == "pacman")
        ghost_wins = sum(1 for w in winners if w == "ghosts")
        return {
            "pacman": pac_wins / total,
            "ghosts": ghost_wins / total,
            "total": total,
        }

    def close(self):
        self.conn.close()
