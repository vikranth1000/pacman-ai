"""Query helpers for analysis and plotting from SQLite metrics database."""

import sqlite3
from pathlib import Path


class Analyzer:
    """Read-only query interface for metrics database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def score_over_time(self, limit: int = 5000) -> list[dict]:
        """Get score progression."""
        cursor = self.conn.execute(
            "SELECT episode_id, score, steps, winner FROM episodes ORDER BY episode_id LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def reward_curves(self, agent_name: str, limit: int = 5000) -> list[dict]:
        """Get reward curve for a specific agent."""
        cursor = self.conn.execute(
            """SELECT episode_id, total_reward, avg_q_value, epsilon, loss
               FROM agent_metrics WHERE agent_name = ? ORDER BY episode_id LIMIT ?""",
            (agent_name, limit)
        )
        return [dict(row) for row in cursor.fetchall()]

    def rolling_win_rate(self, window: int = 50) -> list[dict]:
        """Calculate rolling win rate over episodes."""
        cursor = self.conn.execute(
            "SELECT episode_id, winner FROM episodes ORDER BY episode_id"
        )
        rows = cursor.fetchall()
        results = []
        winners = []
        for row in rows:
            winners.append(row["winner"])
            if len(winners) > window:
                winners.pop(0)
            pac_rate = sum(1 for w in winners if w == "pacman") / len(winners)
            results.append({
                "episode_id": row["episode_id"],
                "pacman_win_rate": pac_rate,
                "ghost_win_rate": 1.0 - pac_rate,
            })
        return results

    def ghost_comparison(self, limit: int = 5000) -> dict[str, list[dict]]:
        """Get per-ghost reward curves for comparison."""
        ghost_names = ["blinky", "pinky", "inky", "clyde"]
        result = {}
        for name in ghost_names:
            result[name] = self.reward_curves(name, limit)
        return result

    def summary_stats(self) -> dict:
        """Get overall summary statistics."""
        cursor = self.conn.execute(
            """SELECT
                COUNT(*) as total_episodes,
                AVG(score) as avg_score,
                MAX(score) as max_score,
                AVG(steps) as avg_steps,
                AVG(pellets_eaten) as avg_pellets,
                AVG(ghosts_eaten) as avg_ghosts_eaten,
                SUM(CASE WHEN winner='pacman' THEN 1 ELSE 0 END) as pacman_wins,
                SUM(CASE WHEN winner='ghosts' THEN 1 ELSE 0 END) as ghost_wins
               FROM episodes"""
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def close(self):
        self.conn.close()
