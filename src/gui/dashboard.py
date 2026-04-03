"""Web metrics dashboard using Dash + Plotly, auto-refreshing from SQLite."""

from pathlib import Path
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from src.data.analyzer import Analyzer


def create_dashboard(db_path: str | Path) -> Dash:
    """Create and configure the Dash web dashboard."""
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Pac-Man AI Training Dashboard",
                style={"textAlign": "center", "color": "#FFD700", "backgroundColor": "#1a1a2e",
                       "padding": "20px", "margin": "0"}),

        dcc.Interval(id="refresh", interval=5000, n_intervals=0),

        html.Div([
            # Summary stats
            html.Div(id="summary-stats",
                     style={"padding": "10px", "backgroundColor": "#16213e",
                            "borderRadius": "8px", "margin": "10px"}),

            # Win rate chart
            html.Div([
                dcc.Graph(id="win-rate-chart"),
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

            # Score chart
            html.Div([
                dcc.Graph(id="score-chart"),
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

            # Per-agent reward chart
            html.Div([
                dcc.Graph(id="reward-chart"),
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

            # Epsilon chart
            html.Div([
                dcc.Graph(id="epsilon-chart"),
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

            # Ghost comparison
            html.Div([
                dcc.Graph(id="ghost-comparison-chart"),
            ], style={"width": "98%", "display": "inline-block"}),

        ], style={"backgroundColor": "#0a0a1a", "minHeight": "100vh", "padding": "10px"}),
    ], style={"backgroundColor": "#0a0a1a"})

    @app.callback(
        [Output("summary-stats", "children"),
         Output("win-rate-chart", "figure"),
         Output("score-chart", "figure"),
         Output("reward-chart", "figure"),
         Output("epsilon-chart", "figure"),
         Output("ghost-comparison-chart", "figure")],
        [Input("refresh", "n_intervals")]
    )
    def update_charts(_):
        try:
            analyzer = Analyzer(db_path)
        except FileNotFoundError:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            return "Waiting for data...", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

        try:
            stats = analyzer.summary_stats()
            summary = html.Div([
                html.Span(f"Episodes: {stats.get('total_episodes', 0)}  |  ",
                          style={"color": "#fff", "fontSize": "16px"}),
                html.Span(f"Avg Score: {stats.get('avg_score', 0):.0f}  |  ",
                          style={"color": "#FFD700", "fontSize": "16px"}),
                html.Span(f"Max Score: {stats.get('max_score', 0)}  |  ",
                          style={"color": "#4caf50", "fontSize": "16px"}),
                html.Span(f"Pac-Man Wins: {stats.get('pacman_wins', 0)}  |  ",
                          style={"color": "#FFD700", "fontSize": "16px"}),
                html.Span(f"Ghost Wins: {stats.get('ghost_wins', 0)}",
                          style={"color": "#ff6b6b", "fontSize": "16px"}),
            ], style={"textAlign": "center"})

            # Win rate
            wr_data = analyzer.rolling_win_rate(window=50)
            win_fig = go.Figure()
            if wr_data:
                eps = [d["episode_id"] for d in wr_data]
                win_fig.add_trace(go.Scatter(x=eps, y=[d["pacman_win_rate"] for d in wr_data],
                                             name="Pac-Man", line=dict(color="#FFD700")))
                win_fig.add_trace(go.Scatter(x=eps, y=[d["ghost_win_rate"] for d in wr_data],
                                             name="Ghosts", line=dict(color="#ff6b6b")))
            win_fig.update_layout(title="Win Rate (rolling 50)", template="plotly_dark",
                                  xaxis_title="Episode", yaxis_title="Win Rate",
                                  yaxis=dict(range=[0, 1]))

            # Score
            score_data = analyzer.score_over_time()
            score_fig = go.Figure()
            if score_data:
                eps = [d["episode_id"] for d in score_data]
                scores = [d["score"] for d in score_data]
                score_fig.add_trace(go.Scatter(x=eps, y=scores, name="Score",
                                               line=dict(color="#4caf50"), opacity=0.3))
                # Rolling average
                window = 50
                if len(scores) >= window:
                    rolling = [sum(scores[max(0, i - window):i]) / min(i, window)
                               for i in range(1, len(scores) + 1)]
                    score_fig.add_trace(go.Scatter(x=eps, y=rolling, name=f"Avg ({window})",
                                                    line=dict(color="#4caf50", width=2)))
            score_fig.update_layout(title="Score Over Time", template="plotly_dark",
                                    xaxis_title="Episode", yaxis_title="Score")

            # Per-agent rewards
            reward_fig = go.Figure()
            colors = {"pacman": "#FFD700", "blinky": "#ff0000", "pinky": "#ffb8ff",
                       "inky": "#00ffff", "clyde": "#ffb852"}
            for agent_name, color in colors.items():
                data = analyzer.reward_curves(agent_name)
                if data:
                    eps = [d["episode_id"] for d in data]
                    rewards = [d["total_reward"] for d in data]
                    # Rolling average
                    window = 50
                    if len(rewards) >= window:
                        rolling = [sum(rewards[max(0, i - window):i]) / min(i, window)
                                   for i in range(1, len(rewards) + 1)]
                        reward_fig.add_trace(go.Scatter(x=eps, y=rolling, name=agent_name,
                                                         line=dict(color=color)))
            reward_fig.update_layout(title="Agent Rewards (rolling 50)", template="plotly_dark",
                                     xaxis_title="Episode", yaxis_title="Total Reward")

            # Epsilon
            eps_data = analyzer.reward_curves("pacman")
            eps_fig = go.Figure()
            if eps_data:
                eps = [d["episode_id"] for d in eps_data]
                epsilons = [d["epsilon"] for d in eps_data]
                eps_fig.add_trace(go.Scatter(x=eps, y=epsilons, name="Epsilon",
                                             line=dict(color="#64ffda")))
            eps_fig.update_layout(title="Exploration Rate (ε)", template="plotly_dark",
                                  xaxis_title="Episode", yaxis_title="Epsilon")

            # Ghost comparison
            ghost_fig = go.Figure()
            ghost_data = analyzer.ghost_comparison()
            ghost_colors = {"blinky": "#ff0000", "pinky": "#ffb8ff",
                           "inky": "#00ffff", "clyde": "#ffb852"}
            for name, data in ghost_data.items():
                if data:
                    eps = [d["episode_id"] for d in data]
                    rewards = [d["total_reward"] for d in data]
                    window = 50
                    if len(rewards) >= window:
                        rolling = [sum(rewards[max(0, i - window):i]) / min(i, window)
                                   for i in range(1, len(rewards) + 1)]
                        ghost_fig.add_trace(go.Scatter(x=eps, y=rolling, name=name,
                                                        line=dict(color=ghost_colors[name])))
            ghost_fig.update_layout(title="Ghost Performance Comparison (rolling 50)",
                                    template="plotly_dark",
                                    xaxis_title="Episode", yaxis_title="Total Reward")

            analyzer.close()
            return summary, win_fig, score_fig, reward_fig, eps_fig, ghost_fig

        except Exception as e:
            analyzer.close()
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark",
                                    annotations=[dict(text=str(e), showarrow=False)])
            return f"Error: {e}", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    return app
