# Pac-Man AI

A deep reinforcement learning system that learns to play Pac-Man from scratch, then builds a neural network that learns to **dream** the game — training a second agent entirely inside imagined gameplay.

<div align="center">

```
 +==============================+
 | ....#.....##.....#....       |
 | .##.#.###.##.###.#.##.      |
 | @..........C..........@     |
 | .##.#.##########.#.##.      |
 | ....#....#GGGG#....#....    |
 | ########.#      #.########  |
 |         .# G  G #.          |
 | ########.########.########  |
 | ....#......................  |
 +==============================+
```

**79 tests** | **128 parallel envs** | **PPO + RND + World Model (RSSM)**

</div>

---

## Overview

This project has two major components:

1. **PPO Agent** -- A convolutional neural network trained with Proximal Policy Optimization across 128 parallel game engines, using curriculum learning and RND curiosity to achieve 95%+ level clear rate.

2. **Dreaming Pac-Man** -- A latent world model (RSSM) that learns to simulate the game from gameplay data, then trains a dream agent entirely in imagination with zero real environment interaction.

The entire game engine is built from scratch in NumPy. Training uses PyTorch with Apple Silicon (MPS), CUDA, or CPU acceleration.

---

## Architecture

### PPO Agent

```
Input: 32×31×28 grid (8ch × 4 frame stack) + 5 scalars
  |
  +-- Conv2d(32->64, 3x3, stride=1)  -> ReLU
  +-- Conv2d(64->128, 3x3, stride=2) -> ReLU
  +-- Conv2d(128->128, 3x3, stride=2) -> ReLU
  +-- Flatten
  |
  +-- Concat with 5 scalars
  +-- Linear(... -> 512) -> ReLU           <- shared backbone
  |
  +---> Linear(512 -> 256 -> 4)            <- policy head
  +---> Linear(512 -> 256 -> 1)            <- value head

Total: ~4.2M parameters
```

### World Model (RSSM)

```
Observation (8x31x28) --> CNN Encoder [64,128,256,256] --> Posterior z
                                                              |
                          +-----------------------------------+
                          v
                    GRU Dynamics (h=512) --> Prior z (for imagination)
                          |
                          +---> Transposed CNN Decoder --> Reconstructed obs
                          +---> Reward Head (MLP)     --> Predicted reward
                          +---> Continue Head (MLP)   --> Termination probability

Latent state: h (512) + z (32 classes x 64 categoricals = 2048) = 2560 dims
Total: ~28M parameters
```

### Dream Agent

```
Latent state (2560) --> MLP (512 -> 256 -> 4 actions)   <- actor
                    --> MLP (512 -> 256 -> 1 value)      <- critic
```

---

## Training Pipeline

### Phase 1: PPO Training (Real Environment)

The agent learns to play Pac-Man through three curriculum stages:

| Phase | Updates | Difficulty | What the Agent Learns |
|-------|---------|------------|----------------------|
| 1 | 0 - 800 | Scatter-only ghosts | Navigate maze, eat pellets |
| 2 | 800 - 3000 | Full ghost AI | Evade ghosts, use power pellets |
| 3 | 3000 - 8000 | Full AI + Cruise Elroy | Master ghost chains, optimize routes |

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Parallel environments | 128 |
| Rollout steps | 128 (= 16,384 transitions/rollout) |
| PPO epochs | 4 |
| Minibatch size | 2,048 |
| Learning rate | 2.5e-4 (annealed) |
| Entropy coefficient | 0.15 -> 0.01 (annealed) |
| Frame stacking | 4 frames |
| RND curiosity | Enabled (intrinsic_coef=0.1) |
| Total updates | 8,000 |

### Phase 2: World Model Training

The trained PPO agent collects 1000 episodes of gameplay. The RSSM world model learns to simulate the game by minimizing:

```
Loss = reconstruction + reward_prediction + continue_prediction + KL(posterior || prior)
```

- **Reconstruction**: MSE between decoded latent state and real observation
- **Reward**: MSE on symlog-transformed rewards
- **Continue**: Binary cross-entropy on episode termination
- **KL**: Keeps the prior (imagination) close to the posterior (reality), with free nats = 1.0

### Phase 3: Imagination Training

The dream agent trains entirely inside the world model's latent space:
1. Encode diverse starting observations into latent states
2. Roll out 512 parallel imagined trajectories for 15 steps each
3. Compute GAE advantages using predicted rewards and continuation probabilities
4. PPO update on the imagined data
5. Periodically evaluate in the real game to measure transfer

---

## Observation Space

The agent sees the game as a multi-channel image:

| Channel | Content | Purpose |
|---------|---------|---------|
| 0 | Walls | Navigate the maze |
| 1 | Pac-Man position | Self-localization |
| 2 | Pellets | Primary objective |
| 3 | Power pellets | Strategic power-ups |
| 4 | Dangerous ghosts | Threats to avoid |
| 5 | Edible ghosts | Targets to chase |
| 6 | Ghost house | Restricted zone |
| 7 | Fruit | Bonus points |

Plus **5 scalar features**: power timer, lives, ghosts eaten, progress, and current direction.

The PPO agent stacks 4 consecutive frames (32 channels) for temporal context. The world model uses single frames (8 channels) -- the GRU hidden state replaces frame stacking.

---

## Ghost AI

Each ghost follows authentic 1980 arcade behavior:

| Ghost | Name | Targeting Strategy |
|-------|------|--------------------|
| Blinky | Red | Directly targets Pac-Man's tile |
| Pinky | Pink | Targets 4 tiles ahead of Pac-Man |
| Inky | Cyan | Flanking maneuver using Blinky's position |
| Clyde | Orange | Chases when far (>8 tiles), retreats when close |

Ghosts cycle through **scatter** (patrol corners) and **chase** (hunt Pac-Man) modes, with **frightened** mode when Pac-Man eats a power pellet.

---

## Reward Design

| Event | Reward | Purpose |
|-------|--------|---------|
| Eat pellet | +1.0 | Core objective |
| Eat power pellet | +2.0 | Strategic value |
| Eat ghost (1st-4th) | +5 / +10 / +15 / +20 | Ghost chain bonus |
| Eat fruit | +3.0 | Bonus target |
| Clear level | +50.0 | Ultimate goal |
| Death | -10.0 | Avoid ghosts |
| Game over | -25.0 | Strong survival signal |
| Ghost proximity | -0.3 | Spatial awareness |
| Time step | -0.01 | Encourage efficiency |

---

## Quick Start

```bash
# Clone
git clone https://github.com/vikranthreddimasu/pacman-ai.git
cd pacman-ai

# Install
pip install -e ".[dev]"

# Train PPO agent (128 parallel games, auto-detects MPS/CUDA/CPU)
python scripts/train.py --total-updates 8000

# Watch the agent play
python scripts/watch.py runs/<run-dir>/checkpoints/latest.pt

# Run tests
pytest tests/ -v
```

### World Model Pipeline

```bash
# Collect gameplay data from trained agent (~35 min)
python scripts/collect_data.py --checkpoint runs/<run-dir>/checkpoints/latest.pt --episodes 1000

# Train world model (~7 hours on MPS)
python scripts/train_world_model.py --data runs/<run-dir>/replay_buffer.pt

# Train dream agent (~2 hours)
python scripts/train_dreamer.py --world-model runs/<run-dir>/world_model/world_model_latest.pt

# Watch real game vs model's dream side-by-side
python scripts/watch_dreams.py --world-model runs/<run-dir>/world_model/world_model_latest.pt
```

### Monitoring

```bash
# Live training dashboard (Dash + Plotly)
python scripts/dashboard.py --log-dir runs/<run-dir>/tensorboard/

# TensorBoard
tensorboard --logdir runs/
```

---

## Project Structure

```
pacman-ai/
├── pacman/
│   ├── agents/
│   │   ├── networks.py        # Actor-Critic CNN (~4.2M params)
│   │   ├── ppo.py             # PPO algorithm (clipped surrogate + GAE)
│   │   ├── rnd.py             # Random Network Distillation curiosity
│   │   └── rollout.py         # Experience buffer with batch generator
│   ├── config/
│   │   └── default.yaml       # All hyperparameters
│   ├── engine/
│   │   ├── constants.py       # Enums, directions, ghost properties
│   │   ├── entities.py        # GameState dataclass
│   │   ├── game.py            # Step logic, collision, rewards
│   │   ├── ghost_ai.py        # Blinky/Pinky/Inky/Clyde targeting
│   │   ├── maze.py            # Grid operations, BFS pathfinding
│   │   └── maze_data.py       # Classic arcade maze layout
│   ├── env/
│   │   ├── pacman_env.py      # Single-game Gymnasium interface
│   │   └── vec_env.py         # N parallel games with auto-reset + frame stacking
│   ├── training/
│   │   ├── trainer.py         # PPO training loop + curriculum
│   │   ├── evaluator.py       # Greedy policy evaluation
│   │   ├── checkpoint.py      # Save/load model state
│   │   ├── wm_trainer.py      # World model training loop
│   │   └── dream_trainer.py   # Imagination PPO training loop
│   ├── world_model/
│   │   ├── rssm.py            # GRU dynamics + categorical stochastic state
│   │   ├── encoder.py         # 4-layer CNN encoder (obs -> latent)
│   │   ├── decoder.py         # Transposed CNN decoder (latent -> obs)
│   │   ├── heads.py           # Reward and continue prediction heads
│   │   ├── world_model.py     # Integrated model: train_step() + imagine()
│   │   └── replay_buffer.py   # Sequential episode storage
│   ├── utils/
│   │   └── config.py          # YAML config loader
│   └── viz/
│       ├── renderer.py        # Pygame game renderer
│       ├── sprites.py         # Pac-Man and ghost sprites
│       ├── dashboard.py       # Live metrics dashboard (Dash + Plotly)
│       └── dream_viewer.py    # Side-by-side real vs dream visualization
├── scripts/
│   ├── train.py               # PPO training entry point
│   ├── evaluate.py            # Evaluate a checkpoint
│   ├── watch.py               # Watch agent play in real-time
│   ├── dashboard.py           # Launch metrics dashboard
│   ├── collect_data.py        # Collect gameplay data for world model
│   ├── train_world_model.py   # Train RSSM from collected data
│   ├── train_dreamer.py       # Train dream agent in imagination
│   └── watch_dreams.py        # Launch dream viewer
├── tests/                     # 79 tests
├── Dockerfile
└── pyproject.toml
```

---

## Key Design Decisions

**Why PPO over DQN?**
PPO handles continuous exploration better, is more stable with large batch sizes, and naturally supports the stochastic policies needed for high-dimensional observations.

**Why a custom engine?**
A pure NumPy engine runs 128 games in parallel without process overhead, achieving 2,800+ environment steps per second with full control over authentic ghost AI.

**Why curriculum learning?**
Without it, ghosts kill the agent before it learns to eat pellets. Scatter-only ghosts let the agent learn navigation first.

**Why frame stacking for PPO but not the world model?**
The PPO agent needs 4 stacked frames to perceive motion (ghost direction, speed). The world model's GRU hidden state serves the same purpose -- it maintains temporal memory across steps, so single frames suffice.

**Why categorical stochastic state?**
Pac-Man is a discrete game -- positions are on a grid, ghosts are in specific modes. Categorical distributions (32 classes x 64 categoricals) model this better than Gaussian alternatives, matching the Dreamer V3 approach.

---

## Research Context

The world model implementation draws from:
- **Dreamer V3** (Hafner et al., 2023) -- RSSM with categorical stochastic state
- **DIAMOND** (NeurIPS 2024) -- world models for game simulation
- **Genie 3** (DeepMind) -- imagination-based agent training

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Game Engine | NumPy (vectorized, 128 parallel games) |
| PPO Agent | PyTorch CNN Actor-Critic (~4.2M params) |
| Curiosity | Random Network Distillation (RND) |
| World Model | RSSM with categorical latent state (~28M params) |
| Acceleration | Apple MPS / CUDA / CPU |
| Visualization | Pygame + Dash/Plotly |
| Tracking | TensorBoard |
| Testing | pytest (79 tests) |

---

## License

MIT
