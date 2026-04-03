# pacman/training/trainer.py
"""PPO training orchestrator."""
import time
from pathlib import Path

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


class _CSVWriter:
    """Fallback metrics logger when TensorBoard is unavailable."""
    def __init__(self, log_dir: str):
        self._path = Path(log_dir) / "metrics.csv"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: list[str] = []
        self._file = None

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        if self._file is None:
            self._file = open(self._path, "w")
            self._file.write("step,tag,value\n")
        self._file.write(f"{global_step},{tag},{value}\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

from ..env.vec_env import VecEnv
from ..agents.networks import ActorCritic
from ..agents.ppo import PPO
from ..agents.rollout import RolloutBuffer
from ..training.evaluator import Evaluator
from ..training.checkpoint import save_checkpoint, load_checkpoint


class RunningMeanStd:
    """Tracks running mean/std for reward normalization."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        total = self.count + count
        new_mean = self.mean + delta * count / total
        m_a = self.var * self.count
        m_b = var * count
        m2 = m_a + m_b + delta ** 2 * self.count * count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / (np.sqrt(self.var) + 1e-8),
            -self._clip, self._clip,
        )

    @property
    def _clip(self):
        return 10.0

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


def get_device(config: dict) -> torch.device:
    choice = config.get("device", "auto")
    if choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(choice)


class Trainer:
    def __init__(self, config: dict, run_dir: Path, resume: bool = False):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device(config)

        # Environment
        env_cfg = config["env"]
        self.vec_env = VecEnv(env_cfg["num_envs"], config, difficulty=0)

        # Network
        net_cfg = config["network"]
        self.network = ActorCritic(
            cnn_channels=net_cfg["cnn_channels"],
            cnn_kernels=net_cfg["cnn_kernels"],
            cnn_strides=net_cfg["cnn_strides"],
            shared_hidden=net_cfg["shared_hidden"],
            head_hidden=net_cfg["head_hidden"],
        )

        # PPO
        self.ppo = PPO(self.network, config, self.device)

        # Rollout buffer
        self.rollout = RolloutBuffer(
            num_envs=env_cfg["num_envs"],
            rollout_steps=config["ppo"]["rollout_steps"],
            grid_shape=(env_cfg["observation_channels"], 31, 28),
            num_scalars=env_cfg["num_scalar_features"],
            num_actions=4,
        )

        # Reward normalization
        self.reward_normalizer = RunningMeanStd()

        # Evaluator
        self.evaluator = Evaluator(config)

        # Logging
        tb_dir = str(self.run_dir / "tensorboard")
        if _HAS_TB:
            self.writer = SummaryWriter(tb_dir)
        else:
            print("[WARN] TensorBoard unavailable — logging metrics to CSV")
            self.writer = _CSVWriter(tb_dir)

        # Curriculum
        self.curriculum_phase = 0
        self.best_clear_rate = 0.0
        self.start_update = 0

        if resume:
            ckpt_dir = self.run_dir / "checkpoints"
            if (ckpt_dir / "latest.pt").exists():
                meta = load_checkpoint(ckpt_dir, self.network, self.ppo.optimizer)
                self.start_update = meta["update"] + 1
                self.curriculum_phase = meta.get("curriculum_phase", 0)
                norm_state = meta.get("reward_normalizer", {})
                if norm_state:
                    self.reward_normalizer.load_state_dict(norm_state)

    def train(self, total_updates: int | None = None) -> None:
        total = total_updates or self.config["training"]["total_updates"]
        train_cfg = self.config["training"]
        ppo_cfg = self.config["ppo"]
        curriculum_cfg = self.config["curriculum"]
        N = self.config["env"]["num_envs"]
        T = ppo_cfg["rollout_steps"]

        obs = self.vec_env.reset(seed=42)

        for update in range(self.start_update, total):
            t_start = time.time()

            # --- Anneal schedules ---
            self.ppo.anneal_lr(update, total)
            entropy_coef = self._get_entropy_coef(update, total)
            self.ppo.set_entropy_coef(entropy_coef)

            # --- Advance curriculum ---
            self._advance_curriculum(update, curriculum_cfg)

            # --- Collect rollout ---
            self.rollout.reset()
            for _step in range(T):
                legal_masks = self.vec_env.get_legal_masks()
                actions, log_probs, values = self.ppo.select_action(
                    obs["grid"], obs["scalars"], legal_masks,
                )
                next_obs, rewards, dones, infos = self.vec_env.step(actions)

                # Normalize rewards
                self.reward_normalizer.update(rewards)
                norm_rewards = self.reward_normalizer.normalize(rewards)

                self.rollout.insert(
                    obs["grid"], obs["scalars"],
                    actions, log_probs, values,
                    norm_rewards, dones, legal_masks,
                )
                obs = next_obs

            # --- Compute GAE ---
            legal_masks = self.vec_env.get_legal_masks()
            last_values = self.ppo.get_value(obs["grid"], obs["scalars"], legal_masks)
            self.rollout.compute_gae(
                last_values, ppo_cfg["gamma"], ppo_cfg["gae_lambda"],
            )

            # --- PPO Update ---
            metrics = self.ppo.update(self.rollout)

            # --- Logging ---
            fps = N * T / (time.time() - t_start)
            if update % train_cfg["log_every"] == 0:
                self.writer.add_scalar("loss/policy", metrics["policy_loss"], update)
                self.writer.add_scalar("loss/value", metrics["value_loss"], update)
                self.writer.add_scalar("loss/entropy", metrics["entropy"], update)
                self.writer.add_scalar("schedule/learning_rate",
                                       self.ppo.optimizer.param_groups[0]["lr"], update)
                self.writer.add_scalar("schedule/entropy_coef", entropy_coef, update)
                self.writer.add_scalar("schedule/curriculum_phase",
                                       self.curriculum_phase, update)
                self.writer.add_scalar("throughput/fps", fps, update)

            # --- Evaluation ---
            if update % train_cfg["eval_every"] == 0 and update > 0:
                eval_result = self.evaluator.evaluate(
                    self.network, train_cfg["eval_episodes"], self.device,
                )
                self.writer.add_scalar("performance/level_clear_rate",
                                       eval_result["level_clear_rate"], update)
                self.writer.add_scalar("performance/mean_score",
                                       eval_result["mean_score"], update)
                self.writer.add_scalar("performance/mean_steps",
                                       eval_result["mean_steps"], update)
                is_best = eval_result["level_clear_rate"] > self.best_clear_rate
                if is_best:
                    self.best_clear_rate = eval_result["level_clear_rate"]
                print(f"[Update {update}] clear={eval_result['level_clear_rate']:.1%} "
                      f"score={eval_result['mean_score']:.0f} fps={fps:.0f}")

            # --- Checkpoint ---
            if update % train_cfg["checkpoint_every"] == 0 and update > 0:
                is_best = False  # already handled above
                save_checkpoint(
                    self.run_dir / "checkpoints", update,
                    self.network, self.ppo.optimizer,
                    self.reward_normalizer.state_dict(),
                    self.curriculum_phase, self.config,
                    is_best=False,
                )

        # Final checkpoint
        save_checkpoint(
            self.run_dir / "checkpoints", total - 1,
            self.network, self.ppo.optimizer,
            self.reward_normalizer.state_dict(),
            self.curriculum_phase, self.config,
            is_best=False,
        )
        self.writer.close()

    def _get_entropy_coef(self, update: int, total: int) -> float:
        cfg = self.config["ppo"]
        start = cfg["entropy_coef_start"]
        end = cfg["entropy_coef_end"]
        anneal_frac = cfg["entropy_anneal_fraction"]
        if update < total * anneal_frac:
            return start
        progress = (update - total * anneal_frac) / (total * (1 - anneal_frac))
        return start + (end - start) * min(progress, 1.0)

    def _advance_curriculum(self, update: int, curriculum_cfg: dict) -> None:
        thresholds = curriculum_cfg["phase_thresholds"]
        difficulties = curriculum_cfg["difficulties"]
        # Find the highest phase whose threshold has been reached
        target_phase = 0
        for phase, threshold in enumerate(thresholds):
            if update >= threshold:
                target_phase = phase
        if target_phase != self.curriculum_phase:
            self.curriculum_phase = target_phase
            self.vec_env.set_difficulty(difficulties[target_phase])
            print(f"[Update {update}] Curriculum -> phase {target_phase} "
                  f"(difficulty={difficulties[target_phase]})")
