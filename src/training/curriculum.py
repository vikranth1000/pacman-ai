"""Curriculum learning — asymmetric training schedules for Pac-Man vs ghosts."""


class CurriculumScheduler:
    """Manages asymmetric difficulty scaling between Pac-Man and ghosts.

    Ghosts start handicapped (high epsilon, low learning rate) and gradually
    ramp up to full strength, giving Pac-Man a developmental window to learn
    pellet-eating fundamentals before facing skilled opponents.
    """

    def __init__(self, config: dict):
        cur = config.get("curriculum", {})
        self.enabled = cur.get("enabled", True)

        # Pac-Man schedule
        agent_cfg = config.get("agent", {})
        self.pac_eps_start = agent_cfg.get("epsilon_start", 1.0)
        self.pac_eps_end = agent_cfg.get("epsilon_end", 0.08)
        self.pac_eps_decay = agent_cfg.get("epsilon_decay_episodes", 2000)

        # Ghost schedule (slower decay, optional head start for Pac-Man)
        self.ghost_eps_start = cur.get("ghost_epsilon_start", 1.0)
        self.ghost_eps_end = cur.get("ghost_epsilon_end", 0.12)
        self.ghost_eps_decay = cur.get("ghost_epsilon_decay_episodes", 3000)
        self.head_start = cur.get("pacman_head_start_episodes", 300)

        # Ghost learning rate ramp
        self.ghost_lr_mult_start = cur.get("ghost_lr_multiplier_start", 0.3)
        self.ghost_lr_mult_end = cur.get("ghost_lr_multiplier_end", 1.0)
        self.ghost_lr_ramp = cur.get("ghost_lr_ramp_episodes", 2000)

        self.base_ghost_lr = agent_cfg.get("learning_rate", 0.0002)

    def get_pacman_epsilon(self, episode: int) -> float:
        if not self.enabled:
            return self._linear_decay(episode, self.pac_eps_start, self.pac_eps_end, self.pac_eps_decay)
        return self._linear_decay(episode, self.pac_eps_start, self.pac_eps_end, self.pac_eps_decay)

    def get_ghost_epsilon(self, episode: int) -> float:
        if not self.enabled:
            return self._linear_decay(episode, self.pac_eps_start, self.pac_eps_end, self.pac_eps_decay)
        # During head start, ghosts are fully random
        if episode < self.head_start:
            return 1.0
        adjusted = episode - self.head_start
        return self._linear_decay(adjusted, self.ghost_eps_start, self.ghost_eps_end, self.ghost_eps_decay)

    def get_ghost_learning_rate(self, episode: int) -> float:
        if not self.enabled:
            return self.base_ghost_lr
        if episode < self.head_start:
            return self.base_ghost_lr * self.ghost_lr_mult_start
        adjusted = episode - self.head_start
        mult = self._linear_decay(
            adjusted,
            self.ghost_lr_mult_start,
            self.ghost_lr_mult_end,
            self.ghost_lr_ramp,
        )
        return self.base_ghost_lr * mult

    @staticmethod
    def _linear_decay(step: int, start: float, end: float, total_steps: int) -> float:
        if step >= total_steps:
            return end
        fraction = step / total_steps
        return start + fraction * (end - start)
