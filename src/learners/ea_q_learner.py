"""
EA-enhanced Q-Learner for ALMA
Extends the original QLearner with evolutionary algorithm optimization for task allocation.
"""

import copy
import torch as th
import numpy as np
from components.episode_buffer import EpisodeBatch

from .q_learner import QLearner
from ea.ea_manager import EAManager


class EAQLearner(QLearner):
    """
    Q-Learner enhanced with Evolutionary Algorithm optimization.

    Combines:
    - Gradient-based learning for low-level Q-functions (from QLearner)
    - Evolutionary optimization for high-level task allocation
    - Bidirectional synchronization between EA and gradient learning
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        # EA will be initialized later when runner is available
        self.ea_manager = None
        self.ea_enabled = args.ea.get('enabled', True)

        # EA training counters
        self.last_ea_T = 0
        self.ea_interval = args.ea.get('interval', 5000)

        # EA logging
        self.log_ea_stats_t = -args.learner_log_interval - 1

    def setup_ea(self, runner):
        """
        Setup EA Manager with runner.

        Must be called after runner is created in run.py
        """
        if not self.ea_enabled:
            return

        # Check if learned allocation policy exists
        task_alloc = self.args.hier_agent.get("task_allocation", None)
        if task_alloc != "aql":
            self.logger.console_logger.warning(
                f"EA enabled but task_allocation='{task_alloc}' (need 'aql'). Disabling EA."
            )
            self.ea_enabled = False
            return

        # Verify MAC has allocation policy
        if not hasattr(self.mac, 'alloc_policy') or self.mac.alloc_policy is None:
            self.logger.console_logger.warning(
                "EA enabled but MAC has no alloc_policy. Disabling EA."
            )
            self.ea_enabled = False
            return

        self.ea_manager = EAManager(
            args=self.args,
            mac=self.mac,
            runner=runner,
            logger=self.logger.console_logger
        )

        self.logger.console_logger.info("EA Manager setup complete")

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        Training step including both gradient and EA updates.
        """
        # Standard Q-learning update
        super().train(batch, t_env, episode_num)

        # EA update (periodic)
        if self.ea_enabled and self.ea_manager is not None:
            self._ea_update(t_env)

    def _ea_update(self, t_env: int):
        """
        Perform EA update if interval has passed.
        """
        if t_env - self.last_ea_T < self.ea_interval:
            return

        self.logger.console_logger.info(f"Performing EA update at t_env={t_env}")

        # Evaluate population
        ea_stats = self.ea_manager.evaluate_population(t_env)

        # Log evaluation stats
        for name, value in ea_stats.items():
            self.logger.log_stat(name, value, t_env)

        # Evolve population
        evolve_stats = self.ea_manager.evolve()

        # Bidirectional synchronization
        main_fitness = self.ea_manager.evaluate_main_fitness()
        sync_result = self.ea_manager.bidirectional_sync_step(main_fitness)

        # Log sync result
        sync_type_map = {"no_sync": 0, "ea_to_main": 1, "main_to_ea": 2}
        self.logger.log_stat("ea/sync_type", sync_type_map.get(sync_result, 0), t_env)
        self.logger.log_stat("ea/main_fitness", main_fitness, t_env)

        self.last_ea_T = t_env

        if t_env - self.log_ea_stats_t >= self.args.learner_log_interval:
            self.logger.console_logger.info(
                f"EA Stats - Gen: {ea_stats['ea/generation']}, "
                f"Best: {ea_stats['ea/best_fitness']:.4f}, "
                f"Mean: {ea_stats['ea/mean_fitness']:.4f}, "
                f"Sync: {sync_result}"
            )
            self.log_ea_stats_t = t_env

    def alloc_train_aql(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        Allocation training with optional EA enhancement.

        When EA is enabled, periodically syncs EA's best allocation
        with the gradient-trained allocation policy.
        """
        # Standard allocation training
        stats, new_alloc = super().alloc_train_aql(batch, t_env, episode_num)

        return stats, new_alloc

    def save_models(self, path):
        """Save models including EA state."""
        super().save_models(path)

        # Save EA state
        if self.ea_enabled and self.ea_manager is not None:
            ea_state = self.ea_manager.save_state()
            th.save(ea_state, "{}ea_state.th".format(path))

    def load_models(self, path, pi_only=False, evaluate=False):
        """Load models including EA state."""
        super().load_models(path, pi_only=pi_only, evaluate=evaluate)

        # Load EA state if exists
        if self.ea_enabled and self.ea_manager is not None and not evaluate:
            try:
                ea_state = th.load(
                    "{}ea_state.th".format(path),
                    map_location=lambda storage, loc: storage
                )
                device = "cuda" if self.args.use_cuda else "cpu"
                self.ea_manager.load_state(ea_state, device)
                self.logger.console_logger.info("Loaded EA state from checkpoint")
            except FileNotFoundError:
                self.logger.console_logger.info("No EA state found in checkpoint")


class EAOnlyQLearner(QLearner):
    """
    Q-Learner that uses EA exclusively for allocation optimization.

    The low-level Q-functions are still trained with gradients,
    but allocation policy is optimized only through evolution.
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        self.ea_manager = None
        self.ea_enabled = args.ea.get('enabled', True)
        self.last_ea_T = 0
        self.ea_interval = args.ea.get('interval', 2000)

        # Disable gradient-based allocation training
        if hasattr(self, 'alloc_pi_optimiser'):
            del self.alloc_pi_optimiser
        if hasattr(self, 'alloc_q_optimiser'):
            del self.alloc_q_optimiser

    def setup_ea(self, runner):
        """Setup EA Manager."""
        if not self.ea_enabled:
            return

        # Check if learned allocation policy exists
        task_alloc = self.args.hier_agent.get("task_allocation", None)
        if task_alloc != "aql":
            self.logger.console_logger.warning(
                f"EA enabled but task_allocation='{task_alloc}' (need 'aql'). Disabling EA."
            )
            self.ea_enabled = False
            return

        # Verify MAC has allocation policy
        if not hasattr(self.mac, 'alloc_policy') or self.mac.alloc_policy is None:
            self.logger.console_logger.warning(
                "EA enabled but MAC has no alloc_policy. Disabling EA."
            )
            self.ea_enabled = False
            return

        self.ea_manager = EAManager(
            args=self.args,
            mac=self.mac,
            runner=runner,
            logger=self.logger.console_logger
        )

        self.logger.console_logger.info("EA-Only Manager setup complete")

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Train with EA for allocation, gradient for Q-functions."""
        # Only Q-learning for low-level (skip allocation gradient update)
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        reset = batch["reset"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.agent['subtask_cond'] is not None:
            rewards = batch['task_rewards'][:, :-1]
            terminated = batch['tasks_terminated'][:, :-1].float()
            mask = mask.repeat(1, 1, self.args.n_tasks)
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            task_has_agents = (1 - batch['entity2task_mask'][:, :-1, :self.args.n_agents]).sum(2) > 0
            mask *= task_has_agents.float()

        self.mac.init_hidden(batch.batch_size)
        self.mac.train()
        self.target_mac.eval()
        if self.mixer is not None:
            self.mixer.train()
            self.target_mixer.eval()

        all_mac_out, mac_info = self.mac.forward(batch, t=None)

        batch_mult = 1
        rep_actions = actions.repeat(batch_mult, 1, 1, 1)
        all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)

        mac_out = all_mac_out
        chosen_action_qvals = all_chosen_action_qvals

        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out, _ = self.target_mac.forward(batch, t=None, target=True)

        if self.args.agent['subtask_cond'] is not None:
            from components.action_selectors import parse_avail_actions
            allocs = (1 - batch['entity2task_mask'][:, :, :self.args.n_agents])
            avail_actions_targ = parse_avail_actions(avail_actions[:, 1:], allocs[:, :-1], self.args)
        else:
            avail_actions_targ = avail_actions[:, 1:]
        target_mac_out = target_mac_out[:, 1:]
        target_mac_out[avail_actions_targ == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()[:, 1:]
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
            chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)
            target_max_qvals = self.target_mixer.denormalize(target_max_qvals)
            targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals).detach()
            if self.args.popart:
                targets = self.mixer.popart_update(targets, mask)
        else:
            targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals).detach()

        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # EA update
        if self.ea_enabled and self.ea_manager is not None:
            self._ea_update(t_env)

    def _ea_update(self, t_env: int):
        """Perform EA update."""
        if t_env - self.last_ea_T < self.ea_interval:
            return

        # Evaluate and evolve
        ea_stats = self.ea_manager.evaluate_population(t_env)
        self.ea_manager.evolve()

        # Always apply best EA solution to policy
        self.ea_manager.apply_best_to_policy()

        # Log
        for name, value in ea_stats.items():
            self.logger.log_stat(name, value, t_env)

        self.last_ea_T = t_env

    def alloc_train_aql(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Skip gradient-based allocation training - EA handles this."""
        return {}, None
