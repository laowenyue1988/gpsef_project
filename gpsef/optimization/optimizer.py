# -*- coding: utf-8 -*-
import optuna
from optuna.trial import Trial
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class OptimizationProblem(ABC):
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config

    @abstractmethod
    def create_param_space(self, trial: Trial, stage: int, fixed_params: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate(self, params: Dict[str, Any]) -> float:
        pass


class MultiStageHyperOptimizer:
    def __init__(
        self,
        problem: OptimizationProblem,
        study_name: str = "gpsef_study",
        storage: Optional[str] = None
    ):
        self.problem = problem
        self.study_name = study_name
        self.storage = storage
        self.accumulated_best_params: Dict[str, Any] = {}

    def _create_or_load_study(self) -> optuna.Study:
        return optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )

    def _get_best_params_up_to_stage(self, study: optuna.Study, stage: int) -> Dict[str, Any]:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return {}
        best_by_stage = {}
        for s in range(1, stage):
            stage_trials = [t for t in completed if t.user_attrs.get('stage') == s]
            if stage_trials:
                best_trial = max(stage_trials, key=lambda t: t.value)
                best_by_stage[s] = best_trial
        merged = {}
        for s in sorted(best_by_stage.keys()):
            merged.update(best_by_stage[s].params)
        return merged

    def _objective_wrapper(self, trial: Trial, stage: int, fixed_params: Dict[str, Any]) -> float:
        params = self.problem.create_param_space(trial, stage, fixed_params)
        trial.set_user_attr('stage', stage)
        return self.problem.evaluate(params)

    def run_stage(self, stage: int, n_trials: int, n_jobs: int = 1) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"开始阶段 {stage} 优化")
        logger.info("=" * 60)

        study = self._create_or_load_study()
        fixed_params = self._get_best_params_up_to_stage(study, stage)
        if fixed_params:
            logger.info(f"从前序阶段继承的固定参数: {fixed_params}")

        existing = len([t for t in study.trials
                        if t.user_attrs.get('stage') == stage
                        and t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, n_trials - existing)

        if remaining == 0:
            logger.info(f"阶段 {stage} 已完成 {n_trials} 次试验，无需继续")
        else:
            logger.info(f"本阶段已存在 {existing} 次试验，还需运行 {remaining} 次")
            def obj(trial):
                return self._objective_wrapper(trial, stage, fixed_params)
            study.optimize(obj, n_trials=remaining, n_jobs=n_jobs, show_progress_bar=True)

        stage_trials = [t for t in study.trials
                        if t.user_attrs.get('stage') == stage
                        and t.state == optuna.trial.TrialState.COMPLETE]
        if stage_trials:
            best_trial = max(stage_trials, key=lambda t: t.value)
            self.accumulated_best_params.update(best_trial.params)
            logger.info(f"阶段 {stage} 最优参数: {best_trial.params}")
            logger.info(f"阶段 {stage} 最优值: {best_trial.value:.4f}")
        else:
            logger.warning(f"阶段 {stage} 无有效试验")

        logger.info(f"截至阶段 {stage} 的累积最优参数: {self.accumulated_best_params}")
        return self.accumulated_best_params.copy()

    def run_all_stages(self, stage_configs: Dict[int, Dict[str, int]]) -> Dict[str, Any]:
        for stage in sorted(stage_configs.keys()):
            cfg = stage_configs[stage]
            self.run_stage(stage, cfg["n_trials"], cfg.get("n_jobs", 1))
        return self.accumulated_best_params