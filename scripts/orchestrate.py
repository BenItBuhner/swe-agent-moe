"""Orchestrate the full multi-stage training pipeline.

Runs all phases in sequence: pretrain → midtrain → SFT → RL.
Handles checkpoint continuity, resume-from-failure, and logging.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.model_config import TrainingConfig


@dataclass
class PipelineState:
    completed_phases: list = None
    current_phase: str = ""
    last_checkpoint: str = ""
    total_steps_taken: int = 0
    start_time: float = 0.0

    def __post_init__(self):
        if self.completed_phases is None:
            self.completed_phases = []


class Pipeline:
    PHASES = ["pretrain", "midtrain", "sft", "rl"]

    def __init__(self, output_dir: str = "outputs", resume: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "pipeline_state.json"
        self.train_config = TrainingConfig()
        self.state = self._load_state() if resume else PipelineState()

    def _load_state(self) -> PipelineState:
        if self.state_path.exists():
            with open(self.state_path) as f:
                return PipelineState(**json.load(f))
        return PipelineState()

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(asdict(self.state), f, indent=2, default=str)

    def run_phase(self, phase: str, extra_args: str = ""):
        if phase in self.state.completed_phases:
            print(f"[Pipeline] Phase '{phase}' already completed, skipping.")
            return

        script_map = {
            "pretrain": "train/pretrain.py",
            "midtrain": "train/midtrain.py",
            "sft": "train/sft.py",
            "rl": "train/rl_train.py",
        }

        script = script_map.get(phase)
        if not script:
            raise ValueError(f"Unknown phase: {phase}")

        self.state.current_phase = phase
        self.state.start_time = time.time()
        self._save_state()

        import torch
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cmd = f"torchrun --nproc_per_node={nproc} {script} {extra_args}"

        print(f"\n{'='*60}")
        print(f"[Pipeline] Starting phase: {phase}")
        print(f"[Pipeline] Command: {cmd}")
        print(f"{'='*60}\n")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            print(f"[Pipeline] Phase '{phase}' failed with code {result.returncode}")
            sys.exit(result.returncode)

        elapsed = time.time() - self.state.start_time
        print(f"[Pipeline] Phase '{phase}' completed in {elapsed/60:.1f} minutes")

        self.state.completed_phases.append(phase)
        self.state.last_checkpoint = str(self.output_dir / phase / "final.pt")
        self._save_state()

    def run_all(self):
        import torch
        for phase in self.PHASES:
            self.run_phase(phase)

        print(f"\n{'='*60}")
        print("[Pipeline] ALL PHASES COMPLETE!")
        print(f"[Pipeline] Final checkpoint: {self.state.last_checkpoint}")
        print(f"{'='*60}")

    def status(self):
        print(f"Pipeline Status:")
        print(f"  Completed phases: {', '.join(self.state.completed_phases) or 'none'}")
        print(f"  Current phase: {self.state.current_phase or 'not started'}")
        print(f"  Last checkpoint: {self.state.last_checkpoint or 'none'}")
        return self.state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SWE-Agent MoE Pipeline Orchestrator")
    parser.add_argument("--phase", choices=Pipeline.PHASES + ["all"], default="all",
                        help="Phase to run (default: all)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from previous run state")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")

    args = parser.parse_args()

    import torch
    pipeline = Pipeline(args.output_dir, args.resume)

    if args.status:
        pipeline.status()
    elif args.phase == "all":
        pipeline.run_all()
    else:
        pipeline.run_phase(args.phase)
