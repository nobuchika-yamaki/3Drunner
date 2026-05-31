#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness analysis for the progressive 3D embodied-agent model.

Purpose
-------
This script performs the minimum robustness checks needed to defend the model
against the criticism that the main results are seed-specific, threshold-specific,
or parameter-specific.

Analyses
--------
1. Seed robustness
   Repeats the full level × variant design across independent seeds.

2. Autonomous-life threshold sensitivity
   Recomputes autonomous-life establishment while varying:
   - course_progress threshold
   - active_fraction threshold
   - living_engagement threshold

3. Key-parameter robustness
   Repeats the design under perturbations of:
   - Q risk scale
   - body-model learning-rate scale
   - vertical/fall-damage scale

Expected input
--------------
The original agent script:
    progressive_course_3d_validation_v2.py

Example
-------
python3 -u ~/Desktop/run_progressive_3d_robustness.py \
  --agent-script ~/Desktop/progressive_course_3d_validation_v2.py \
  --mode full \
  --outdir ~/Desktop/progressive_3d_robustness \
  --resume \
  2>&1 | tee ~/Desktop/progressive_3d_robustness.log

Smoke test
----------
python3 -u ~/Desktop/run_progressive_3d_robustness.py \
  --agent-script ~/Desktop/progressive_course_3d_validation_v2.py \
  --mode smoke \
  --outdir ~/Desktop/progressive_3d_robustness_smoke \
  --resume
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import shutil
import statistics
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


LEVELS = ["simple3d", "terrain3d", "gaps3d", "hidden3d", "full3d"]
VARIANTS = ["full_core", "no_q", "damage_only_q", "no_body_model", "frozen_predictor"]

SUMMARY_METRICS = [
    "survival",
    "autonomous_life_established",
    "autonomy_proper",
    "goal_reached",
    "course_progress",
    "final_viability",
    "damage_total",
    "falls",
    "fall_distance_total",
    "successful_jumps",
    "successful_climbs",
    "controlled_drops",
    "safe_intervention_success",
    "q_damage_corr",
    "prediction_mae",
    "law_score",
    "body_model_samples",
]

THRESH_PROGRESS = [0.30, 0.35, 0.40]
THRESH_ACTIVE = [0.08, 0.10, 0.12]
THRESH_LIVING = [0.12, 0.14, 0.16]


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else 0.0


def safe_sd(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


class Logger:
    def __init__(self, outdir: Path) -> None:
        self.t0 = time.time()
        self.path = outdir / "robustness_run.log"
        mkdir(outdir)
        with self.path.open("w", encoding="utf-8") as f:
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclasses.dataclass
class RobustnessConfig:
    agent_script: Path
    outdir: Path
    mode: str
    resume: bool
    levels: List[str]
    variants: List[str]
    steps: int
    length: int
    width: int
    height: int
    seed_count: int
    seed_episodes: int
    param_seed_count: int
    param_episodes: int
    threshold_source: Optional[Path]
    skip_seed: bool
    skip_parameters: bool
    skip_threshold: bool
    skip_figure: bool
    q_scales: List[float]
    body_lr_scales: List[float]
    vertical_damage_scales: List[float]


def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_strings(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_agent_module(agent_script: Path):
    module_name = f"progressive_3d_agent_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(agent_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import agent script: {agent_script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_module(mod, q_scale: float, body_lr_scale: float, vertical_damage_scale: float) -> None:
    # Q risk scale: multiply the effective Q used in action selection.
    original_q_for_decision = mod.Agent3D.q_for_decision

    def q_for_decision_scaled(self):
        return mod.clamp(original_q_for_decision(self) * q_scale)

    mod.Agent3D.q_for_decision = q_for_decision_scaled

    # Body-model learning-rate scale: preserve architecture, perturb online learning speed.
    def body_init_scaled(self):
        self.damage_model = mod.OnlineLinearPredictor(18, lr=0.018 * body_lr_scale)
        self.progress_model = mod.OnlineLinearPredictor(18, lr=0.018 * body_lr_scale)
        self.friction_samples = []
        self.slope_samples = []
        self.height_samples = []
        self.safe_interventions = 0
        self.samples = 0

    mod.BodyConsequenceModel.__init__ = body_init_scaled

    # Vertical/fall-damage scale: perturb damage/pain for vertical events after the base physics step.
    original_course_step = mod.ProgressiveCourse3D.step

    def step_vertical_scaled(self, action, stability, fatigue):
        out = original_course_step(self, action, stability, fatigue)
        if vertical_damage_scale != 1.0:
            event = str(out.get("event", ""))
            fall = float(out.get("fall", 0.0))
            height_diff = float(out.get("height_diff", 0.0))
            vertical_event = (
                fall > 0.0
                or abs(height_diff) > 1.0
                or event in {
                    "uncontrolled_drop",
                    "controlled_drop",
                    "jump_failed",
                    "jump_to_void_failed",
                    "fall_into_gap",
                    "climb_failed",
                    "climb_success",
                    "drop_success",
                    "slip",
                }
            )
            if vertical_event:
                out["damage"] = max(0.0, float(out.get("damage", 0.0)) * vertical_damage_scale)
                out["pain"] = mod.clamp(float(out.get("pain", 0.0)) * vertical_damage_scale)
                if float(out.get("stability_delta", 0.0)) < 0.0:
                    out["stability_delta"] = float(out["stability_delta"]) * vertical_damage_scale
        return out

    mod.ProgressiveCourse3D.step = step_vertical_scaled


def run_agent_batch(
    cfg: RobustnessConfig,
    batch_outdir: Path,
    seed: int,
    episodes: int,
    q_scale: float,
    body_lr_scale: float,
    vertical_damage_scale: float,
    batch_label: str,
    logger: Logger,
) -> Path:
    mkdir(batch_outdir)
    done_marker = batch_outdir / "_ROBUSTNESS_BATCH_DONE.json"
    output_csv = batch_outdir / "all_episode_metrics.csv"

    if cfg.resume and done_marker.exists() and output_csv.exists():
        logger.log(f"SKIP existing batch: {batch_label}")
        return output_csv

    logger.log(
        f"RUN batch={batch_label} seed={seed} episodes={episodes} "
        f"steps={cfg.steps} q_scale={q_scale} body_lr_scale={body_lr_scale} "
        f"vertical_damage_scale={vertical_damage_scale}"
    )

    mod = load_agent_module(cfg.agent_script)
    patch_module(mod, q_scale=q_scale, body_lr_scale=body_lr_scale, vertical_damage_scale=vertical_damage_scale)

    # Use the original Config and run_condition functions to keep the exact model mechanics.
    agent_cfg = mod.Config(
        outdir=batch_outdir,
        seed=seed,
        episodes=episodes,
        steps=cfg.steps,
        length=cfg.length,
        width=cfg.width,
        height=cfg.height,
        resume=cfg.resume,
        trace_stride=max(50, min(250, cfg.steps // 4)),
        mode="robustness",
        levels=cfg.levels,
        variants=cfg.variants,
        make_figures=False,
    )

    agent_logger = mod.ProgressLogger(batch_outdir)
    for level in cfg.levels:
        for variant in cfg.variants:
            mod.run_condition(agent_cfg, level, variant, agent_logger)

    mod.write_outputs(agent_cfg, agent_logger)

    manifest = {
        "batch_label": batch_label,
        "seed": seed,
        "episodes": episodes,
        "steps": cfg.steps,
        "levels": cfg.levels,
        "variants": cfg.variants,
        "q_scale": q_scale,
        "body_lr_scale": body_lr_scale,
        "vertical_damage_scale": vertical_damage_scale,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    done_marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"DONE batch={batch_label}")
    return output_csv


def load_batch_metrics(csv_path: Path, metadata: Dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for k, v in metadata.items():
        df[k] = v
    return df


def summarize_grouped(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = len(g)
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = safe_mean(g[m].values)
                row[f"{m}_sd"] = safe_sd(g[m].values)
        rows.append(row)
    return pd.DataFrame(rows)


def add_delta_vs_full(summary: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    non_variant_group = [c for c in group_cols if c != "variant"]
    for keys, g in summary.groupby(non_variant_group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        full = g[g["variant"] == "full_core"]
        if len(full) == 0:
            continue
        full_row = full.iloc[0]
        for _, r in g.iterrows():
            row = {c: r[c] for c in group_cols}
            if r["variant"] == "full_core":
                continue
            for m in metrics:
                col = f"{m}_mean"
                if col in g.columns:
                    row[f"delta_{m}_variant_minus_full"] = float(r[col]) - float(full_row[col])
            rows.append(row)
    return pd.DataFrame(rows)


def collect_seed_batches(cfg: RobustnessConfig, logger: Logger) -> pd.DataFrame:
    logger.log("COLLECT seed robustness batches")
    dfs = []
    seed_base = cfg.outdir / "seed_robustness"
    for seed_idx in range(cfg.seed_count):
        seed = 1000 + 7919 * seed_idx
        batch_dir = seed_base / f"seed_{seed_idx:02d}_{seed}"
        csv_path = batch_dir / "all_episode_metrics.csv"
        if not csv_path.exists():
            continue
        dfs.append(load_batch_metrics(csv_path, {
            "analysis": "seed_robustness",
            "seed_index": seed_idx,
            "seed_value": seed,
            "q_scale": 1.0,
            "body_lr_scale": 1.0,
            "vertical_damage_scale": 1.0,
            "parameter_family": "baseline",
            "parameter_value": 1.0,
        }))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def collect_parameter_batches(cfg: RobustnessConfig, logger: Logger) -> pd.DataFrame:
    logger.log("COLLECT parameter robustness batches")
    dfs = []
    param_base = cfg.outdir / "parameter_robustness"

    families = []
    for x in cfg.q_scales:
        families.append(("q_scale", x, x, 1.0, 1.0))
    for x in cfg.body_lr_scales:
        families.append(("body_lr_scale", x, 1.0, x, 1.0))
    for x in cfg.vertical_damage_scales:
        families.append(("vertical_damage_scale", x, 1.0, 1.0, x))

    for family, value, q_scale, body_lr_scale, vertical_damage_scale in families:
        for seed_idx in range(cfg.param_seed_count):
            seed = 50000 + 1543 * seed_idx + int(round(value * 1000))
            label = f"{family}_{value:g}_seed_{seed_idx:02d}"
            batch_dir = param_base / family / f"value_{value:g}" / f"seed_{seed_idx:02d}_{seed}"
            csv_path = batch_dir / "all_episode_metrics.csv"
            if not csv_path.exists():
                continue
            dfs.append(load_batch_metrics(csv_path, {
                "analysis": "parameter_robustness",
                "seed_index": seed_idx,
                "seed_value": seed,
                "q_scale": q_scale,
                "body_lr_scale": body_lr_scale,
                "vertical_damage_scale": vertical_damage_scale,
                "parameter_family": family,
                "parameter_value": value,
            }))

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def run_seed_robustness(cfg: RobustnessConfig, logger: Logger) -> pd.DataFrame:
    if cfg.skip_seed:
        logger.log("SKIP seed robustness by request")
        return pd.DataFrame()

    logger.log("STEP 1/5 seed robustness")
    seed_base = cfg.outdir / "seed_robustness"
    mkdir(seed_base)

    for seed_idx in range(cfg.seed_count):
        seed = 1000 + 7919 * seed_idx
        batch_label = f"seed_robustness_seed_{seed_idx:02d}_{seed}"
        batch_dir = seed_base / f"seed_{seed_idx:02d}_{seed}"
        run_agent_batch(
            cfg,
            batch_outdir=batch_dir,
            seed=seed,
            episodes=cfg.seed_episodes,
            q_scale=1.0,
            body_lr_scale=1.0,
            vertical_damage_scale=1.0,
            batch_label=batch_label,
            logger=logger,
        )

    df = collect_seed_batches(cfg, logger)
    if len(df):
        out = cfg.outdir / "seed_robustness_all_episode_metrics.csv"
        df.to_csv(out, index=False)
        logger.log(f"WROTE {out.name}")

        summary = summarize_grouped(df, ["analysis", "seed_index", "seed_value", "level", "variant"], SUMMARY_METRICS)
        summary.to_csv(cfg.outdir / "seed_robustness_summary_by_seed.csv", index=False)

        pooled = summarize_grouped(df, ["analysis", "level", "variant"], SUMMARY_METRICS)
        pooled.to_csv(cfg.outdir / "seed_robustness_summary_pooled.csv", index=False)

        deltas = add_delta_vs_full(pooled, ["analysis", "level", "variant"], SUMMARY_METRICS)
        deltas.to_csv(cfg.outdir / "seed_robustness_deltas_vs_full_core.csv", index=False)

    return df


def run_parameter_robustness(cfg: RobustnessConfig, logger: Logger) -> pd.DataFrame:
    if cfg.skip_parameters:
        logger.log("SKIP parameter robustness by request")
        return pd.DataFrame()

    logger.log("STEP 2/5 key-parameter robustness")
    param_base = cfg.outdir / "parameter_robustness"
    mkdir(param_base)

    families = []
    for x in cfg.q_scales:
        families.append(("q_scale", x, x, 1.0, 1.0))
    for x in cfg.body_lr_scales:
        families.append(("body_lr_scale", x, 1.0, x, 1.0))
    for x in cfg.vertical_damage_scales:
        families.append(("vertical_damage_scale", x, 1.0, 1.0, x))

    # Deduplicate exact baseline repeated in all three families only at reporting time.
    for family, value, q_scale, body_lr_scale, vertical_damage_scale in families:
        for seed_idx in range(cfg.param_seed_count):
            seed = 50000 + 1543 * seed_idx + int(round(value * 1000))
            batch_label = f"{family}_{value:g}_seed_{seed_idx:02d}"
            batch_dir = param_base / family / f"value_{value:g}" / f"seed_{seed_idx:02d}_{seed}"
            run_agent_batch(
                cfg,
                batch_outdir=batch_dir,
                seed=seed,
                episodes=cfg.param_episodes,
                q_scale=q_scale,
                body_lr_scale=body_lr_scale,
                vertical_damage_scale=vertical_damage_scale,
                batch_label=batch_label,
                logger=logger,
            )

    df = collect_parameter_batches(cfg, logger)
    if len(df):
        out = cfg.outdir / "parameter_robustness_all_episode_metrics.csv"
        df.to_csv(out, index=False)
        logger.log(f"WROTE {out.name}")

        summary = summarize_grouped(
            df,
            ["analysis", "parameter_family", "parameter_value", "q_scale", "body_lr_scale", "vertical_damage_scale", "level", "variant"],
            SUMMARY_METRICS,
        )
        summary.to_csv(cfg.outdir / "parameter_robustness_summary.csv", index=False)

        deltas = add_delta_vs_full(
            summary,
            ["analysis", "parameter_family", "parameter_value", "q_scale", "body_lr_scale", "vertical_damage_scale", "level", "variant"],
            SUMMARY_METRICS,
        )
        deltas.to_csv(cfg.outdir / "parameter_robustness_deltas_vs_full_core.csv", index=False)

    return df


def recompute_living_engagement(row: pd.Series, course_length: int = 120) -> float:
    p = float(row.get("course_progress", 0.0))
    af = float(row.get("active_fraction", 0.0))
    u = clamp(float(row.get("unknown_revealed", 0.0)) / max(1.0, course_length * 0.55))
    r = clamp(float(row.get("resources_collected", 0.0)) / max(1.0, course_length * 0.35))
    law = clamp(float(row.get("law_score", 0.0)))
    interventions = (
        float(row.get("successful_climbs", 0.0))
        + float(row.get("successful_jumps", 0.0))
        + float(row.get("controlled_drops", 0.0))
    )
    i = clamp(interventions / 15.0)
    return clamp(0.50 * p + 0.14 * af + 0.10 * u + 0.08 * r + 0.10 * law + 0.08 * i)


def threshold_sensitivity(source_df: pd.DataFrame, cfg: RobustnessConfig, logger: Logger) -> pd.DataFrame:
    logger.log("STEP 3/5 threshold sensitivity")

    rows = []
    df = source_df.copy()
    if len(df) == 0:
        logger.log("No source rows for threshold sensitivity")
        return pd.DataFrame()

    df["living_engagement_recomputed"] = df.apply(lambda r: recompute_living_engagement(r, cfg.length), axis=1)

    for p_thr in THRESH_PROGRESS:
        for af_thr in THRESH_ACTIVE:
            for le_thr in THRESH_LIVING:
                d = df.copy()
                d["autonomous_life_established_recomputed"] = (
                    (d["survival"].astype(float) >= 1.0)
                    & (d["bounded_internal"].astype(float) >= 1.0)
                    & (d["final_viability"].astype(float) > 0.10)
                    & (d["course_progress"].astype(float) > p_thr)
                    & (d["active_fraction"].astype(float) > af_thr)
                    & (d["living_engagement_recomputed"].astype(float) > le_thr)
                    & (d["passive_shutdown_fraction"].astype(float) < 0.20)
                ).astype(int)

                for (level, variant), g in d.groupby(["level", "variant"]):
                    rows.append({
                        "progress_threshold": p_thr,
                        "active_fraction_threshold": af_thr,
                        "living_engagement_threshold": le_thr,
                        "level": level,
                        "variant": variant,
                        "n": len(g),
                        "autonomous_life_established_recomputed_mean": safe_mean(g["autonomous_life_established_recomputed"]),
                        "course_progress_mean": safe_mean(g["course_progress"]),
                        "falls_mean": safe_mean(g["falls"]),
                        "law_score_mean": safe_mean(g["law_score"]),
                    })

    out = pd.DataFrame(rows)
    out.to_csv(cfg.outdir / "threshold_sensitivity_summary.csv", index=False)

    deltas = []
    for keys, g in out.groupby(["progress_threshold", "active_fraction_threshold", "living_engagement_threshold", "level"]):
        full = g[g["variant"] == "full_core"]
        if len(full) == 0:
            continue
        full_v = float(full["autonomous_life_established_recomputed_mean"].iloc[0])
        for _, r in g.iterrows():
            if r["variant"] == "full_core":
                continue
            deltas.append({
                "progress_threshold": keys[0],
                "active_fraction_threshold": keys[1],
                "living_engagement_threshold": keys[2],
                "level": keys[3],
                "variant": r["variant"],
                "delta_autonomous_life_established_variant_minus_full": float(r["autonomous_life_established_recomputed_mean"]) - full_v,
            })

    pd.DataFrame(deltas).to_csv(cfg.outdir / "threshold_sensitivity_deltas_vs_full_core.csv", index=False)
    logger.log("WROTE threshold_sensitivity_summary.csv")
    return out


def build_claim_checks(seed_df: pd.DataFrame, param_df: pd.DataFrame, thresh_df: pd.DataFrame, cfg: RobustnessConfig, logger: Logger) -> pd.DataFrame:
    logger.log("STEP 4/5 claim checks")
    checks = []

    def add_check(source: str, claim: str, passed: bool, value: Any, details: str) -> None:
        checks.append({
            "source": source,
            "claim": claim,
            "passed": int(bool(passed)),
            "value": value,
            "details": details,
        })

    if len(seed_df):
        pooled = summarize_grouped(seed_df, ["level", "variant"], SUMMARY_METRICS)
        pivot_progress = pooled.pivot(index="level", columns="variant", values="course_progress_mean")
        pivot_falls = pooled.pivot(index="level", columns="variant", values="falls_mean")
        pivot_law = pooled.pivot(index="level", columns="variant", values="law_score_mean")

        fc_simple = float(pivot_progress.loc["simple3d", "full_core"])
        fc_terrain = float(pivot_progress.loc["terrain3d", "full_core"])
        fc_gaps = float(pivot_progress.loc["gaps3d", "full_core"])
        fc_hidden = float(pivot_progress.loc["hidden3d", "full_core"])
        fc_full = float(pivot_progress.loc["full3d", "full_core"])

        add_check("seed_robustness", "full_core high progress in simple3d and terrain3d", fc_simple >= 0.95 and fc_terrain >= 0.95, f"{fc_simple:.3f}, {fc_terrain:.3f}", "Required both >= 0.95")
        add_check("seed_robustness", "progression boundary emerges from gaps3d onward", fc_gaps < fc_terrain and fc_hidden < fc_gaps and fc_full < fc_gaps, f"terrain={fc_terrain:.3f}, gaps={fc_gaps:.3f}, hidden={fc_hidden:.3f}, full={fc_full:.3f}", "Required gaps < terrain and hidden/full < gaps")
        add_check("seed_robustness", "falls increase in full3d relative to gaps3d", float(pivot_falls.loc["full3d", "full_core"]) > float(pivot_falls.loc["gaps3d", "full_core"]), f"gaps={float(pivot_falls.loc['gaps3d','full_core']):.3f}, full={float(pivot_falls.loc['full3d','full_core']):.3f}", "Required full3d falls > gaps3d falls")
        add_check("seed_robustness", "law score increases from simple3d to full3d", float(pivot_law.loc["full3d", "full_core"]) > float(pivot_law.loc["simple3d", "full_core"]), f"simple={float(pivot_law.loc['simple3d','full_core']):.3f}, full={float(pivot_law.loc['full3d','full_core']):.3f}", "Required full3d law_score > simple3d law_score")

        # Q-specific gap claim.
        no_q_delta_prog = float(pivot_progress.loc["gaps3d", "no_q"] - pivot_progress.loc["gaps3d", "full_core"])
        no_q_delta_falls = float(pivot_falls.loc["gaps3d", "no_q"] - pivot_falls.loc["gaps3d", "full_core"])
        add_check("seed_robustness", "no_q impairs gaps3d progress", no_q_delta_prog < 0.0, f"{no_q_delta_prog:.3f}", "Required no_q - full_core < 0")
        add_check("seed_robustness", "no_q increases gaps3d falls", no_q_delta_falls > 0.0, f"{no_q_delta_falls:.3f}", "Required no_q - full_core > 0")

        # Body model claim.
        law_body_delta = float(pivot_law.loc["gaps3d", "no_body_model"] - pivot_law.loc["gaps3d", "full_core"])
        add_check("seed_robustness", "no_body_model abolishes gaps3d law score", law_body_delta < -0.25, f"{law_body_delta:.3f}", "Required large negative law-score delta")

    if len(param_df):
        summ = summarize_grouped(param_df, ["parameter_family", "parameter_value", "level", "variant"], SUMMARY_METRICS)
        for (family, value), g in summ.groupby(["parameter_family", "parameter_value"]):
            try:
                pvt_prog = g.pivot(index="level", columns="variant", values="course_progress_mean")
                pvt_falls = g.pivot(index="level", columns="variant", values="falls_mean")
                pvt_law = g.pivot(index="level", columns="variant", values="law_score_mean")
                boundary_ok = float(pvt_prog.loc["gaps3d", "full_core"]) < float(pvt_prog.loc["terrain3d", "full_core"])
                law_ok = float(pvt_law.loc["gaps3d", "no_body_model"]) < 0.05 and float(pvt_law.loc["gaps3d", "full_core"]) > 0.20
                add_check("parameter_robustness", f"{family}={value:g}: boundary retained", boundary_ok, f"terrain={float(pvt_prog.loc['terrain3d','full_core']):.3f}, gaps={float(pvt_prog.loc['gaps3d','full_core']):.3f}", "Required gaps full_core progress < terrain full_core progress")
                add_check("parameter_robustness", f"{family}={value:g}: body model law-score contrast retained", law_ok, f"full={float(pvt_law.loc['gaps3d','full_core']):.3f}, no_body={float(pvt_law.loc['gaps3d','no_body_model']):.3f}", "Required full_core law_score > 0.20 and no_body_model < 0.05")
                if "no_q" in pvt_falls.columns:
                    q_falls_delta = float(pvt_falls.loc["gaps3d", "no_q"] - pvt_falls.loc["gaps3d", "full_core"])
                    add_check("parameter_robustness", f"{family}={value:g}: no_q gaps3d fall penalty retained", q_falls_delta > 0.0, f"{q_falls_delta:.3f}", "Required no_q - full_core falls > 0")
            except Exception as e:
                add_check("parameter_robustness", f"{family}={value:g}: claim check failed to compute", False, "NA", str(e))

    if len(thresh_df):
        # Check whether full_core remains at least as good as no_body_model and no_q in gaps3d in most threshold settings.
        gaps = thresh_df[thresh_df["level"] == "gaps3d"]
        if len(gaps):
            counts = []
            for keys, g in gaps.groupby(["progress_threshold", "active_fraction_threshold", "living_engagement_threshold"]):
                full = g[g["variant"] == "full_core"]
                no_q = g[g["variant"] == "no_q"]
                no_body = g[g["variant"] == "no_body_model"]
                if len(full) and len(no_q):
                    counts.append(float(full["autonomous_life_established_recomputed_mean"].iloc[0]) >= float(no_q["autonomous_life_established_recomputed_mean"].iloc[0]))
                if len(full) and len(no_body):
                    counts.append(float(full["autonomous_life_established_recomputed_mean"].iloc[0]) >= float(no_body["autonomous_life_established_recomputed_mean"].iloc[0]))
            frac = sum(counts) / len(counts) if counts else 0.0
            add_check("threshold_sensitivity", "gaps3d full_core establishment is robust to threshold variation", frac >= 0.80, f"{frac:.3f}", "Required >=80% of threshold comparisons full_core >= ablation")

    out = pd.DataFrame(checks)
    out.to_csv(cfg.outdir / "robustness_claim_checks.csv", index=False)
    logger.log("WROTE robustness_claim_checks.csv")
    return out


def write_report(seed_df: pd.DataFrame, param_df: pd.DataFrame, thresh_df: pd.DataFrame, checks: pd.DataFrame, cfg: RobustnessConfig, logger: Logger) -> Path:
    logger.log("STEP 5/5 consolidated report")
    path = cfg.outdir / "robustness_consolidated_report.txt"

    lines = []
    lines.append("Progressive 3D robustness analysis")
    lines.append("=" * 38)
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append("This analysis tests whether the main progressive-3D results depend on a single seed, a single autonomous-life threshold definition, or a narrow parameter setting.")
    lines.append("")
    lines.append("Configuration")
    lines.append("-------------")
    lines.append(f"agent_script: {cfg.agent_script}")
    lines.append(f"mode: {cfg.mode}")
    lines.append(f"levels: {', '.join(cfg.levels)}")
    lines.append(f"variants: {', '.join(cfg.variants)}")
    lines.append(f"steps: {cfg.steps}")
    lines.append(f"seed_count: {cfg.seed_count}")
    lines.append(f"seed_episodes: {cfg.seed_episodes}")
    lines.append(f"param_seed_count: {cfg.param_seed_count}")
    lines.append(f"param_episodes: {cfg.param_episodes}")
    lines.append(f"q_scales: {cfg.q_scales}")
    lines.append(f"body_lr_scales: {cfg.body_lr_scales}")
    lines.append(f"vertical_damage_scales: {cfg.vertical_damage_scales}")
    lines.append("")

    if len(seed_df):
        pooled = summarize_grouped(seed_df, ["level", "variant"], SUMMARY_METRICS)
        lines.append("Seed robustness: selected pooled means")
        lines.append("-------------------------------------")
        keep = ["level", "variant", "course_progress_mean", "autonomous_life_established_mean", "damage_total_mean", "falls_mean", "law_score_mean"]
        lines.append(pooled[keep].to_string(index=False))
        lines.append("")

    if len(param_df):
        param_summary = summarize_grouped(param_df, ["parameter_family", "parameter_value", "level", "variant"], SUMMARY_METRICS)
        lines.append("Parameter robustness: selected rows")
        lines.append("-----------------------------------")
        keep = ["parameter_family", "parameter_value", "level", "variant", "course_progress_mean", "falls_mean", "law_score_mean"]
        lines.append(param_summary[keep].head(80).to_string(index=False))
        lines.append("")

    if len(thresh_df):
        lines.append("Threshold sensitivity")
        lines.append("---------------------")
        lines.append(f"threshold settings evaluated: {len(thresh_df[['progress_threshold','active_fraction_threshold','living_engagement_threshold']].drop_duplicates())}")
        lines.append("")

    if len(checks):
        lines.append("Claim checks")
        lines.append("------------")
        lines.append(checks.to_string(index=False))
        lines.append("")
        pass_rate = safe_mean(checks["passed"].values)
        lines.append(f"Claim-check pass fraction: {pass_rate:.3f}")
        lines.append("")

    lines.append("Generated output files")
    lines.append("----------------------")
    for p in sorted(cfg.outdir.glob("*.csv")):
        lines.append(f"- {p.name}")
    lines.append("- robustness_consolidated_report.txt")
    lines.append("- robustness_outputs.zip")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.log(f"WROTE {path.name}")
    return path


def zip_outputs(cfg: RobustnessConfig, logger: Logger) -> Path:
    zip_path = cfg.outdir / "robustness_outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(cfg.outdir.glob("*.csv")):
            z.write(p, arcname=p.name)
        for p in sorted(cfg.outdir.glob("*.txt")):
            z.write(p, arcname=p.name)
        for p in sorted(cfg.outdir.glob("*.log")):
            z.write(p, arcname=p.name)
        report = cfg.outdir / "robustness_consolidated_report.txt"
        if report.exists():
            z.write(report, arcname=report.name)
    logger.log(f"WROTE {zip_path.name}")
    return zip_path


def build_config(args: argparse.Namespace) -> RobustnessConfig:
    if args.mode == "smoke":
        seed_count = 1
        seed_episodes = 2
        param_seed_count = 1
        param_episodes = 2
        steps = 250
        q_scales = [1.0]
        body_lr_scales = [1.0]
        vertical_damage_scales = [1.0]
    elif args.mode == "quick":
        seed_count = 3
        seed_episodes = 10
        param_seed_count = 2
        param_episodes = 6
        steps = 800
        q_scales = [0.75, 1.0, 1.25]
        body_lr_scales = [0.5, 1.0, 2.0]
        vertical_damage_scales = [0.75, 1.0, 1.25]
    else:
        seed_count = 5
        seed_episodes = 20
        param_seed_count = 3
        param_episodes = 10
        steps = 1200
        q_scales = [0.75, 1.0, 1.25]
        body_lr_scales = [0.5, 1.0, 2.0]
        vertical_damage_scales = [0.75, 1.0, 1.25]

    if args.seed_count is not None:
        seed_count = args.seed_count
    if args.seed_episodes is not None:
        seed_episodes = args.seed_episodes
    if args.param_seed_count is not None:
        param_seed_count = args.param_seed_count
    if args.param_episodes is not None:
        param_episodes = args.param_episodes
    if args.steps is not None:
        steps = args.steps
    if args.q_scales is not None:
        q_scales = parse_list_floats(args.q_scales)
    if args.body_lr_scales is not None:
        body_lr_scales = parse_list_floats(args.body_lr_scales)
    if args.vertical_damage_scales is not None:
        vertical_damage_scales = parse_list_floats(args.vertical_damage_scales)

    levels = parse_list_strings(args.levels)
    variants = parse_list_strings(args.variants)
    for level in levels:
        if level not in LEVELS:
            raise ValueError(f"Unknown level: {level}")
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

    threshold_source = Path(args.threshold_source).expanduser() if args.threshold_source else None

    return RobustnessConfig(
        agent_script=Path(args.agent_script).expanduser(),
        outdir=Path(args.outdir).expanduser(),
        mode=args.mode,
        resume=args.resume,
        levels=levels,
        variants=variants,
        steps=steps,
        length=args.length,
        width=args.width,
        height=args.height,
        seed_count=seed_count,
        seed_episodes=seed_episodes,
        param_seed_count=param_seed_count,
        param_episodes=param_episodes,
        threshold_source=threshold_source,
        skip_seed=args.skip_seed,
        skip_parameters=args.skip_parameters,
        skip_threshold=args.skip_threshold,
        skip_figure=args.skip_figure,
        q_scales=q_scales,
        body_lr_scales=body_lr_scales,
        vertical_damage_scales=vertical_damage_scales,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robustness analysis for progressive 3D embodied-agent results.")
    p.add_argument("--agent-script", required=True, help="Path to progressive_course_3d_validation_v2.py")
    p.add_argument("--outdir", required=True)
    p.add_argument("--mode", choices=["smoke", "quick", "full"], default="full")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--levels", default=",".join(LEVELS))
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--length", type=int, default=120)
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--height", type=int, default=6)

    p.add_argument("--seed-count", type=int, default=None)
    p.add_argument("--seed-episodes", type=int, default=None)
    p.add_argument("--param-seed-count", type=int, default=None)
    p.add_argument("--param-episodes", type=int, default=None)

    p.add_argument("--q-scales", default=None, help="Comma-separated values, e.g. 0.75,1.0,1.25")
    p.add_argument("--body-lr-scales", default=None, help="Comma-separated values, e.g. 0.5,1.0,2.0")
    p.add_argument("--vertical-damage-scales", default=None, help="Comma-separated values, e.g. 0.75,1.0,1.25")

    p.add_argument("--threshold-source", default=None, help="Optional existing all_episode_metrics.csv for threshold-only analysis.")
    p.add_argument("--skip-seed", action="store_true")
    p.add_argument("--skip-parameters", action="store_true")
    p.add_argument("--skip-threshold", action="store_true")
    p.add_argument("--skip-figure", action="store_true", help="Reserved; no figures are generated by default.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    mkdir(cfg.outdir)
    logger = Logger(cfg.outdir)

    logger.log("Progressive 3D robustness analysis started")
    logger.log(f"agent_script={cfg.agent_script}")
    logger.log(f"outdir={cfg.outdir}")

    if not cfg.agent_script.exists():
        raise FileNotFoundError(f"Agent script does not exist: {cfg.agent_script}")

    seed_df = run_seed_robustness(cfg, logger)
    param_df = run_parameter_robustness(cfg, logger)

    threshold_source_df = pd.DataFrame()
    if not cfg.skip_threshold:
        if cfg.threshold_source is not None:
            logger.log(f"Loading threshold source: {cfg.threshold_source}")
            threshold_source_df = pd.read_csv(cfg.threshold_source)
        elif len(seed_df):
            threshold_source_df = seed_df
        else:
            candidate = cfg.outdir / "seed_robustness_all_episode_metrics.csv"
            if candidate.exists():
                threshold_source_df = pd.read_csv(candidate)

    thresh_df = threshold_sensitivity(threshold_source_df, cfg, logger) if not cfg.skip_threshold else pd.DataFrame()

    checks = build_claim_checks(seed_df, param_df, thresh_df, cfg, logger)
    write_report(seed_df, param_df, thresh_df, checks, cfg, logger)
    zip_outputs(cfg, logger)

    logger.log("Robustness analysis completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
