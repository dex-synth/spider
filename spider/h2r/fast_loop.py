"""Upstream run_mjwp_fast retry/revert loop, without viewer or dataset I/O.

The driver supplies private checkpoints, optimization and snapshot replay.
No callback can reset the independent evaluator.
"""

import math
import time


def run_fast(driver, max_steps, report):
    """Preserve the native local retries, fixed anchor recovery and final replay."""
    cfg = driver.config
    original_iters = cfg.max_num_iterations
    original_params = list(cfg.env_params_list)
    original_noise = cfg.noise_scale.clone()

    def boost(level):
        iterations = original_iters * 2**level
        cfg.max_num_iterations = iterations
        cfg.beta_traj = cfg.final_noise_scale ** (1 / iterations)
        cfg.env_params_list = original_params + [original_params[-1]] * max(
            0, iterations - len(original_params)
        )
        cfg.noise_scale = original_noise * math.sqrt(2) ** level

    history = []
    recovering = False
    anchor = None
    anchor_step = None
    revert_attempts = local_retries = boost_level = 0
    while True:
        started = time.perf_counter()
        checkpoint = driver.save_checkpoint()
        step, info = driver.optimize_chunk()
        succeeded = bool(info["succeeded"][0])
        report(
            {
                "native_step": step,
                "compute_s": time.perf_counter() - started,
                "succeeded": succeeded,
                "opt_steps": info["opt_steps"].tolist(),
                "num_attempts": info["num_attempts"].tolist(),
                "local_retries": local_retries,
                "boost_level": boost_level,
                "recovering": recovering,
                "anchor_step": anchor_step,
                "revert_attempts": revert_attempts,
            }
        )
        if succeeded:
            if recovering and step > anchor_step:
                boost_level = revert_attempts = local_retries = 0
                anchor = anchor_step = None
                boost(0)
                recovering = False
            local_retries = 0
            if not recovering:
                history.append(checkpoint)
            step = driver.replay_chunk(step, info)
        elif local_retries < cfg.max_local_retries:
            local_retries += 1
            driver.load_checkpoint(checkpoint)
            boost_level += 1
            boost(boost_level)
            continue
        elif recovering and anchor is not None:
            revert_attempts += 1
            if revert_attempts > cfg.max_revert_depth:
                boost(0)
                driver.truncate(history[-1] if history else None)
                driver.replay_chunk(step, info)
                return False
            local_retries = 0
            driver.load_checkpoint(anchor)
            boost_level = 0
            boost(0)
            continue
        elif history:
            local_retries = 0
            anchor = history.pop()
            driver.load_checkpoint(anchor)
            anchor_step = driver.step
            revert_attempts = 1
            boost_level = 0
            boost(0)
            recovering = True
            continue
        else:
            boost(0)
            driver.truncate(history[-1] if history else None)
            driver.replay_chunk(step, info)
            return False
        if step >= max_steps:
            return True
