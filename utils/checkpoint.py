"""Checkpoint and run-config helpers guarding the alignment-mode experiment.

The scoring mode ("sim" / "attn" / "both") is behaviorally invisible to the
state dict (identical keys in all modes), so a checkpoint trained under one
mode would silently evaluate under another if the mode came only from
config.json. save_checkpoint stamps it next to the weights; loaders prefer
the stamp via resolve_alignment_mode. check_alignment_supervision warns when
lambda_align cannot act in the chosen mode.
"""

import torch


def save_checkpoint(state_dict, path, alignment_mode):
    torch.save(
        {"model_state_dict": state_dict, "alignment_mode": alignment_mode},
        path,
    )


def resolve_alignment_mode(checkpoint, config_mode):
    """The mode recorded in the checkpoint wins; config is only a fallback.

    Returns the mode to evaluate with. Warns loudly on disagreement: the
    classifier's alignment-score feature and the eer_thresholds.json
    calibration were produced under the checkpoint's mode, not config's.
    """
    ckpt_mode = checkpoint.get("alignment_mode") if isinstance(checkpoint, dict) else None
    if ckpt_mode is None:
        print(f"Note: checkpoint does not record alignment_mode; using config value '{config_mode}'.")
        return config_mode
    if ckpt_mode != config_mode:
        print(
            f"Warning: checkpoint was trained with alignment_mode='{ckpt_mode}' but config says "
            f"'{config_mode}'. Using '{ckpt_mode}' from the checkpoint. EER thresholds are "
            f"calibrated per mode - re-run test.py under '{ckpt_mode}' config if this is intentional."
        )
    return ckpt_mode


def check_alignment_supervision(alignment_mode, lambda_align):
    """'attn' mode produces no sim_matrix, so the frame-level KL boundary
    loss is skipped and lambda_align is a no-op. Say so at startup instead
    of silently training without boundary supervision.
    """
    if alignment_mode == "attn" and lambda_align > 0:
        print(
            f"Warning: lambda_align={lambda_align} has no effect in alignment_mode='attn' "
            f"(no sim_matrix to supervise) - boundary supervision is inactive this run."
        )
