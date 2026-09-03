"""Qwen3.5 instruct models.

``Qwen/Qwen3.5-4B`` *is* the instruct variant -- this generation has no
``-Instruct`` suffix, and ``Qwen/Qwen3.5-4B-Base`` is the pretrained one.  The
longest-prefix rule in :func:`src.models.profile.resolve` means a ``-Base``
profile could be added later without disturbing this one.
"""

from __future__ import annotations

from src.models.profile import ModelProfile

QWEN3_5 = ModelProfile(
    match="Qwen/Qwen3.5-",
    torch_dtype="bfloat16",
    lora_target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "out_proj",
    ),
    prompt_format="chat",
    chat_template_sha="a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    chat_template_kwargs={},  # {} == the template's own defaults == thinking ON
    # The template ends "<|im_start|>assistant\n<think>\n"; <think> is a
    # registered token, so the cut lands on an atomic boundary and moves the
    # trailing "\n" into the completion.  This is exactly what the deleted
    # heuristic found by accident: a template ending in a bare
    # "<|im_start|>assistant\n" has no atomic token after "assistant" to land on,
    # and lost that word silently.
    prompt_end_token="<think>",
    # Both moved here from scripts/smoke_base_model.py, which hardcoded them as
    # Qwen literals; the reasoning for each is in the notes below.
    expected_lora_params=12_386_304,
    excluded_lora_modules=("in_proj_a", "in_proj_b"),
    notes="""Hybrid attention: 24 linear-attention layers and 8 full-attention
    layers in a 3:1 pattern (full at indices 3, 7, 11, ... 31).  The repo's
    default q/k/v/o target list matches ONLY the 8 full-attention layers -- PEFT
    matches by name suffix and errors only on zero matches, so that would adapt a
    quarter of the depth without warning and record an identical target list in
    adapter_config.json.  Adding in_proj_qkv / in_proj_z / out_proj reaches all
    24 linear-attention layers, giving 32/32 coverage.

    in_proj_a and in_proj_b are deliberately excluded: out_features is 32, so a
    rank-16 adapter is a near-full-rank reparameterization of a tiny gate vector
    -- roughly 2.0M parameters for little expressive gain.

    Trainable at r=16: 12,386,304.  The Llama suite's adapters are 13.6M, so the
    two suites land within ~10% on adapter capacity, which is the comparability
    argument for this particular target set.  (An earlier estimate of 11.86M
    assumed q_proj is 2560->4096; it is 2560->8192, because attn_output_gate is
    true and q_proj emits the query and its gate together.)

    dtype is bfloat16, not the repo default float16.  The checkpoint is natively
    bf16 and declares mamba_ssm_dtype float32; the linear-attention path carries
    a recurrent state accumulated over the sequence, and fp16's 10-bit mantissa
    and +/-65504 range are a poor fit for a running state.

    chat_template_sha pins the template that ships with the model.  It is
    published twice -- chat_template.jinja and the chat_template key in
    tokenizer_config.json -- and the two are byte-identical (7756 bytes).""",
)
