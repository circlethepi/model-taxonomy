#!/usr/bin/env python
"""Prove a new base model works before spending GPU-hours on a suite.

Adding a base model to the pipeline has one failure mode worth real fear: the
checkpoint loads, training runs, the loss looks plausible, and the decoder was
randomly initialized the whole time.  ``from_pretrained`` reports missing keys
and then carries on, so nothing downstream would notice.  Test 2 below is that
check, and it is the reason this script exists.

The rest is ordered so each stage only runs if the one it depends on passed, and
so the two decisions that change what gets submitted -- does LoRA reach every
layer, do generations finish inside the token budget -- are answered before the
shards go out rather than after.  Stage 1b needs no GPU and therefore runs first.

Nothing here names a model.  Per-checkpoint *choices* (the expected LoRA count,
the modules that must stay unadapted) are read from the resolved ModelProfile;
everything else is derived from the tokenizer and config, so a model with a
reasoning block is instrumented for closure and one without is instrumented for
termination, with no flag to set either way.

This is a *chat-suite* pre-flight: stage 4 asserts completion-only loss and a
recorded chat_template_sha, so it does not apply to a raw suite.

Runs the *real* code paths (``finetune_all``, ``BehavioralTaxonomy``) rather than
reimplementing them, because a smoke test that exercises a parallel
implementation proves nothing about the one that will run.

Usage:
    python scripts/smoke_base_model.py experiments/simplex3_qwen/train_shard0.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def stage(name: str):
    """Run one stage, record pass/fail, and keep going where that is useful."""
    def wrap(fn):
        def run(*a, **kw):
            print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}", flush=True)
            try:
                detail = fn(*a, **kw) or ""
                RESULTS.append((name, True, detail))
                print(f"  [PASS] {detail}", flush=True)
                return True
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  [FAIL] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                return False
        return run
    return wrap


@stage("1b. the chat template renders the same prompt twice")
def t_template(cfg_path: Path, token):
    """A template that interpolates anything mutable silently breaks the pin.

    ``chat_template_sha`` hashes the *template*, not its output.  Llama-3.1's
    template writes a "Cutting Knowledge Date"/"Today Date" header, so a template
    of that shape can render different training prompts on different days while
    the pin still matches -- adapters that are no longer comparable, with nothing
    in any config to show it.  Rendering one row twice is the cheapest check that
    catches it, and it needs no GPU, so it runs before any weights are held.

    Also asserts the profile actually describes the tokenizer, which is the
    ``assert_compatible`` contract, for the same fail-early reason.
    """
    from transformers import AutoTokenizer

    import src.datasets._chat_projection as cp
    from src.models.profile import assert_compatible, resolve

    cfg = yaml.safe_load(cfg_path.read_text())
    base_model = cfg["base_models"][0]
    tok = AutoTokenizer.from_pretrained(base_model, token=token,
                                        trust_remote_code=True)
    profile = resolve(base_model)
    assert_compatible(profile, tok)

    fmt = cp.PromptFormat.from_config(cfg.get("prompt_format"))
    if fmt.format != "chat":
        # Nothing to render twice, and no pin to defend.  Reported rather than
        # failed: a raw suite is a legitimate configuration, it just has no
        # template for this stage to be about.
        return f"raw suite ({profile.match!r}); no chat template to check"

    row = {"question_title": "Why is the sky blue?", "question_content": "curious"}
    first = cp.render_prompt(tok, row, fmt)
    second = cp.render_prompt(tok, row, fmt)
    assert first == second, (
        "the chat template rendered two different prompts for one row; something "
        "mutable (a date, a uuid, a clock) is interpolated into it, so the "
        "chat_template_sha pin does not actually pin the prompts"
    )
    return (f"profile {profile.match!r} matches the tokenizer; "
            f"prompt is {len(first)} chars and stable across renders")


@stage("2. checkpoint loads with every weight it should have")
def t_load(base_model: str, dtype, token):
    """The one that could force a structural change to how the model is loaded.

    Qwen3.5 ships as ``Qwen3_5ForConditionalGeneration`` while the auto-factory
    builds the text-only ``Qwen3_5ForCausalLM``, so the stored weights may carry a
    ``model.language_model.`` prefix the causal-LM class does not look for.
    Unexpected *vision* keys are fine -- we are deliberately not loading a tower.
    Missing keys are not.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model, info = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, token=token,
        trust_remote_code=True, output_loading_info=True,
    )
    missing = [k for k in info.get("missing_keys", [])]
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    assert not missing, f"{len(missing)} missing key(s), first few: {missing[:5]}"
    assert n_params > 4.0e9, f"only {n_params/1e9:.2f}B params -- decoder did not load"

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        out = model(ids.to(model.device), output_hidden_states=True)
    n_hidden = len(out.hidden_states)
    assert n_hidden == n_layers + 1, f"{n_hidden} hidden states for {n_layers} layers"

    del model
    torch.cuda.empty_cache()
    return (f"{n_params/1e9:.2f}B params, {n_layers} layers, {n_hidden} hidden states, "
            f"0 missing keys ({len(info.get('unexpected_keys', []))} unexpected)")


@stage("3. LoRA attaches to every layer, and only where intended")
def t_lora(base_model: str, dtype, token, targets, rank, expect_params, excluded):
    """PEFT matches ``target_modules`` by name *suffix* and errors only on ZERO
    matches, so a target list that reaches a quarter of the depth is silent --
    and ``adapter_config.json`` records the same list either way.  Count the
    layers actually touched rather than trusting the list.
    """
    import re

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, token=token, trust_remote_code=True,
    )
    n_layers = model.config.num_hidden_layers
    torch.manual_seed(0)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=32, target_modules=list(targets),
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    ))

    adapted, layers = [], set()
    for name, _ in model.named_modules():
        if name.endswith("lora_A.default"):
            adapted.append(name)
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                layers.add(int(m.group(1)))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert len(layers) == n_layers, f"adapters in {len(layers)} of {n_layers} layers"
    # Which modules must stay unadapted is a per-checkpoint *choice*, so it comes
    # from the profile rather than being spelled here: Qwen3.5 excludes
    # in_proj_a/in_proj_b deliberately, and a uniform-attention model excludes
    # nothing.  An empty tuple therefore checks nothing, which is correct.
    for name_ in excluded:
        stray = [a for a in adapted if a.split(".lora_A")[0].endswith(name_)]
        assert not stray, f"{name_} was matched ({len(stray)} modules)"
    tower = [a for a in adapted if "visual" in a or "vision" in a or "audio" in a]
    assert not tower, f"a non-text tower was matched: {tower[:3]}"
    # None means "report, do not assert": a family-wide profile covering several
    # widths cannot name one true count, and a wrong constant is worse than none.
    if expect_params is None:
        print(f"  (no expected_lora_params for this profile; observed {trainable:,})",
              flush=True)
    else:
        assert trainable == expect_params, (
            f"trainable={trainable:,}, expected {expect_params:,}. The plan's sizing "
            f"and the cross-suite capacity comparison both use this number."
        )

    del model
    torch.cuda.empty_cache()
    return (f"{trainable:,} trainable across {len(layers)}/{n_layers} layers, "
            f"{len(adapted)} adapted modules")


@stage("4. a short real training run supervises the completion only")
def t_train(cfg_path: Path, scratch: Path, token):
    """Runs ``finetune_all`` itself, against an isolated cache_dir.

    Loss ~0.0 on step 1 would mean every label is -100 -- a mask that supervises
    nothing trains silently and produces a plausible-looking adapter.
    """
    import torch

    from scripts.finetune_lora import finetune_all

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["cache_dir"] = str(scratch / "cache")
    cfg["output_dir"] = str(scratch / "out")
    first = cfg["fine_tuning"]["datasets"][0]
    cfg["fine_tuning"]["datasets"] = [first]
    cfg["datasets"] = [d for d in cfg["datasets"] if d["name"] == first]
    cfg["fine_tuning"]["total_train_samples"] = 80
    cfg["fine_tuning"]["n_samples"] = 200

    # Recipes live under output_dir and nothing else writes them here.
    from scripts.build_datasets import build_datasets
    build_datasets(cfg)

    written = finetune_all(cfg, force=True)
    assert written, "finetune_all wrote no adapter"
    meta = json.loads((written[0] / "experiment_meta.json").read_text())

    assert meta["completion_only_loss"] is True, "run trained the full sequence"
    assert meta["prompt_format"].get("format") == "chat", meta["prompt_format"]
    assert meta["prompt_format_id"], "no prompt_format_id recorded"
    assert meta["chat_template_sha"], "no chat_template_sha recorded"
    assert meta["truncation"] is not None, "no truncation record"
    assert written[0].name.endswith(f"_f{meta['prompt_format_id']}"), (
        f"adapter dir {written[0].name} does not carry its prompt format"
    )

    torch.cuda.empty_cache()
    return (f"adapter={written[0].name}; completion_only_loss=True; "
            f"template={meta['chat_template_sha'][:12]}; "
            f"truncation={meta['truncation']}")


@stage("5. generation closes its reasoning block inside the token budget")
def t_generate(cfg_path: Path, scratch: Path, adapter: Path, token):
    """THE decision point, surfaced before eight behavioral shards run.

    If closure rate is near zero the behavioral level is embedding reasoning
    traces rather than answers, which is a different construct from the one the
    raw-prompted suites measured.  The fix costs no retraining -- re-run
    behavioral with ``chat_template_kwargs: {enable_thinking: false}``, which
    changes the format id and therefore lands in its own cache entry.
    """
    import torch

    from scripts._utils import make_behavioral_taxonomy
    from src.taxonomy.behavioral import _think_close_token_id
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["cache_dir"] = str(scratch / "cache")
    cfg["output_dir"] = str(scratch / "out")
    cfg.setdefault("extraction", {})
    cfg["extraction"]["batch_size"] = 2
    cfg["extraction"]["taxonomies"] = {
        "behavioral": {"max_new_tokens": 128, "replicates": 2, "do_sample": True,
                       "temperature": 1.0, "top_p": 1.0, "generation_seed": 0}
    }

    tok = AutoTokenizer.from_pretrained(cfg["base_models"][0], token=token,
                                        trust_remote_code=True)
    # Branch on the derived value rather than requiring one, exactly as
    # behavioral.py does: a model with no reasoning block is not a broken model,
    # it is a different class of model, and the stage's question -- is this level
    # embedding answers, or something else? -- is the same for both.
    think_id = _think_close_token_id(tok)

    # The value that will actually be used at inference, which means running the
    # pipeline's own rule rather than reading the raw field.  Checkpoints commonly
    # ship no pad token at all; apply_pad_token then takes one from the profile if
    # the profile names one and falls back to eos if it does not -- and pad == eos
    # is precisely the condition this assertion exists to forbid.  Reading
    # pad_token_id raw gets both directions wrong: vacuous on a checkpoint that
    # declares nothing, and spuriously failing on one whose profile supplies a pad.
    from src.models.profile import apply_pad_token, resolve

    pad_source = apply_pad_token(tok, resolve(cfg["base_models"][0]))
    assert tok.pad_token_id != tok.eos_token_id, (
        f"effective pad ({tok.pad_token_id}) == eos ({tok.eos_token_id}); padding "
        f"and a real end-of-turn are indistinguishable -- {pad_source}"
    )

    import src.datasets._chat_projection as cp
    fmt = cp.PromptFormat.from_config(cfg.get("prompt_format"))
    rows = [{"question_title": "Why is the sky blue?", "question_content": ""},
            {"question_title": "How do I boil an egg?", "question_content": ""}]
    queries = [cp.render_prompt(tok, r, fmt) for r in rows]

    tax = make_behavioral_taxonomy(cfg, queries, query_key=None, cache=None)
    rep = tax.extract(str(adapter))
    closure = rep.metadata.get("think_closure")
    texts = rep.metadata["generated_texts"]

    flat = [t for per_query in texts for t in per_query]
    n_empty = sum(1 for t in flat if not t.strip())

    torch.cuda.empty_cache()
    print(f"  pad: {pad_source}", flush=True)
    print(f"\n  --- sample generation ---\n  {texts[0][0][:300]!r}\n", flush=True)

    if think_id is not None:
        assert closure is not None, "think_closure was not recorded"
        assert closure["token_id"] == think_id
        assert n_empty == 0, f"{n_empty}/{len(flat)} generations empty after the split"
        return (f"closure_rate={closure['closure_rate']:.2f} "
                f"median_close_index={closure['median_close_index']} "
                f"of max_new_tokens={closure['max_new_tokens']}; "
                f"{n_empty}/{len(flat)} empty after the split")

    # No reasoning block.  The equivalent property is that generations are real
    # answers rather than a budget's worth of anything: non-empty, and *finished*
    # inside max_new_tokens rather than truncated at it.  Termination rate is
    # reported where closure rate is reported, for the same reason -- it is a
    # validity statistic for the level, not a performance metric.
    assert closure is None, "think_closure recorded for a model with no </think>"
    assert n_empty == 0, f"{n_empty}/{len(flat)} generations empty"

    budget = cfg["extraction"]["taxonomies"]["behavioral"]["max_new_tokens"]
    n_tok = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in flat]
    terminated = sum(1 for n in n_tok if n < budget)
    rate = terminated / len(flat) if flat else 0.0
    assert rate > 0.0, (
        f"every generation ran the full {budget}-token budget; the behavioral "
        f"level would be embedding truncated text rather than answers"
    )
    return (f"no reasoning block; termination_rate={rate:.2f} "
            f"({terminated}/{len(flat)} finished inside max_new_tokens={budget}), "
            f"median length {sorted(n_tok)[len(n_tok) // 2]}; "
            f"{n_empty}/{len(flat)} empty")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="A train shard YAML for the suite.")
    ap.add_argument("--scratch", type=Path,
                    default=Path("/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/"
                                 "results/_smoke"))
    ap.add_argument("--expect-lora-params", type=int, default=None,
                    help="Override the profile's expected_lora_params for a one-off.")
    ap.add_argument("--keep", action="store_true", help="Do not delete the scratch tree.")
    args = ap.parse_args()

    from scripts._utils import hf_token, parse_dtype
    from src.models.profile import resolve

    cfg = yaml.safe_load(args.config.read_text())
    base_model = cfg["base_models"][0]
    ft = cfg["fine_tuning"]
    dtype = parse_dtype(ft.get("torch_dtype", "float16"))
    token = hf_token(cfg)

    # The two facts that are neither derivable from the checkpoint nor properties
    # of this run come from the profile; --expect-lora-params stays as a one-off
    # override.
    profile = resolve(base_model)
    expect_params = (args.expect_lora_params if args.expect_lora_params is not None
                     else profile.expected_lora_params)

    print(f"base model : {base_model}")
    print(f"profile    : {profile.match!r}")
    print(f"dtype      : {dtype}")
    print(f"targets    : {ft.get('target_modules')}")
    print(f"expect LoRA: {expect_params if expect_params is not None else 'report only'}")
    print(f"excluded   : {profile.excluded_lora_modules or '(none)'}")
    print(f"scratch    : {args.scratch}")

    if args.scratch.exists():
        shutil.rmtree(args.scratch)
    args.scratch.mkdir(parents=True)

    # Runs first: it needs no GPU, so a bad template or a mismatched profile
    # fails before any weights are held.  Not gated on -- a raw suite has no
    # template to render, and the stages below are still worth running.
    t_template(args.config, token)

    ok = t_load(base_model, dtype, token)
    if ok:
        ok = t_lora(base_model, dtype, token, ft["target_modules"],
                    ft["lora_rank"], expect_params, profile.excluded_lora_modules)
    trained = None
    if ok:
        # Deliberately not gated on test 3: a wrong LoRA count is a sizing
        # question, not a correctness one, and the loss-mask answer is worth
        # having either way.
        if t_train(args.config, args.scratch, token):
            adapters = list((args.scratch / "cache" / "03_adapters").rglob("adapter_config.json"))
            trained = adapters[0].parent if adapters else None
    if trained is not None:
        t_generate(args.config, args.scratch, trained, token)

    print(f"\n{'=' * 72}\n  SUMMARY\n{'=' * 72}")
    for name, passed, detail in RESULTS:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}\n         {detail}")
    n_fail = sum(1 for _, p, _ in RESULTS if not p)
    print(f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed")

    if not args.keep and args.scratch.exists():
        shutil.rmtree(args.scratch)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
