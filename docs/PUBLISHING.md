# Publishing

Molt is published in four places. All four are generated from
`benchmarks/results/summary.json`, so refresh that first (`make bench`) and then
regenerate before uploading — nothing here is hand-maintained.

| where | what | why it exists |
|---|---|---|
| [github.com/NagaYu/molt](https://github.com/NagaYu/molt) | the code | the runtime and everything needed to reproduce |
| [Space `NagaYu/molt`](https://huggingface.co/spaces/NagaYu/molt) | interactive replay + results | the fastest path from "what is this" to "oh, I see" |
| [Model `NagaYu/molt-kv-projectors-qwen2.5`](https://huggingface.co/NagaYu/molt-kv-projectors-qwen2.5) | the six fitted maps | the actual novel artefact; reusable without rerunning the fit |
| [Dataset `NagaYu/molt-benchmark-results`](https://huggingface.co/datasets/NagaYu/molt-benchmark-results) | every measurement | so the claims can be checked instead of believed |

## Refresh everything

```bash
make bench          # re-measure (slow)
make figures        # figures + README table + report + Space page
git add -A && git commit && git push
```

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(repo_id="NagaYu/molt", repo_type="space",
                  folder_path="artifacts/space")
api.upload_folder(repo_id="NagaYu/molt-kv-projectors-qwen2.5", repo_type="model",
                  folder_path="artifacts/projectors")
EOF
```

The dataset payload (flat CSVs for the viewer plus the raw JSON) is assembled by
`scripts/build_hf_dataset.py`.

## Two things that will bite you

- **The Space must be created with `space_sdk="static"`.** `hf upload` on a
  non-existent Space auto-creates a *Gradio* one, and Gradio Spaces on free
  hardware require PRO — the upload then fails with a subscription error that
  does not mention the SDK at all. A live-inference Space would need paid
  hardware anyway: ~6 GB of weights and a 1.5B forward pass is a multi-minute
  cold start for something you watch for twelve seconds.
- **The `hf` CLI's `upload` was unreliable here**; `HfApi().upload_folder()`
  worked every time. Prefer the Python API in scripts.
