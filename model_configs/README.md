# MoE model configurations

This directory contains pinned upstream `config.json` snapshots for a range of
public MoE model families. Files retain each source repository's field names and
nesting; they are not normalized to one schema.

`utils/config.py` maps the common top-level and `text_config` variants used by
these files. Upstream model names and configuration data remain subject to their
respective source terms; see [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

The configurations were retrieved on 2026-08-30. Each link is pinned to the exact
Hugging Face repository revision used for the local file.

| Model preset | Local file | Source |
| --- | --- | --- |
| `mixtral-8x7b` | `mixtral-8x7b.json` | [`mistralai/Mixtral-8x7B-v0.1@fc7ac946`](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/fc7ac94680e38d7348cfa806e51218e6273104b0/config.json) |
| `deepseek-v3-671b` | `deepseek-v3.json` | [`deepseek-ai/DeepSeek-V3@e815299b`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) |
| `qwen3-235b-a22b` | `qwen3-235b-a22b.json` | [`Qwen/Qwen3-235B-A22B@8efa6172`](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json) |
| `qwen3-30b-a3b` | `qwen3-30b-a3b.json` | [`Qwen/Qwen3-30B-A3B@ad44e777`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json) |
| `llama-4-maverick-17b-128e` | `llama-4-maverick-17b-128e.json` | [`unsloth/Llama-4-Maverick-17B-128E@764d2884`](https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E/blob/764d2884ed58223ffc67f6904a32d3dc9a6267d7/config.json) |
| `llama-4-scout-17b-16e` | `llama-4-scout-17b-16e.json` | [`unsloth/Llama-4-Scout-17B-16E@62d69dc0`](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E/blob/62d69dc0a79bb9251fe44d1397ddf34fc23e1068/config.json) |
| `kimi-k3` | `kimi-k3.json` | [`moonshotai/Kimi-K3@a590ce09`](https://huggingface.co/moonshotai/Kimi-K3/blob/a590ce090cb049c93a33dfe8c208ec652aa20503/config.json) |
| `deepseek-v4-flash` | `deepseek-v4-flash.json` | [`deepseek-ai/DeepSeek-V4-Flash@60d8d707`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) |
| `glm-5.2` | `glm-5.2.json` | [`zai-org/GLM-5.2@b4734de4`](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json) |
| `gpt-oss-120b` | `gpt-oss-120b.json` | [`openai/gpt-oss-120b@b5c939de`](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json) |

The official Meta Llama 4 repositories require authenticated license access and
returned HTTP 401 during retrieval. Their two entries therefore use public,
unquantized Unsloth base-model mirrors. All other entries come directly from the
model publisher's Hugging Face organization.
