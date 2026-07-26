#!/usr/bin/env python3
"""Export deterministic Qwen3.5 layer truth from pinned Transformers code."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import struct
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

TRANSFORMERS_VERSION = "5.14.1"
TRANSFORMERS_COMMIT = "a08ace4bbd97e721c98751deec37d87b026acadc"
MODELING_SHA256 = "0e2cd8dc50885b2701d26b116c585eedcdc62a24080ec34345af55b963126ded"
SEED = 20260725
MAGIC = b"BRTQ35F1"
VERSION = 1
ENDIAN_MARKER = 0x01020304


def require_pinned_transformers() -> None:
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    source_path = Path(inspect.getsourcefile(modeling_qwen3_5) or "")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != MODELING_SHA256:
        raise RuntimeError(
            f"unexpected modeling_qwen3_5.py SHA-256 {digest}; "
            f"expected {MODELING_SHA256}"
        )


def random_tensor(
    shape: tuple[int, ...], generator: torch.Generator, scale: float = 0.25
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


def assign_random(
    parameter: torch.nn.Parameter,
    generator: torch.Generator,
    scale: float = 0.25,
    base: float = 0.0,
) -> None:
    with torch.no_grad():
        parameter.copy_(random_tensor(tuple(parameter.shape), generator, scale) + base)


def flatten(tensor: torch.Tensor) -> list[float]:
    return (
        tensor.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .view(-1)
        .tolist()
    )


def make_full_attention(
    generator: torch.Generator,
) -> tuple[dict[str, int | float], list[list[float]]]:
    tokens = 4
    hidden_size = 8
    query_heads = 2
    kv_heads = 1
    head_dim = 4
    rotary_dim = 2
    rope_base = 10_000.0
    epsilon = 1.0e-6

    config = Qwen3_5TextConfig(
        hidden_size=hidden_size,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        layer_types=["full_attention"],
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": rope_base,
            "partial_rotary_factor": rotary_dim / head_dim,
        },
    )
    config._attn_implementation = "eager"
    attention = Qwen3_5Attention(config, layer_idx=0).float().eval()
    assign_random(attention.q_proj.weight, generator, scale=0.2)
    assign_random(attention.k_proj.weight, generator, scale=0.2)
    assign_random(attention.v_proj.weight, generator, scale=0.2)
    assign_random(attention.o_proj.weight, generator, scale=0.2)
    assign_random(attention.q_norm.weight, generator, scale=0.15)
    assign_random(attention.k_norm.weight, generator, scale=0.15)

    hidden = random_tensor((1, tokens, hidden_size), generator, scale=0.4)
    position_ids = torch.arange(tokens, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids)
    causal_mask = torch.full((1, 1, tokens, tokens), torch.finfo(torch.float32).min)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    with torch.no_grad():
        projected = attention.q_proj(hidden).view(1, tokens, query_heads, head_dim * 2)
        query, gate = torch.chunk(projected, 2, dim=-1)
        key = attention.k_proj(hidden).view(1, tokens, kv_heads, head_dim)
        value = attention.v_proj(hidden).view(1, tokens, kv_heads, head_dim)
        gate = gate.reshape(1, tokens, hidden_size)
        canonical_input = torch.cat(
            [
                query.reshape(1, tokens, -1),
                key.reshape(1, tokens, -1),
                value.reshape(1, tokens, -1),
                gate,
            ],
            dim=-1,
        )
        expected, _ = attention(
            hidden_states=hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            past_key_values=None,
        )

    config_fields: dict[str, int | float] = {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "position_offset": 0,
        "rope_base": rope_base,
        "epsilon": epsilon,
    }
    vectors = [
        flatten(canonical_input),
        # llama.cpp converts Qwen3.5's offset-form HF RMSNorm parameters to
        # GGUF-native multiplicative scales before execution.
        flatten(1.0 + attention.q_norm.weight),
        flatten(1.0 + attention.k_norm.weight),
        flatten(attention.o_proj.weight.transpose(0, 1)),
        flatten(expected),
    ]
    return config_fields, vectors


def make_gated_delta(
    generator: torch.Generator,
) -> tuple[dict[str, int | float], list[list[float]]]:
    tokens = 4
    hidden_size = 8
    key_heads = 1
    value_heads = 2
    key_dim = 4
    value_dim = 4
    conv_width = 4
    epsilon = 1.0e-6
    conv_dim = 2 * key_heads * key_dim + value_heads * value_dim

    raw_qkv = random_tensor((1, tokens, conv_dim), generator, scale=0.35)
    beta_logits = random_tensor((1, tokens, value_heads), generator, scale=0.3)
    decay_logits = random_tensor((1, tokens, value_heads), generator, scale=0.3)
    gate = random_tensor((1, tokens, hidden_size), generator, scale=0.35)
    conv_weight = random_tensor((conv_dim, conv_width), generator, scale=0.2)
    a_log = torch.log(
        torch.rand((value_heads,), generator=generator, dtype=torch.float32) * 1.5
        + 0.25
    )
    dt_bias = random_tensor((value_heads,), generator, scale=0.25)
    output_norm = Qwen3_5RMSNormGated(value_dim, eps=epsilon).float().eval()
    assign_random(output_norm.weight, generator, scale=0.15, base=1.0)
    initial_convolution = random_tensor(
        (1, conv_dim, conv_width - 1), generator, scale=0.2
    )
    initial_recurrent = random_tensor(
        (1, value_heads, key_dim, value_dim), generator, scale=0.1
    )

    convolution_state = initial_convolution.clone()
    with torch.no_grad():
        convolved = torch_causal_conv1d_update(
            raw_qkv.transpose(1, 2),
            convolution_state,
            conv_weight,
            bias=None,
            activation="silu",
        ).transpose(1, 2)
        query, key, value = torch.split(
            convolved,
            [
                key_heads * key_dim,
                key_heads * key_dim,
                value_heads * value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(1, tokens, key_heads, key_dim)
        key = key.reshape(1, tokens, key_heads, key_dim)
        value = value.reshape(1, tokens, value_heads, value_dim)
        head_index = torch.arange(value_heads, device=query.device) % key_heads
        query = query[:, :, head_index, :]
        key = key[:, :, head_index, :]
        beta = beta_logits.sigmoid()
        decay = -a_log.float().exp() * F.softplus(decay_logits.float() + dt_bias)
        core, final_recurrent = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=decay,
            beta=beta,
            initial_state=initial_recurrent,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        chunk_core, chunk_state = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=decay,
            beta=beta,
            initial_state=initial_recurrent,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(chunk_core, core, atol=1.0e-5, rtol=1.0e-5)
        torch.testing.assert_close(
            chunk_state, final_recurrent, atol=1.0e-5, rtol=1.0e-5
        )
        expected = output_norm(
            core.reshape(-1, value_dim), gate.reshape(-1, value_dim)
        ).reshape(1, tokens, hidden_size)

    canonical_input = torch.cat([raw_qkv, beta_logits, decay_logits, gate], dim=-1)
    config_fields: dict[str, int | float] = {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "conv_width": conv_width,
        "epsilon": epsilon,
    }
    vectors = [
        flatten(canonical_input),
        flatten(conv_weight),
        # llama.cpp stores -exp(A_log) in blk.N.ssm_a.
        flatten(-a_log.exp()),
        flatten(dt_bias),
        flatten(output_norm.weight),
        flatten(initial_convolution),
        flatten(initial_recurrent),
        flatten(expected),
        flatten(convolution_state),
        flatten(final_recurrent),
    ]
    return config_fields, vectors


def pack_vector(values: list[float]) -> bytes:
    return struct.pack("<Q", len(values)) + struct.pack(f"<{len(values)}f", *values)


def export_fixture(output: Path) -> None:
    require_pinned_transformers()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    full, full_vectors = make_full_attention(generator)
    delta, delta_vectors = make_gated_delta(generator)

    body = bytearray()
    body += struct.pack(
        "<7I2f",
        full["tokens"],
        full["hidden_size"],
        full["query_heads"],
        full["kv_heads"],
        full["head_dim"],
        full["rotary_dim"],
        full["position_offset"],
        full["rope_base"],
        full["epsilon"],
    )
    body += struct.pack(
        "<7If",
        delta["tokens"],
        delta["hidden_size"],
        delta["key_heads"],
        delta["value_heads"],
        delta["key_dim"],
        delta["value_dim"],
        delta["conv_width"],
        delta["epsilon"],
    )
    for values in [*full_vectors, *delta_vectors]:
        body += pack_vector(values)

    header_size = struct.calcsize("<8sIIQ")
    total_size = header_size + len(body)
    payload = struct.pack("<8sIIQ", MAGIC, VERSION, ENDIAN_MARKER, total_size) + bytes(
        body
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"wrote {output} ({len(payload)} bytes)")
    print(f"sha256 {hashlib.sha256(payload).hexdigest()}")
    print(f"transformers_commit {TRANSFORMERS_COMMIT}")
    print(f"modeling_sha256 {MODELING_SHA256}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/qwen35/tiny-layer-v1.bin"),
    )
    args = parser.parse_args()
    export_fixture(args.output)


if __name__ == "__main__":
    main()
