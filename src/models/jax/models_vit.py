"""Vision Transformer (Flax) compatible con los checkpoints AugReg de Google.

Implementación clean-room en `src/models/jax/`. Sin soporte de hybrid (resnet)
ni `representation_size` (no necesario para los checkpoints AugReg estándar).

Los nombres de submódulos (`Transformer`, `posembed_input`, `encoderblock_{i}`,
`encoder_norm`, `LayerNorm_0`, `MultiHeadDotProductAttention_0`, `LayerNorm_1`,
`MlpBlock_0`, `embedding`, `cls`, `head`) coinciden 1:1 con los del repo
oficial de Google para que los `.npz` se puedan cargar sin renombrado.

Cada bloque del encoder hace `self.sow('intermediates', 'block_out', x, ...)`
para que `model.apply(..., capture_intermediates=True, mutable=['intermediates'])`
devuelva las features por capa en `state['intermediates']['Transformer']['block_out']`.
Igualmente, `nn.MultiHeadDotProductAttention(..., sow_weights=True)` expone los
pesos de atención por bloque, usados para el rollout.
"""

from typing import Optional

import jax.numpy as jnp
from flax import linen as nn


class _IdentityLayer(nn.Module):
    """Identidad nombrada (replica el `pre_logits` del repo upstream)."""

    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    """Embeddings posicionales aprendidos."""

    @nn.compact
    def __call__(self, inputs):
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden), got {inputs.shape}"
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param(
            "pos_embedding", nn.initializers.normal(stddev=0.02), pos_emb_shape
        )
        return inputs + pe


class MlpBlock(nn.Module):
    mlp_dim: int
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs, *, deterministic: bool):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dense(
            features=actual_out_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )(x)
        return x


class Encoder1DBlock(nn.Module):
    mlp_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, inputs, *, deterministic: bool):
        x = nn.LayerNorm()(inputs)
        x = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            num_heads=self.num_heads,
        )(x, x, sow_weights=True)
        x = x + inputs

        y = nn.LayerNorm()(x)
        y = MlpBlock(mlp_dim=self.mlp_dim)(y, deterministic=deterministic)
        return x + y


class Encoder(nn.Module):
    num_layers: int
    mlp_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, *, train: bool):
        assert x.ndim == 3
        x = AddPositionEmbs(name="posembed_input")(x)

        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                name=f"encoderblock_{lyr}",
            )(x, deterministic=not train)
            # Snapshot output of every block so adapter puede reconstruir features
            # por capa sin reimplementar el encoder.
            self.sow(
                "intermediates",
                "block_out",
                x,
                init_fn=lambda: (),
                reduce_fn=lambda acc, v: acc + (v,),
            )

        return nn.LayerNorm(name="encoder_norm")(x)


class VisionTransformer(nn.Module):
    """Vision Transformer (token classifier)."""

    num_classes: int
    patches_size: tuple[int, int]
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    classifier: str = "token"

    @nn.compact
    def __call__(self, inputs, *, train: bool = False):
        assert inputs.ndim == 4, f"Expected NHWC input, got {inputs.shape}"
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches_size,
            strides=self.patches_size,
            padding="VALID",
            name="embedding",
        )(inputs)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        if self.classifier in ("token", "token_unpooled"):
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c))
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        x = Encoder(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            name="Transformer",
        )(x, train=train)

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))
        elif self.classifier in ("unpooled", "token_unpooled"):
            pass
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        x = _IdentityLayer(name="pre_logits")(x)

        if self.num_classes:
            x = nn.Dense(
                features=self.num_classes,
                name="head",
                kernel_init=nn.initializers.zeros,
            )(x)
        return x
