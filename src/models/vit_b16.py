import math
from typing import Any, Optional

import timm
import torch

from ..models.base import BaseModelAdapter
from ..utils.common_enums import BackendEnum
from ..utils.common_types import ArrayLike


class TimmViTAdapter(BaseModelAdapter):
    """Clase base para adaptadores de Vision Transformers usando timm.

    Usa get_intermediate_layers de timm para extraer features intermedias
    de forma consistente y mantenible.

    Subclases solo necesitan definir:
        - MODEL_NAME: nombre del modelo en timm
        - DISPLAY_NAME: nombre para mostrar/logs
    """

    MODEL_NAME: str  # Debe ser definido por subclases
    DISPLAY_NAME: str  # Debe ser definido por subclases

    def __init__(self, config: dict[str, Any] = {}):
        super().__init__(self.DISPLAY_NAME, BackendEnum.TORCH, config)
        self.transform = timm.data.create_transform(
            **timm.data.resolve_data_config({}, model=self.MODEL_NAME),
            is_training=False,
        )
        # Configuración para attention rollout
        self.head_fusion = "max"
        self.discard_ratio = 0.95
        self._EPS = 1e-8

    def _load_model(self, config: dict[str, Any]) -> Any:
        model = timm.create_model(self.MODEL_NAME, pretrained=True)
        model.eval()
        return model.to(self.device)

    def forward_features_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> ArrayLike:
        """Ejecuta el modelo y retorna las features por capa.

        Usa get_intermediate_layers con reshape=False y return_prefix_tokens=True
        para obtener formato [B, num_tokens, hidden_dim] (CLS + patches).

        Args:
            batch: Tensor de entrada [B, C, H, W]
            layers: Lista de índices de capas a extraer. None = todas.

        Returns:
            Lista de tensores [B, num_tokens, hidden_dim] por cada capa.
        """
        with torch.no_grad():
            x = batch.to(self.device)

            # Determinar qué capas extraer (por defecto todas)
            num_blocks = len(self.model.blocks)
            indices = list(range(num_blocks)) if layers is None else layers

            # Usar API nativa de timm
            intermediates = self.model.get_intermediate_layers(
                x,
                indices,
                reshape=False,  # Mantener formato [B, N, C]
                return_prefix_tokens=True,  # Devolver CLS token separado
                norm=False,  # Sin normalización en capas intermedias
            )

            # Concatenar [CLS, patches] -> [B, num_tokens, hidden_dim]
            hidden_states = [
                torch.cat([prefix, spatial], dim=1) for spatial, prefix in intermediates
            ]

            return hidden_states

    def forward_saliency_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> list[torch.Tensor]:
        """Ejecuta el modelo y retorna attention rollout maps por capa.

        Usa hooks para capturar attention weights durante el forward pass normal,
        sin reimplementar el transformer.

        Args:
            batch: Tensor de entrada [B, C, H, W]
            layers: Lista de índices de capas. Si es None, usa todas las capas.

        Returns:
            Lista de tensores [B, grid_size, grid_size], uno por cada capa solicitada.
            Cada tensor representa el rollout attention acumulado hasta esa capa.
        """
        with torch.no_grad():
            x = batch.to(self.device)

            # Determinar qué capas procesar
            num_blocks = len(self.model.blocks)
            target_layers = list(range(num_blocks)) if layers is None else layers

            # Variables compartidas entre hooks (via closure)
            rollouts_per_layer = []
            result = None  # Se inicializará en el primer hook
            identity = None
            protect_mask_flat = None
            k = None

            def create_hook(block_idx, attn_module):
                """Crea un hook que captura input y calcula attention weights."""

                def hook_fn(module, input, output):
                    nonlocal result, identity, protect_mask_flat, k

                    # El input[0] es x_norm (ya normalizado por block.norm1)
                    x_norm = input[0]
                    B, N, C = x_norm.shape

                    # Inicializar en el primer bloque
                    if result is None:
                        identity = torch.eye(N, device=x.device, dtype=x.dtype)
                        result = identity.unsqueeze(0).expand(B, -1, -1).clone()

                        # Configurar máscara de protección para CLS token
                        protect_mask = torch.zeros(
                            N, N, device=x.device, dtype=torch.bool
                        )
                        protect_mask[0, :] = True
                        protect_mask[:, 0] = True
                        valid_entries = int((~protect_mask).sum().item())
                        total_entries = N * N
                        k = 0
                        if self.discard_ratio > 0 and valid_entries > 0:
                            k = min(
                                int(total_entries * self.discard_ratio), valid_entries
                            )
                        protect_mask_flat = protect_mask.view(-1)

                    # Calcular attention weights usando los pesos del módulo
                    qkv = attn_module.qkv(x_norm)
                    qkv = qkv.reshape(
                        B, N, 3, attn_module.num_heads, attn_module.head_dim
                    )
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k_proj, v = qkv.unbind(0)

                    # Scaled dot-product attention
                    attn_weights = (q * attn_module.scale) @ k_proj.transpose(-2, -1)
                    attn_weights = attn_weights.softmax(dim=-1)
                    # No aplicamos dropout ya que estamos en eval mode

                    # Fusión de heads
                    if self.head_fusion == "mean":
                        fused = attn_weights.mean(dim=1)
                    elif self.head_fusion == "max":
                        fused = attn_weights.max(dim=1).values
                    elif self.head_fusion == "min":
                        fused = attn_weights.min(dim=1).values
                    else:
                        raise ValueError(f"Unsupported head fusion: {self.head_fusion}")

                    # Aplicar descarte de atenciones bajas (si configurado)
                    if k > 0:
                        flat = fused.reshape(B, -1)
                        protected = flat.masked_fill(
                            protect_mask_flat.unsqueeze(0), float("inf")
                        )
                        _, indices_to_drop = torch.topk(
                            protected, k, dim=1, largest=False
                        )
                        flat_modified = flat.clone()
                        flat_modified.scatter_(1, indices_to_drop, 0.0)
                        fused = flat_modified.view(B, N, N)

                    # Aplicar residual connection y normalizar
                    fused = (fused + identity) * 0.5
                    fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(self._EPS)

                    # Acumular rollout
                    result = torch.bmm(fused, result)

                    # Si esta capa está en target_layers, guardar el rollout actual
                    if block_idx in target_layers:
                        # Extraer attention del CLS token a los patches (excluir CLS)
                        masks = result[:, 0, 1:]  # [B, seq_len-1]
                        grid_size = int(math.sqrt(masks.shape[-1]))
                        masks_2d = masks.view(B, grid_size, grid_size)
                        # Normalizar
                        masks_normalized = masks_2d / masks_2d.amax(
                            dim=(1, 2), keepdim=True
                        ).clamp_min(self._EPS)
                        rollouts_per_layer.append(masks_normalized.clone())

                return hook_fn

            # Registrar hooks en cada bloque de atención
            hooks = []
            for idx, block in enumerate(self.model.blocks):
                hook = block.attn.register_forward_hook(create_hook(idx, block.attn))
                hooks.append(hook)

            # Ejecutar forward pass normal del modelo (los hooks se disparan automáticamente)
            _ = self.model(x)

            # Limpiar hooks
            for hook in hooks:
                hook.remove()

            return rollouts_per_layer


# =============================================================================
# Modelos específicos - Solo definen MODEL_NAME y DISPLAY_NAME
# =============================================================================


class ViT_B_16(TimmViTAdapter):
    """ViT-Base/16 entrenado con CLIP en LAION-2B."""

    MODEL_NAME = "vit_base_patch16_clip_224.laion2b"
    DISPLAY_NAME = "ViT-B/16"


class ViT_B_32(TimmViTAdapter):
    """ViT-Base/32 entrenado con CLIP en LAION-2B."""

    MODEL_NAME = "vit_base_patch32_clip_224.laion2b"
    DISPLAY_NAME = "ViT-B/32"


class ViT_L_14(TimmViTAdapter):
    """ViT-Large/14 entrenado con CLIP en LAION-2B."""

    MODEL_NAME = "vit_large_patch14_clip_224.laion2b"
    DISPLAY_NAME = "ViT-L/14"


class ViT_H_14(TimmViTAdapter):
    """ViT-Huge/14 entrenado con CLIP en LAION-2B."""

    MODEL_NAME = "vit_huge_patch14_clip_224.laion2b"
    DISPLAY_NAME = "ViT-H/14"


class DynamicTimmModelAdapter(TimmViTAdapter):
    """Adaptador genérico para modelos de timm declarados dinámicamente."""

    MODEL_NAME = ""
    DISPLAY_NAME = ""

    def __init__(self, timm_model_name: str, config: dict[str, Any] | None = None):
        self.MODEL_NAME = timm_model_name
        self.DISPLAY_NAME = timm_model_name
        super().__init__(config or {})
