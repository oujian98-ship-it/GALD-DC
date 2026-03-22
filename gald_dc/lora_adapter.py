import torch
import torch.nn as nn
from typing import Dict, List, Optional


class LoRALinear(nn.Module):

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA low-rank matrices
        # A: [rank, in_features] - reduce dim
        # B: [out_features, rank] - increase dim
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialization: A with Kaiming, B with zeros (ensures initial LoRA output is 0)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # [Fix] Ensure new parameters are on the correct device
        device = original_layer.weight.device
        self.lora_A.data = self.lora_A.data.to(device)
        self.lora_B.data = self.lora_B.data.to(device)

        # Dropout (optional)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer output (frozen weights)
        original_output = self.original_layer(x)

        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        return original_output + lora_output * self.scaling

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Return LoRA trainable parameters"""
        return [self.lora_A, self.lora_B]


class LoRAAdapter:
    def __init__(self, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_layers: Dict[str, LoRALinear] = {}

    def inject_lora_into_model(
        self, model: nn.Module, target_modules: Optional[List[str]] = None
    ) -> nn.Module:
        if target_modules is None:
            # Default targets: Key linear layers in UNet
            # Specific to UNet_conditional architecture in ddpm_conditional.py
            target_modules = [
                "time_mlp",  # Time embedding MLP
                "fc_256in",  # Input projection
                "final_out256",  # Output projection
                "mlp",  # General MLP
                "1",  # Match Sequential Linear layers
                "3",
            ]

        modules_to_replace = []

        # Find target modules by iterating through the model
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(target in name for target in target_modules):
                    modules_to_replace.append((name, module))

        # Replace target modules with LoRA versions
        for name, module in modules_to_replace:
            lora_layer = LoRALinear(
                module, rank=self.rank, alpha=self.alpha, dropout=self.dropout
            )
            self.lora_layers[name] = lora_layer

            # Replace via parent module
            self._replace_module(model, name, lora_layer)

        return model

    def _replace_module(
        self, model: nn.Module, target_name: str, new_module: nn.Module
    ):
        parts = target_name.split(".")
        parent = model

        # Navigate to parent module
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        # Replace target module
        final_name = parts[-1]
        if final_name.isdigit():
            parent[int(final_name)] = new_module
        else:
            setattr(parent, final_name, new_module)

    def get_lora_parameters(self) -> List[nn.Parameter]:
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend(lora_layer.get_lora_parameters())
        return params

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            state_dict[f"{name}.lora_A"] = lora_layer.lora_A
            state_dict[f"{name}.lora_B"] = lora_layer.lora_B
        return state_dict

    def count_parameters(self) -> Dict[str, int]:
        lora_params = sum(p.numel() for p in self.get_lora_parameters())
        return {"lora_parameters": lora_params, "lora_layers": len(self.lora_layers)}


def apply_lora_to_diffusion_model(
    diffusion_model: nn.Module, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0
) -> tuple:
    adapter = LoRAAdapter(rank=rank, alpha=alpha, dropout=dropout)

    if hasattr(diffusion_model, "model"):
        adapter.inject_lora_into_model(diffusion_model.model)
    else:
        adapter.inject_lora_into_model(diffusion_model)

    lora_params = adapter.get_lora_parameters()
    param_count = adapter.count_parameters()

    print(
        f">>> [LoRA] Injected {param_count['lora_layers']} layers, "
        f"{param_count['lora_parameters']:,} trainable parameters"
    )

    return adapter, lora_params
