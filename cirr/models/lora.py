"""Low-Rank Adaptation (LoRA) implementation for parameter-efficient fine-tuning.

LoRA allows for efficient adaptation of pre-trained models by injecting trainable
rank decomposition matrices into frozen model layers.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation.

    Instead of fine-tuning W directly, we freeze W and learn:
        h = W_0 x + (B A) x
    where:
        - W_0 is frozen (original weights)
        - A ∈ R^{r x d_in}, B ∈ R^{d_out x r} are trainable (r << min(d_in, d_out))
        - r is the rank hyperparameter controlling adaptation capacity

    Parameters
    ----------
    original_layer : nn.Linear
        The original linear layer to adapt.
    rank : int
        Rank of the low-rank decomposition. Lower rank = fewer parameters.
        Typical values: 1-16 for extreme efficiency, 32-128 for moderate adaptation.
    alpha : float, optional
        Scaling factor for LoRA outputs. Helps with hyperparameter tuning across ranks.
        By default 1.0.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        self.weight = original_layer.weight
        self.weight.requires_grad = False

        # Handle bias
        if original_layer.bias is not None:
            self.bias = original_layer.bias
            self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)

        # LoRA trainable parameters
        # A: projects input to low-rank space (in_features -> rank)
        # B: projects from low-rank space to output (rank -> out_features)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialize A with Kaiming uniform (like nn.Linear default)
        # Initialize B with zeros (so initially LoRA contributes nothing)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Scaling factor
        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining frozen weights and LoRA adaptation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features)

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (batch_size, out_features)
        """
        # Original forward pass (frozen)
        result = nn.functional.linear(x, self.weight, self.bias)

        # LoRA adaptation: x @ A^T @ B^T
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        result = result + lora_output * self.scaling

        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into the original layer (for inference efficiency).

        After merging, the layer behaves like a standard Linear layer with
        updated weights, eliminating the LoRA computation overhead.
        """
        if self.rank > 0:
            # W_new = W_0 + scaling * B @ A
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.weight.data += delta_w

            # Reset LoRA matrices
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters in LoRA matrices."""
        return self.rank * (self.in_features + self.out_features)

    def num_total_parameters(self) -> int:
        """Return the total number of parameters (frozen + trainable)."""
        total = self.in_features * self.out_features
        if self.bias is not None:
            total += self.out_features
        total += self.num_trainable_parameters()
        return total


class LoRAConv2d(nn.Module):
    """Conv2d layer with Low-Rank Adaptation.

    For convolutional layers, LoRA decomposes the weight update as:
        h = W_0 * x + (B * A) * x
    where:
        - W_0 ∈ R^{out_ch x in_ch x k x k} is frozen (original conv weights)
        - A ∈ R^{r x in_ch x 1 x 1} (pointwise convolution for dimensionality reduction)
        - B ∈ R^{out_ch x r x 1 x 1} (pointwise convolution for dimensionality expansion)
        - r is the rank (r << min(in_ch, out_ch))

    This allows efficient adaptation of convolutional feature extractors.

    Parameters
    ----------
    original_layer : nn.Conv2d
        The original convolutional layer to adapt.
    rank : int
        Rank of the low-rank decomposition. Lower rank = fewer parameters.
    alpha : float, optional
        Scaling factor for LoRA outputs, by default 1.0.
    """

    def __init__(
        self,
        original_layer: nn.Conv2d,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        self.weight = original_layer.weight
        self.weight.requires_grad = False

        # Handle bias
        if original_layer.bias is not None:
            self.bias = original_layer.bias
            self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)

        # LoRA trainable parameters (1x1 convolutions for efficiency)
        # A: projects channels to low-rank space (in_channels -> rank)
        # B: projects from low-rank space to output (rank -> out_channels)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, 1, 1))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank, 1, 1))

        # Initialize A with Kaiming uniform (like nn.Conv2d default)
        # Initialize B with zeros (so initially LoRA contributes nothing)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Scaling factor
        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining frozen weights and LoRA adaptation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width)

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (batch, out_channels, height', width')
        """
        # Original forward pass (frozen)
        result = nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # LoRA adaptation: Apply A (1x1 conv) with original stride/padding to match spatial dims
        # Then apply B (1x1 conv) with no additional spatial transformation
        lora_out = nn.functional.conv2d(
            x,
            self.lora_A,
            stride=self.stride,
            padding=0,  # 1x1 conv doesn't need padding
        )
        lora_out = nn.functional.conv2d(
            lora_out,
            self.lora_B,
            stride=1,  # No additional downsampling
            padding=0,
        )
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into the original layer.

        For Conv2d, this expands the 1x1 LoRA convolutions to match the original kernel size.
        """
        if self.rank > 0:
            # Compute effective LoRA kernel: B @ A
            # This requires expanding to full kernel size
            lora_weight = torch.zeros_like(self.weight)

            # For 1x1 LoRA on arbitrary kernel: contribution is centered
            k_h, k_w = self.kernel_size
            pad_h = k_h // 2
            pad_w = k_w // 2

            # Compute B @ A (both are 1x1, result is 1x1)
            ba = self.lora_B.squeeze(-1).squeeze(-1) @ self.lora_A.squeeze(-1).squeeze(-1)
            # Place in center of kernel
            lora_weight[:, :, pad_h, pad_w] = ba

            self.weight.data += self.scaling * lora_weight

            # Reset LoRA matrices
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters in LoRA matrices."""
        # A: rank x in_channels x 1 x 1
        # B: out_channels x rank x 1 x 1
        return self.rank * (self.in_channels + self.out_channels)

    def num_total_parameters(self) -> int:
        """Return the total number of parameters (frozen + trainable)."""
        k_h, k_w = self.kernel_size
        total = self.in_channels * self.out_channels * k_h * k_w
        if self.bias is not None:
            total += self.out_channels
        total += self.num_trainable_parameters()
        return total


def replace_linear_with_lora(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: list[str] | None = None,
    replace_conv2d: bool = False,
) -> nn.Module:
    """Replace nn.Linear (and optionally nn.Conv2d) layers with LoRA versions.

    Parameters
    ----------
    module : nn.Module
        The module to modify.
    rank : int, optional
        Rank for LoRA decomposition, by default 4.
    alpha : float, optional
        Scaling factor for LoRA, by default 1.0.
    target_modules : list[str], optional
        List of module names to replace. If None, replaces all matching layer types.
    replace_conv2d : bool, optional
        Whether to also replace Conv2d layers (for CNN encoders), by default False.

    Returns
    -------
    module : nn.Module
        Modified module with LoRA layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Check if this layer should be replaced
            if target_modules is None or name in target_modules:
                # Replace with LoRA version
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, name, lora_layer)
        elif isinstance(child, nn.Conv2d) and replace_conv2d:
            # Replace Conv2d layers if requested (for CNNs like ResNet)
            if (target_modules is None or name in target_modules) and (child.groups == 1 or child.groups == child.in_channels):
                lora_layer = LoRAConv2d(child, rank=rank, alpha=alpha)
                setattr(module, name, lora_layer)
        else:
            # Recursively apply to nested modules
            replace_linear_with_lora(
                child, rank=rank, alpha=alpha, target_modules=target_modules, replace_conv2d=replace_conv2d
            )

    return module


def count_lora_parameters(module: nn.Module) -> dict[str, int]:
    """Count LoRA parameters in a module.

    Parameters
    ----------
    module : nn.Module
        Module to inspect.

    Returns
    -------
    stats : dict[str, int]
        Dictionary with parameter counts:
        - trainable: Number of trainable LoRA parameters
        - frozen: Number of frozen parameters
        - total: Total parameters
        - efficiency_pct: Percentage of parameters that are trainable
    """
    trainable = 0
    frozen = 0

    for param in module.parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable += num_params
        else:
            frozen += num_params

    total = trainable + frozen
    efficiency_pct = (trainable / total * 100) if total > 0 else 0.0

    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "efficiency_pct": efficiency_pct,
    }


def print_lora_info(module: nn.Module) -> None:
    """Print information about LoRA layers in a module.

    Parameters
    ----------
    module : nn.Module
        Module to inspect.
    """
    print("\nLoRA Layer Information:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Type':<8} {'Rank':<6} {'Trainable':<12} {'Total':<12}")
    print("-" * 80)

    total_trainable = 0
    total_params = 0

    for name, layer in module.named_modules():
        if isinstance(layer, LoRALinear | LoRAConv2d):
            layer_type = "Linear" if isinstance(layer, LoRALinear) else "Conv2d"
            trainable = layer.num_trainable_parameters()
            total = layer.num_total_parameters()
            total_trainable += trainable
            total_params += total
            print(f"{name:<40} {layer_type:<8} {layer.rank:<6} {trainable:<12,} {total:<12,}")

    if total_params > 0:
        print("-" * 80)
        print(f"{'TOTAL':<40} {'':<8} {'':<6} {total_trainable:<12,} {total_params:<12,}")
        print(f"\nParameter Efficiency: {total_trainable/total_params*100:.4f}% trainable")
    else:
        print("No LoRA layers found.")
