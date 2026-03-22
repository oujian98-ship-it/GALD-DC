import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .config import TrainingConfig


class LossCalculator:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def compute_real_loss(
        self,
        classifier: nn.Module,
        real_features: torch.Tensor,
        labels: torch.Tensor,
        class_sample_counts=None,
        loss_type="ce",
        class_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if loss_type == "ce":
            real_logits = classifier(real_features)
            return F.cross_entropy(
                real_logits, labels, weight=class_weights, label_smoothing=0
            )
        else:
            real_logits = classifier(real_features)
            return F.cross_entropy(real_logits, labels, weight=class_weights)

    def compute_diffusion_loss(
        self,
        diffusion_model: nn.Module,
        real_features: torch.Tensor,
        labels: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        real_features_reshaped = real_features.unsqueeze(1)

        return self._manual_diffusion_loss(
            diffusion_model, real_features_reshaped, None, labels
        )

    def _manual_diffusion_loss(
        self,
        diffusion_model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        b, c, n = x_start.shape
        device = x_start.device

        # 1. Generate timestamp t (if not provided)
        if t is None:
            t = torch.randint(0, self.config.diffusion_steps, (b,), device=device)

        # 2. Generate noise
        noise = torch.randn_like(x_start)

        # 3. Forward noise addition (q_sample) - applied only once
        x_noisy = diffusion_model.q_sample(x_start=x_start, t=t, noise=noise)

        # Handle NaN values
        if torch.isnan(x_noisy).any():
            x_noisy = torch.nan_to_num(x_noisy, nan=0.0)

        # 4. Predict target
        if diffusion_model.objective == "pred_noise":
            target = noise
        elif diffusion_model.objective == "pred_x0":
            target = x_start  # Target is the original x_start, not normalized
        else:
            raise ValueError(f"Unknown objective: {diffusion_model.objective}")

        # 5. Model prediction
        model_output = diffusion_model.model(x_noisy, t, labels)

        # 6. Compute loss
        loss = F.mse_loss(model_output, target, reduction="none")
        loss = loss.mean(dim=[1, 2])

        loss_weight = diffusion_model.loss_weight[t]
        loss_weight = torch.clamp(loss_weight, max=self.config.max_loss_weight)

        diffusion_loss = (loss * loss_weight).mean()
        diffusion_loss = torch.clamp(diffusion_loss, max=self.config.max_diffusion_loss)

        return diffusion_loss

    def compute_prototype_loss(
        self,
        estimated_clean_features: torch.Tensor,
        labels: torch.Tensor,
        class_mu: Dict[int, torch.Tensor],
        num_classes: int,
    ) -> torch.Tensor:
        batch_size = estimated_clean_features.size(0)
        device = estimated_clean_features.device

        prototype_loss = torch.tensor(0.0, device=device)
        valid_samples = 0

        if (
            torch.isnan(estimated_clean_features).any()
            or torch.isinf(estimated_clean_features).any()
        ):
            print(
                "Warning: Estimated clean features contain NaN or Inf values, skipping prototype loss"
            )
            return torch.tensor(0.0, device=device)

        estimated_clean_features = torch.clamp(
            estimated_clean_features,
            min=-self.config.feature_clamp_max,
            max=self.config.feature_clamp_max,
        )

        for i in range(batch_size):
            cls_idx = labels[i].item()
            if 0 <= cls_idx < num_classes:
                if (
                    torch.isnan(class_mu[cls_idx]).any()
                    or torch.isinf(class_mu[cls_idx]).any()
                ):
                    continue

                proto = torch.clamp(
                    class_mu[cls_idx].detach(),
                    min=-self.config.feature_clamp_max,
                    max=self.config.feature_clamp_max,
                )

                loss_item = F.mse_loss(estimated_clean_features[i], proto)

                if not torch.isnan(loss_item) and not torch.isinf(loss_item):
                    loss_item = torch.clamp(loss_item, max=10.0)
                    prototype_loss += loss_item
                    valid_samples += 1

        prototype_loss = (
            prototype_loss / valid_samples
            if valid_samples > 0
            else torch.tensor(0.0, device=device)
        )
        prototype_loss = torch.clamp(prototype_loss, max=5.0)

        return prototype_loss

    def compute_radius_constraint_loss(
        self,
        estimated_clean_features: torch.Tensor,
        labels: torch.Tensor,
        class_mu: Dict[int, torch.Tensor],
        r_obs: List[torch.Tensor],
        num_classes: int,
    ) -> torch.Tensor:
        device = estimated_clean_features.device
        radius_loss = torch.tensor(0.0, device=device)
        valid_samples = 0

        if (
            torch.isnan(estimated_clean_features).any()
            or torch.isinf(estimated_clean_features).any()
        ):
            print(
                "Warning: Estimated clean features contain NaN or Inf values, skipping radius constraint loss"
            )
            return torch.tensor(0.0, device=device)

        estimated_clean_features = torch.clamp(
            estimated_clean_features,
            min=-self.config.feature_clamp_max,
            max=self.config.feature_clamp_max,
        )

        batch_size = estimated_clean_features.size(0)
        for i in range(batch_size):
            cls_idx = labels[i].item()
            if 0 <= cls_idx < num_classes:
                feature = estimated_clean_features[i]
                proto = class_mu[cls_idx]

                if torch.isnan(feature).any() or torch.isinf(feature).any():
                    continue

                if torch.isnan(proto).any() or torch.isinf(proto).any():
                    continue

                distance = torch.norm(feature - proto, p=2)

                if cls_idx < len(r_obs):
                    target_radius = r_obs[cls_idx]
                    if target_radius.device != device:
                        target_radius = target_radius.to(device)
                else:
                    if torch.any(r_obs > 0):
                        mask = r_obs > 0
                        avg_radius = r_obs[mask].mean()
                        target_radius = avg_radius.to(device)
                    else:
                        target_radius = torch.tensor(
                            self.config.target_radius, device=device
                        )

                delta = getattr(self.config, "radius_slack", 0.5)
                loss_item = F.relu(torch.abs(distance - target_radius) - delta)

                if not torch.isnan(loss_item) and not torch.isinf(loss_item):
                    radius_loss += loss_item
                    valid_samples += 1

        radius_loss = (
            radius_loss / valid_samples
            if valid_samples > 0
            else torch.tensor(0.0, device=device)
        )
        radius_loss = torch.clamp(radius_loss, max=self.config.max_radius_loss)

        return radius_loss

    def compute_total_loss(
        self,
        real_loss: torch.Tensor,
        semantic_loss: torch.Tensor,
        gen_loss: torch.Tensor,
        loss_weights: Dict[str, float],
    ) -> torch.Tensor:
        lambda_sem = loss_weights["lambda_sem"]
        gamma_ge = loss_weights["gamma_ge"]

        total_loss = real_loss + lambda_sem * semantic_loss + gamma_ge * gen_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: Total loss is NaN or Inf")
            print(
                f"Real loss: {real_loss.item()}, Semantic loss: {semantic_loss.item()}, Gen loss: {gen_loss.item()}"
            )
            total_loss = real_loss

        return total_loss

    def compute_margin_loss(
        self,
        estimated_clean: torch.Tensor,
        labels: torch.Tensor,
        class_mu: Dict[int, torch.Tensor],
        num_classes: int,
        margin: float,
    ) -> torch.Tensor:
        device = estimated_clean.device
        batch_size = estimated_clean.size(0)

        if torch.isnan(estimated_clean).any() or torch.isinf(estimated_clean).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        estimated_clean = torch.clamp(
            estimated_clean,
            min=-self.config.feature_clamp_max,
            max=self.config.feature_clamp_max,
        )

        # Build prototype matrix [num_classes, feature_dim]
        prototype_matrix = torch.zeros(
            num_classes, estimated_clean.size(1), device=device
        )
        valid_prototypes = torch.zeros(num_classes, dtype=torch.bool, device=device)

        for cls_idx in range(num_classes):
            if cls_idx in class_mu:
                proto = class_mu[cls_idx]
                if not (torch.isnan(proto).any() or torch.isinf(proto).any()):
                    prototype_matrix[cls_idx] = proto.to(device)
                    valid_prototypes[cls_idx] = True

        if valid_prototypes.sum() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Calculate squared distances [batch_size, num_classes]
        dists = torch.cdist(estimated_clean, prototype_matrix, p=2) ** 2

        margin_losses = []

        for i in range(batch_size):
            cls_idx = labels[i].item()
            if not (0 <= cls_idx < num_classes) or not valid_prototypes[cls_idx]:
                continue

            # Distance to own class prototype
            dist_to_pos = dists[i, cls_idx]

            # Find closest negative class prototypes
            neg_mask = valid_prototypes.clone()
            neg_mask[cls_idx] = False
            if neg_mask.sum() == 0:
                continue

            neg_dists = dists[i, neg_mask]

            # [R4 Fix] Top-K Soft Margin: average over K nearest negative classes to smooth gradients
            top_k = getattr(self.config, "margin_top_k", 3)
            actual_k = min(top_k, neg_dists.size(0))

            top_k_neg_dists, _ = neg_dists.topk(actual_k, largest=False)

            loss_items = F.relu(margin + dist_to_pos - top_k_neg_dists)
            loss_item = loss_items.mean()

            if not (torch.isnan(loss_item) or torch.isinf(loss_item)):
                margin_losses.append(loss_item)

        if len(margin_losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        margin_loss = torch.stack(margin_losses).mean()
        margin_loss = torch.clamp(margin_loss, max=10.0)

        return margin_loss

    def compute_calibrated_radius(
        self,
        observed_radii: torch.Tensor,
        r_prior: float,
        class_counts: torch.Tensor,
        tau: int,
        lambda_cal: float,
    ) -> torch.Tensor:
        device = observed_radii.device
        r_cal = observed_radii.clone()

        tail_mask = class_counts < tau
        r_cal[tail_mask] = (
            lambda_cal * observed_radii[tail_mask] + (1 - lambda_cal) * r_prior
        )

        return r_cal

    def compute_head_class_prior(
        self, observed_radii: torch.Tensor, class_counts: torch.Tensor, tau: int
    ) -> float:
        head_mask = class_counts >= tau

        if head_mask.sum() == 0:
            return observed_radii.mean().item()

        r_prior = observed_radii[head_mask].mean().item()
        return r_prior

    def compute_consistency_loss(
        self, current_features: torch.Tensor, frozen_features: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(current_features, frozen_features.detach())
