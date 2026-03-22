import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

from dataloader.Custom_Dataloader import Custom_dataset, data_loader_wrapper_cust
from dataloader.data_loader_wrapper import data_loader_wrapper
from dataloader.data_loader_wrapper import Custom_dataset_ImageNet
from utilis.config_parse import config_setup
from model.metrics import *
from model.balanced_softmax import balanced_softmax_probs

from .config import TrainingConfig
from .model_manager import ModelManager
from .loss_calculator import LossCalculator
from .training_monitor import TrainingMonitor
from .lora_adapter import apply_lora_to_diffusion_model


class GALDDCTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_manager = ModelManager(config, self.device)
        self.loss_calculator = LossCalculator(config)
        self.monitor = TrainingMonitor(config)

        self.ce_loss_history = []
        self.actual_stage1_end = None

        print(f"Using device: {self.device}")

    def train(self):
        cfg, finish = config_setup(
            self.config.config, None, self.config.datapath, update=False
        )
        train_set, val_set, test_set, num_classes, dataset_info = self._load_data(cfg)

        stage3_accs = []
        stage3_bs_accs = []

        if "per_class_img_num" in dataset_info:
            train_class_counts = torch.tensor(
                dataset_info["per_class_img_num"], device=self.device
            ).float()
        else:
            print(">>> Counting training samples per class...")
            train_class_counts = torch.zeros(num_classes, device=self.device)
            for _, labels in train_set:
                train_class_counts.index_add_(
                    0, labels.to(self.device), torch.ones_like(labels).float()
                )

        print(">>> [Mode] Standard CE Training (Baseline Attack) + LDMLR Equipment")
        (
            encoder,
            classifier,
            diffusion_model,
            feature_dim,
        ) = self.model_manager.initialize_models(num_classes, dataset_info)

        optimizer = optim.SGD(
            [
                {"params": encoder.parameters(), "lr": self.config.lr * 0.01},
                {"params": classifier.parameters(), "lr": self.config.lr},
                {"params": diffusion_model.parameters(), "lr": self.config.lr},
            ],
            momentum=0.9,
            nesterov=True,
            weight_decay=self.config.weight_decay,
        )

        print(">>> [Optimizer] SGD with momentum=0.9, nesterov=True")
        print(
            f">>> [Optimizer] Encoder lr: {self.config.lr * 0.01:.6f}, Classifier lr: {self.config.lr:.6f}, Diffusion lr: {self.config.lr:.6f}"
        )

        self.optimizer = optimizer

        (
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        ) = self._get_diffusion_schedule()

        with torch.no_grad():
            cache_dir = os.path.join(os.path.dirname(self.config.config), "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = f"{self.config.dataset}_mu_r_cache.pth"
            cache_path = os.path.join(cache_dir, cache_name)

            loaded_from_cache = False
            if os.path.exists(cache_path):
                print(f">>> [GALD-DC] Loading initialization cache from {cache_path}")
                try:
                    cache_data = torch.load(cache_path, map_location=self.device)
                    class_mu = cache_data["class_mu"]
                    r_obs = cache_data["r_obs"]
                    loaded_from_cache = True
                except Exception as e:
                    print(f">>> [GALD-DC] Cache loading failed: {e}. Re-computing...")

            if not loaded_from_cache:
                class_mu, r_obs = self._compute_initial_stats(
                    encoder, train_set, num_classes, feature_dim
                )

                print(
                    f">>> [GALD-DC] Saving initialization stats to cache: {cache_path}"
                )
                torch.save({"class_mu": class_mu, "r_obs": r_obs}, cache_path)

        if self.config.tau == -1:
            tau = self._compute_auto_tau(
                train_class_counts, self.config.imb_factor, self.config.dataset
            )
            print(f">>> [GALD-DC] Auto-computed tau: {tau}")
        else:
            tau = self.config.tau
            print(f">>> [GALD-DC] Using manual tau: {tau}")

        self.tau = tau

        r_prior = self.loss_calculator.compute_head_class_prior(
            r_obs, train_class_counts, tau
        )
        print(f">>> [GALD-DC] r_prior (head class radius prior): {r_prior:.4f}")
        print(f">>> [GALD-DC] Stage 3 mode: {self.config.stage3_mode}")

        head_count = (train_class_counts >= tau).sum().item()
        tail_count = (train_class_counts < tau).sum().item()
        print(
            f">>> [GALD-DC] Head classes: {int(head_count)}, Tail classes: {int(tail_count)}"
        )

        import copy

        self.r_prior = r_prior
        self.class_counts = train_class_counts

        print(f"\n{'='*60}")
        print("Three-Stage Training Configuration:")
        print(
            f"  Stage 1 (Enc+Cls Pre-training): Epoch 0-{self.config.stage1_end_epoch-1}"
        )
        print(
            f"  Stage 2 (Diffusion Training):   Epoch {self.config.stage1_end_epoch}-{self.config.stage2_end_epoch-1}"
        )
        print(
            f"  Stage 3 (Controlled Fine-tune): Epoch {self.config.stage2_end_epoch}-{self.config.epochs-1}"
        )
        print(f"{'='*60}\n")

        for epoch in range(self.config.epochs):
            stage = self._get_training_stage(epoch)
            loss_weights = self._get_dynamic_loss_weights(epoch)

            if stage == 1 and epoch >= self.config.min_stage1_epochs:
                if self._is_ce_loss_converged():
                    print(f"\n{'='*70}")
                    print(f">>> Stage 1 Early Exit Triggered at Epoch {epoch}")
                    print(">>> CE Loss has converged, transitioning to Stage 2...")
                    print(f"{'='*70}\n")

                    self.config.stage1_end_epoch = epoch + 1
                    self.config.stage2_end_epoch = (
                        self.config.stage1_end_epoch + self.config.stage2_epochs
                    )
                    self.config.epochs = (
                        self.config.stage2_end_epoch + self.config.stage3_epochs
                    )
                    self.actual_stage1_end = epoch

                    print(">>> Updated Stage boundaries:")
                    print(f"    Stage 1: Epoch 0-{self.config.stage1_end_epoch-1}")
                    print(
                        f"    Stage 2: Epoch {self.config.stage1_end_epoch}-{self.config.stage2_end_epoch-1} ({self.config.stage2_epochs} epochs)"
                    )
                    print(
                        f"    Stage 3: Epoch {self.config.stage2_end_epoch}-{self.config.epochs-1} ({self.config.stage3_epochs} epochs)"
                    )
                    print(f"    Total: {self.config.epochs} epochs\n")

                    stage = self._get_training_stage(epoch)

            if epoch == self.config.stage1_end_epoch:
                print(f"\n{'='*60}")
                print(
                    "[Stage 1 Complete] Building latent dataset and freezing Encoder..."
                )
                with torch.no_grad():
                    latent_features_list = []
                    latent_labels_list = []

                    encoder.eval()
                    for inputs, labels in tqdm(
                        train_set, desc="Building Latent Dataset D_z"
                    ):
                        inputs = inputs.to(self.device)
                        features = encoder.forward_no_fc(inputs)
                        latent_features_list.append(features.cpu())
                        latent_labels_list.append(labels)

                    all_latent_features = torch.cat(latent_features_list, dim=0)
                    all_latent_labels = torch.cat(latent_labels_list, dim=0)

                    print(
                        f"  Latent Dataset size: {all_latent_features.size(0)} samples"
                    )
                    print(f"  Feature dimension: {all_latent_features.size(1)}")
                    memory_mb = (
                        all_latent_features.element_size()
                        * all_latent_features.nelement()
                    ) / (1024**2)
                    print(f"  Memory usage: {memory_mb:.2f} MB")

                    from .latent_dataset import LatentDataset

                    latent_dataset = LatentDataset(
                        all_latent_features, all_latent_labels
                    )

                    from torch.utils.data import DataLoader

                    self.latent_train_loader = DataLoader(
                        latent_dataset,
                        batch_size=train_set.batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                    )
                    print(
                        f"  Created latent DataLoader with batch_size={train_set.batch_size}"
                    )

                    class_mu, r_obs = self._compute_initial_stats_from_latent(
                        all_latent_features, all_latent_labels, num_classes, feature_dim
                    )

                    r_prior = self.loss_calculator.compute_head_class_prior(
                        r_obs, train_class_counts, self.tau
                    )
                    self.r_prior = r_prior
                    print(f"  Updated r_prior: {r_prior:.4f}")

                    self.frozen_encoder = copy.deepcopy(encoder)
                    self.frozen_encoder.eval()
                    for param in self.frozen_encoder.parameters():
                        param.requires_grad = False
                    print("  Frozen encoder copy E^(0) saved")
                print(f"{'='*60}\n")

            if (
                epoch == 0
                or epoch == self.config.stage1_end_epoch
                or epoch == self.config.stage2_end_epoch
            ):
                print(f"\n>>> [Stage {stage}] Started - Epoch {epoch}")

            if stage == 2 and hasattr(self, "latent_train_loader"):
                active_loader = self.latent_train_loader
            else:
                active_loader = train_set

            train_loss, train_accuracy = self._train_epoch(
                epoch,
                encoder,
                classifier,
                diffusion_model,
                optimizer,
                active_loader,
                class_mu,
                r_obs,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                num_classes,
                feature_dim,
                loss_weights,
                train_class_counts,
                r_prior,
                stage,
            )

            if stage == 1:
                self.ce_loss_history.append(train_loss)

            if stage == 3:
                t_loss, t_acc, t_bs_acc, t_mmf, t_mmf_bs = self._validate(
                    encoder, classifier, test_set, dataset_info, train_class_counts
                )

                self.monitor.log_validation(
                    epoch, t_acc, t_loss, t_bs_acc, t_mmf, t_mmf_bs, mode="Test"
                )

                if t_bs_acc > self.monitor.best_bs_acc:
                    self.monitor.best_bs_acc = t_bs_acc
                    self._save_checkpoint(
                        epoch,
                        encoder,
                        classifier,
                        diffusion_model,
                        optimizer,
                        t_bs_acc,
                        "bs",
                    )

                stage3_accs.append(t_acc)
                stage3_bs_accs.append(t_bs_acc)

            if epoch == self.config.stage2_end_epoch - 1:
                print("\n[Stage 2 Complete] Saving final diffusion model...")
                self.model_manager.save_diffusion_model_to_pretrained(
                    diffusion_model, epoch
                )

                if getattr(self.config, "enable_lora", True):
                    print("\n[R6 LoRA] Injecting LoRA layers into diffusion model...")
                    self.lora_adapter, self.lora_params = apply_lora_to_diffusion_model(
                        diffusion_model,
                        rank=getattr(self.config, "lora_rank", 4),
                        alpha=getattr(self.config, "lora_alpha", 8.0),
                    )

                    if self.lora_params:
                        optimizer.add_param_group(
                            {"params": self.lora_params, "lr": self.config.lr}
                        )
                        print("[R6 LoRA] LoRA parameters added to optimizer")

        self.encoder = encoder
        self.classifier = classifier
        self.diffusion_model = diffusion_model

        avg_acc = sum(stage3_accs) / len(stage3_accs) if stage3_accs else None
        avg_bs_acc = (
            sum(stage3_bs_accs) / len(stage3_bs_accs) if stage3_bs_accs else None
        )

        self._save_accuracy_history(avg_acc, avg_bs_acc)

    def _is_ce_loss_converged(self) -> bool:
        if not self.config.enable_dynamic_stage1:
            return False

        window = self.config.convergence_window
        threshold = self.config.convergence_threshold

        if len(self.ce_loss_history) < window:
            return False

        recent_losses = self.ce_loss_history[-window:]
        avg_loss = sum(recent_losses) / window

        max_deviation = max(abs(loss - avg_loss) for loss in recent_losses)
        relative_change = max_deviation / avg_loss if avg_loss > 0 else 1.0

        converged = relative_change < threshold

        if converged:
            print(
                f"\n>>> CE Loss converged! Recent {window} epochs: avg={avg_loss:.4f}, max_dev={max_deviation:.4f}, rel_change={relative_change:.4f}"
            )

        return converged

    def _get_training_stage(self, epoch: int) -> int:
        if epoch < self.config.stage1_end_epoch:
            return 1
        elif epoch < self.config.stage2_end_epoch:
            return 2
        else:
            return 3

    def _set_models_mode(
        self,
        stage: int,
        encoder: nn.Module,
        classifier: nn.Module,
        diffusion_model: nn.Module,
    ):
        if stage == 1:
            encoder.train()
            classifier.train()
            diffusion_model.eval()
            for param in encoder.parameters():
                param.requires_grad = True
            for param in classifier.parameters():
                param.requires_grad = True
            for param in diffusion_model.parameters():
                param.requires_grad = False

        elif stage == 2:
            encoder.eval()
            classifier.eval()
            diffusion_model.train()
            for param in encoder.parameters():
                param.requires_grad = False
            for param in classifier.parameters():
                param.requires_grad = False
            for param in diffusion_model.parameters():
                param.requires_grad = True

        else:
            if self.config.stage3_mode == "hybrid":
                encoder.train()
                classifier.train()
                diffusion_model.eval()
                for param in encoder.parameters():
                    param.requires_grad = True
                for param in classifier.parameters():
                    param.requires_grad = True
                for param in diffusion_model.parameters():
                    param.requires_grad = False
            else:
                encoder.eval()
                classifier.train()
                diffusion_model.eval()
                for param in encoder.parameters():
                    param.requires_grad = False
                for param in classifier.parameters():
                    param.requires_grad = True
                for param in diffusion_model.parameters():
                    param.requires_grad = False

    def _prepare_batch_data(self, stage: int, batch_data, encoder: nn.Module):
        if stage == 2:
            real_features, labels = batch_data
            real_features = real_features.to(self.device)
            labels = labels.to(self.device)
            inputs = None
            frozen_features = None
        else:
            inputs, labels = batch_data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if stage == 3 and self.config.stage3_mode == "hybrid":
                real_features = encoder.forward_no_fc(inputs)
                with torch.no_grad():
                    frozen_features = self.frozen_encoder.forward_no_fc(inputs)
            elif stage == 1:
                real_features = encoder.forward_no_fc(inputs)
                frozen_features = None
            else:
                with torch.no_grad():
                    real_features = encoder.forward_no_fc(inputs)
                frozen_features = None

        return real_features, inputs, labels, frozen_features

    def _compute_stage1_loss(self, classifier, real_features, labels):
        L_real = self.loss_calculator.compute_real_loss(
            classifier, real_features, labels
        )

        total_loss = L_real

        return L_real, total_loss

    def _compute_stage2_loss(
        self,
        diffusion_model,
        real_features,
        labels,
        class_mu,
        r_obs,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        num_classes,
        epoch,
        train_class_counts,
        r_prior,
    ):
        L_ldm = self.loss_calculator.compute_diffusion_loss(
            diffusion_model,
            real_features,
            labels,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        )

        estimated_clean = self._estimate_clean_features(
            diffusion_model,
            real_features,
            labels,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            epoch=epoch,
        )

        L_proto = self.loss_calculator.compute_prototype_loss(
            estimated_clean, labels, class_mu, num_classes
        )

        r_cal = self.loss_calculator.compute_calibrated_radius(
            r_obs, r_prior, train_class_counts, self.tau, self.config.lambda_cal
        )
        L_rad = self.loss_calculator.compute_radius_constraint_loss(
            estimated_clean, labels, class_mu, r_cal, num_classes
        )

        L_margin = self.loss_calculator.compute_margin_loss(
            estimated_clean, labels, class_mu, num_classes, self.config.margin_m
        )

        L_ldm = self._safe_loss(L_ldm, 5.0)
        L_proto = self._safe_loss(L_proto, 10.0)
        L_rad = self._safe_loss(L_rad, 10.0)
        L_margin = self._safe_loss(L_margin, 10.0)

        stage2_progress = (epoch - self.config.stage1_end_epoch) / (
            self.config.stage2_end_epoch - self.config.stage1_end_epoch
        )

        L_semantic = (
            L_ldm
            + self.config.eta_p * L_proto
            + self.config.eta_r * L_rad
            + self.config.eta_m * L_margin
        )
        L_semantic = self._safe_loss(L_semantic, self.config.max_L_semantic)

        total_loss = L_semantic

        return L_ldm, L_proto, L_rad, L_margin, L_semantic, total_loss

    def _compute_stage3_loss(
        self,
        classifier,
        diffusion_model,
        batch_size,
        feature_dim,
        real_features,
        labels,
        frozen_features,
        class_mu,
        r_obs,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        num_classes,
        batch_idx,
    ):
        L_real = self.loss_calculator.compute_real_loss(
            classifier, real_features, labels
        )

        if self.config.stage3_mode == "hybrid" and frozen_features is not None:
            L_cons = self.loss_calculator.compute_consistency_loss(
                real_features, frozen_features
            )
            L_cons = self._safe_loss(L_cons, 5.0)

        L_ge = self._compute_generation_loss(
            diffusion_model,
            classifier,
            batch_size,
            feature_dim,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            num_classes,
            batch_idx,
            class_mu=class_mu,
            r_obs=r_obs,
        )
        L_ge = self._safe_loss(L_ge, self.config.max_L_ge)

        total_loss = (
            L_real + self.config.gamma_pseudo * L_ge + self.config.beta_cons * L_cons
        )

        return L_real, L_cons, L_ge, total_loss

    def _train_epoch(
        self,
        epoch: int,
        encoder: nn.Module,
        classifier: nn.Module,
        diffusion_model: nn.Module,
        optimizer: optim.Optimizer,
        train_set: DataLoader,
        class_mu: Dict[int, torch.Tensor],
        r_obs: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
        num_classes: int,
        feature_dim: int,
        loss_weights: Dict[str, float],
        train_class_counts: torch.Tensor,
        r_prior: float = 1.0,
        stage: int = None,
    ):
        stage = self._get_training_stage(epoch)

        self._set_models_mode(stage, encoder, classifier, diffusion_model)
        running_losses = {
            "real": 0.0,
            "diffusion": 0.0,
            "prototype": 0.0,
            "radius": 0.0,
            "margin": 0.0,
            "semantic": 0.0,
            "gen": 0.0,
            "consistency": 0.0,
            "total": 0.0,
        }
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(train_set):
            real_features, inputs, labels, frozen_features = self._prepare_batch_data(
                stage, batch_data, encoder
            )

            losses = self._compute_batch_losses(
                encoder,
                classifier,
                diffusion_model,
                real_features,
                inputs,
                labels,
                class_mu,
                r_obs,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                num_classes,
                feature_dim,
                epoch,
                batch_idx,
                loss_weights,
                train_class_counts,
                stage,
                frozen_features,
                r_prior,
            )

            losses["total"].backward()
            self._control_gradients(
                encoder, classifier, diffusion_model, epoch * len(train_set), batch_idx
            )
            optimizer.step()
            optimizer.zero_grad()

            ema_warmup_epochs = getattr(self.config, "ema_warmup_epochs", 10)
            if stage == 2:
                stage2_epoch = epoch - self.config.stage1_end_epoch
                if stage2_epoch >= ema_warmup_epochs:
                    self._update_stats_ema(
                        real_features.detach(), labels, class_mu, r_obs, num_classes
                    )

            for key in running_losses:
                running_losses[key] += losses[key].item()
            total_loss += losses["total"].item()

            if batch_idx % 50 == 0:
                self.monitor.log_batch_progress(
                    epoch, batch_idx, {k: v.item() for k, v in losses.items()}, stage
                )

        train_loss = total_loss / len(train_set)
        if stage == 3:
            train_accuracy = self._compute_train_accuracy(
                encoder, classifier, train_set, train_class_counts, num_classes
            )
        else:
            train_accuracy = None

        avg_losses = {
            key: value / len(train_set) for key, value in running_losses.items()
        }
        self.monitor.log_epoch_summary(
            epoch, avg_losses, train_accuracy, train_loss, stage
        )
        return train_loss, train_accuracy

    def _compute_batch_losses(
        self,
        encoder,
        classifier,
        diffusion_model,
        real_features: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        class_mu: Dict[int, torch.Tensor],
        r_obs: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
        num_classes: int,
        feature_dim: int,
        epoch: int,
        batch_idx: int,
        loss_weights: Dict[str, float],
        train_class_counts: torch.Tensor,
        stage: int = 1,
        frozen_features: torch.Tensor = None,
        r_prior: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        batch_size = inputs.size(0) if inputs is not None else real_features.size(0)

        L_real = torch.tensor(0.0, device=self.device)
        L_semantic = torch.tensor(0.0, device=self.device)
        L_ge = torch.tensor(0.0, device=self.device)
        L_margin = torch.tensor(0.0, device=self.device)
        L_cons = torch.tensor(0.0, device=self.device)

        if stage == 1:
            L_real, total_loss = self._compute_stage1_loss(
                classifier, real_features, labels
            )
        elif stage == 2:
            (
                L_ldm,
                L_proto,
                L_rad,
                L_margin,
                L_semantic,
                total_loss,
            ) = self._compute_stage2_loss(
                diffusion_model,
                real_features,
                labels,
                class_mu,
                r_obs,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                num_classes,
                epoch,
                train_class_counts,
                r_prior,
            )
        else:
            L_real, L_cons, L_ge, total_loss = self._compute_stage3_loss(
                classifier,
                diffusion_model,
                batch_size,
                feature_dim,
                real_features,
                labels,
                frozen_features,
                class_mu,
                r_obs,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                num_classes,
                batch_idx,
            )

        if stage == 2:
            return {
                "real": L_real,
                "diffusion": L_ldm,
                "prototype": L_proto,
                "radius": L_rad,
                "margin": L_margin,
                "semantic": L_semantic,
                "gen": L_ge,
                "consistency": L_cons,
                "total": total_loss,
            }
        else:
            return {
                "real": L_real,
                "diffusion": torch.tensor(0.0, device=self.device),
                "prototype": torch.tensor(0.0, device=self.device),
                "radius": torch.tensor(0.0, device=self.device),
                "margin": L_margin,
                "semantic": L_semantic,
                "gen": L_ge,
                "consistency": L_cons,
                "total": total_loss,
            }

    def _compute_generation_loss(
        self,
        diffusion_model,
        classifier,
        batch_size,
        feature_dim,
        sqrt_alphas,
        sqrt_one_minus_alphas,
        num_classes,
        batch_idx,
        class_mu: Dict[int, torch.Tensor] = None,
        r_obs: torch.Tensor = None,
    ):
        if batch_idx % self.config.generation_interval != 0:
            return torch.tensor(0.0, device=self.device)

        time_steps = torch.linspace(
            self.config.diffusion_steps - 1,
            0,
            self.config.ddim_steps + 1,
            dtype=torch.long,
            device=self.device,
        )
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=self.device)
        fake_features = torch.randn(batch_size, feature_dim, device=self.device)

        fake_features = self._ddim_sample(
            diffusion_model,
            fake_features,
            fake_labels,
            time_steps,
            sqrt_alphas,
            sqrt_one_minus_alphas,
        )

        if (
            getattr(self.config, "enable_stage3_calibration", True)
            and class_mu is not None
            and r_obs is not None
            and hasattr(self, "class_counts")
        ):
            r_cal = self.loss_calculator.compute_calibrated_radius(
                r_obs, self.r_prior, self.class_counts, self.tau, self.config.lambda_cal
            )

            calibration_strength = getattr(
                self.config, "stage3_calibration_strength", 0.5
            )
            fake_features = self._calibrate_features(
                fake_features,
                fake_labels,
                class_mu,
                r_cal,
                calibration_strength=calibration_strength,
            )

        fake_logits = classifier(fake_features)

        L_ge = F.cross_entropy(fake_logits, fake_labels)
        return L_ge

    def _validate(
        self, encoder, classifier, test_set, dataset_info, train_class_counts
    ):
        encoder.eval()
        classifier.eval()
        loss_fn = nn.CrossEntropyLoss()
        test_loss, correct, total = 0, 0, 0
        probs, logits_list, labels_list = [], [], []

        num_classes = dataset_info["class_num"]

        with torch.no_grad():
            for img, label in tqdm(test_set, desc="[GALD-DC] Testing"):
                img, label = img.to(self.device), label.to(self.device)
                labels_list.append(label)
                features = encoder.forward_no_fc(img)

                logits = classifier(features)

                test_loss += loss_fn(logits, label).item()

                logits_list.extend(list(logits.cpu().numpy()))

                prob = F.softmax(logits, dim=1)
                probs.extend(list(prob.cpu().numpy()))
                pred = prob.argmax(dim=1)
                correct += (pred == label).type(torch.float).sum().item()
                total += label.size(0)

        probs = np.array(probs)
        logits_array = np.array(logits_list)
        labels = torch.cat(labels_list)
        accuracy = correct / total
        test_loss /= len(test_set)

        _, mmf_acc = self._get_metrics(probs, labels, dataset_info["per_class_img_num"])

        bs_probs = balanced_softmax_probs(
            logits_array, cls_num_list=dataset_info["per_class_img_num"]
        )
        bs_acc, mmf_acc_bs = self._get_metrics(
            bs_probs, labels, dataset_info["per_class_img_num"]
        )
        bs_acc /= 100.0

        return test_loss, accuracy, bs_acc, mmf_acc, mmf_acc_bs

    def _update_stats_ema(self, real_features, labels, class_mu, r_obs, num_classes):
        unique_labels = torch.unique(labels)

        if torch.isnan(real_features).any() or torch.isinf(real_features).any():
            return

        real_features = torch.clamp(
            real_features, -self.config.feature_clamp_max, self.config.feature_clamp_max
        )

        with torch.no_grad():
            for cls in unique_labels:
                cls_idx = cls.item()
                if not (0 <= cls_idx < num_classes):
                    continue
                mask = labels == cls
                if mask.sum() == 0:
                    continue

                cls_feats = real_features[mask]
                cls_mean = cls_feats.mean(dim=0)
                old_proto = class_mu[cls_idx]
                if torch.isnan(old_proto).any():
                    old_proto = cls_mean

                new_proto = (
                    1 - self.config.lambda_ema
                ) * old_proto + self.config.lambda_ema * cls_mean
                class_mu[cls_idx] = new_proto

                if mask.sum() >= 1:
                    dists = torch.norm(cls_feats - new_proto, p=2, dim=1)
                    current_avg_radius = dists.mean()
                    old_radius = r_obs[cls_idx]
                    new_radius = (
                        1 - self.config.beta_radius
                    ) * old_radius + self.config.beta_radius * current_avg_radius
                    r_obs[cls_idx] = new_radius

    def _ddim_sample(
        self,
        diffusion_model,
        fake_features,
        fake_labels,
        time_steps,
        sqrt_alphas,
        sqrt_one_minus_alphas,
    ):
        batch_size = fake_features.size(0)
        estimated_clean = fake_features

        for i in range(self.config.ddim_steps):
            current_step = time_steps[i]
            next_step = time_steps[i + 1]

            fake_features_reshaped = fake_features.unsqueeze(1)
            batched_times = torch.full(
                (batch_size,), current_step, device=self.device, dtype=torch.long
            )

            predictions = diffusion_model.model_predictions(
                fake_features_reshaped, batched_times, fake_labels
            )
            estimated_clean = predictions.pred_x_start.squeeze(1)
            predicted_noise = predictions.pred_noise.squeeze(1)

            estimated_clean = torch.clamp(
                estimated_clean,
                -self.config.feature_clamp_max,
                self.config.feature_clamp_max,
            )

            if next_step >= 0:
                alpha_next = sqrt_alphas[next_step]
                sigma = torch.sqrt(
                    max(torch.tensor(0.0, device=self.device), 1 - alpha_next**2)
                )
                fake_features = alpha_next * estimated_clean + sigma * predicted_noise

        return estimated_clean

    def _calibrate_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        class_mu: Dict[int, torch.Tensor],
        r_cal: torch.Tensor,
        calibration_strength: float = 1.0,
    ) -> torch.Tensor:
        if calibration_strength <= 0.0:
            return features

        calibrated = features.clone()
        batch_size = features.size(0)

        for i in range(batch_size):
            cls_idx = labels[i].item()

            if cls_idx not in class_mu:
                continue
            if cls_idx >= len(r_cal):
                continue

            prototype = class_mu[cls_idx]
            target_radius = r_cal[cls_idx]

            if torch.isnan(prototype).any() or torch.isinf(prototype).any():
                continue
            if torch.isnan(target_radius) or torch.isinf(target_radius):
                continue
            if target_radius <= 0:
                continue

            direction = features[i] - prototype
            current_radius = torch.norm(direction, p=2)

            if current_radius < 1e-6:
                direction = torch.randn_like(direction)
                current_radius = torch.norm(direction, p=2)

            direction_normalized = direction / current_radius

            target_position = prototype + direction_normalized * target_radius

            calibrated[i] = (1 - calibration_strength) * features[
                i
            ] + calibration_strength * target_position

        return calibrated

    def _estimate_clean_features(
        self,
        diffusion_model,
        real_features,
        labels,
        sqrt_alphas,
        sqrt_one_minus_alphas,
        epoch=None,
    ):
        batch_size = real_features.size(0)

        start_t = 200
        end_t = self.config.diffusion_steps

        if (
            epoch is not None
            and epoch >= self.config.stage1_end_epoch
            and epoch < self.config.stage2_end_epoch
        ):
            stage_len = self.config.stage2_end_epoch - self.config.stage1_end_epoch
            progress = (epoch - self.config.stage1_end_epoch) / max(1, stage_len)

            current_max_t = int(start_t + (end_t - start_t) * progress)
            max_t = min(current_max_t, end_t)
        else:
            max_t = start_t

        t = torch.randint(0, max_t, (batch_size,), device=self.device)
        noise = torch.randn_like(real_features)

        noisy_features = (
            sqrt_alphas[t].view(-1, 1) * real_features
            + sqrt_one_minus_alphas[t].view(-1, 1) * noise
        )
        noisy_features_reshaped = noisy_features.unsqueeze(1)

        predictions = diffusion_model.model_predictions(
            noisy_features_reshaped, t, labels
        )
        estimated_clean_features = predictions.pred_x_start.squeeze(1)

        return estimated_clean_features

    def _safe_loss(self, loss_tensor, max_val=10.0):
        if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
            return torch.tensor(0.0, device=self.device)
        return torch.clamp(loss_tensor, max=max_val)

    def _load_data(self, cfg) -> Tuple[DataLoader, DataLoader, DataLoader, int, Dict]:
        if self.config.dataset == "CIFAR10" or self.config.dataset == "CIFAR100":
            dataset_info = Custom_dataset(self.config)
            train_set, val_set, test_set, dset_info = data_loader_wrapper_cust(
                dataset_info
            )
        elif self.config.dataset == "ImageNet":
            dataset_info = Custom_dataset_ImageNet(self.config)
            train_set, val_set, test_set, dset_info = data_loader_wrapper(cfg.dataset)

        num_classes = dset_info["class_num"]
        dataset_info = {
            "path": self.config.datapath,
            "class_num": num_classes,
            "dataset_name": self.config.dataset,
            "per_class_img_num": dset_info["per_class_img_num"],
        }

        return train_set, val_set, test_set, num_classes, dataset_info

    def _get_dynamic_loss_weights(self, epoch: int) -> Dict[str, float]:
        return {"lambda_sem": self.config.lambda_sem, "gamma_ge": self.config.gamma_ge}

    def _compute_auto_tau(
        self, class_counts: torch.Tensor, imb_factor: float, dataset: str
    ) -> int:
        counts = class_counts.cpu().numpy()
        max_count = np.max(counts)
        min_count = np.min(counts)

        tau = int(np.sqrt(max_count * min_count))

        tau = max(tau, 100) if dataset == "CIFAR10" else max(tau, 50)

        print(
            f">>> [Auto-tau] Dataset: {dataset}, Range: [{int(min_count)}, {int(max_count)}]"
        )
        print(
            f">>> [Auto-tau] Mathematical center (Geometric Mean): {int(np.sqrt(max_count * min_count))}"
        )
        print(f">>> [Auto-tau] Final selected tau: {tau}")

        return tau

    def _initialize_class_mu(
        self, num_classes: int, feature_dim: int
    ) -> Dict[int, torch.Tensor]:
        class_mu = {}
        for cls in range(num_classes):
            class_mu[cls] = torch.zeros(feature_dim).to(self.device)
        return class_mu

    def _get_diffusion_schedule(self) -> Tuple[torch.Tensor, torch.Tensor]:
        betas = torch.linspace(0.0001, 0.02, self.config.diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        return sqrt_alphas_cumprod.to(self.device), sqrt_one_minus_alphas_cumprod.to(
            self.device
        )

    def _compute_true_class_mu(
        self, encoder, dataloader, class_mu, num_classes, feature_dim
    ):
        class_features = [[] for _ in range(num_classes)]
        print(
            f">>> [GALD-DC] Computing initialization stats for {len(dataloader)} batches..."
        )
        encoder.eval()
        for inputs, labels in tqdm(dataloader, desc="Initializing Class Means"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            real_features = encoder.forward_no_fc(inputs)
            for i in range(len(labels)):
                cls_idx = labels[i].item()
                if 0 <= cls_idx < num_classes:
                    class_features[cls_idx].append(real_features[i].detach())

        for cls_idx in range(num_classes):
            if len(class_features[cls_idx]) > 0:
                features_tensor = torch.stack(class_features[cls_idx])
                class_mu[cls_idx] = torch.mean(features_tensor, dim=0).detach()
        return class_mu

    def _compute_initial_stats(self, encoder, dataloader, num_classes, feature_dim):
        class_features = [[] for _ in range(num_classes)]

        print(
            f"\n>>> [GALD-DC] One-time Initialization Pass: Collecting features for {len(dataloader)} batches..."
        )
        encoder.eval()
        for inputs, labels in tqdm(
            dataloader, desc="[GALD-DC] Init Stats (Mu + Radius)"
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            real_features = encoder.forward_no_fc(inputs)
            for i in range(len(labels)):
                cls_idx = labels[i].item()
                if 0 <= cls_idx < num_classes:
                    class_features[cls_idx].append(real_features[i].detach())

        print(">>> [GALD-DC] Processing collected features to compute mu and r_obs...")
        class_mu = {}
        r_obs = torch.zeros(num_classes, device=self.device)
        class_counts = torch.zeros(num_classes, device=self.device)

        for cls_idx in range(num_classes):
            if len(class_features[cls_idx]) > 0:
                features_stack = torch.stack(class_features[cls_idx])

                mu = torch.mean(features_stack, dim=0)
                class_mu[cls_idx] = mu

                dists = torch.norm(features_stack - mu, p=2, dim=1)
                r_obs[cls_idx] = dists.mean()
                class_counts[cls_idx] = len(class_features[cls_idx])
            else:
                class_mu[cls_idx] = torch.zeros(feature_dim, device=self.device)

        mask = class_counts > 0
        if mask.any():
            avg_radius = r_obs[mask].mean()
            r_obs[~mask] = avg_radius
        else:
            r_obs = self.config.target_radius * torch.ones(
                num_classes, device=self.device
            )

        print(">>> [GALD-DC] Initialization complete.")
        return class_mu, r_obs

    def _compute_initial_stats_from_latent(
        self,
        latent_features: torch.Tensor,
        latent_labels: torch.Tensor,
        num_classes: int,
        feature_dim: int,
    ):
        print(">>> [GALD-DC] Computing statistics from latent dataset...")

        latent_features = latent_features.to(self.device)
        latent_labels = latent_labels.to(self.device)

        class_features = [[] for _ in range(num_classes)]

        for i in range(len(latent_labels)):
            cls_idx = latent_labels[i].item()
            if 0 <= cls_idx < num_classes:
                class_features[cls_idx].append(latent_features[i])

        class_mu = {}
        r_obs = torch.zeros(num_classes, device=self.device)
        class_counts = torch.zeros(num_classes, device=self.device)

        for cls_idx in range(num_classes):
            if len(class_features[cls_idx]) > 0:
                features_stack = torch.stack(class_features[cls_idx])

                mu = torch.mean(features_stack, dim=0)
                class_mu[cls_idx] = mu

                dists = torch.norm(features_stack - mu, p=2, dim=1)
                r_obs[cls_idx] = dists.mean()
                class_counts[cls_idx] = len(class_features[cls_idx])
            else:
                class_mu[cls_idx] = torch.zeros(feature_dim, device=self.device)

        mask = class_counts > 0
        if mask.any():
            avg_radius = r_obs[mask].mean()
            r_obs[~mask] = avg_radius
        else:
            r_obs = self.config.target_radius * torch.ones(
                num_classes, device=self.device
            )

        print(">>> [GALD-DC] Statistics computation from latent dataset complete.")
        return class_mu, r_obs

    def _compute_true_class_mu(
        self, encoder, dataloader, class_mu, num_classes, feature_dim
    ):
        mu, _ = self._compute_initial_stats(
            encoder, dataloader, num_classes, feature_dim
        )
        return mu

    def _compute_r_obs_from_real_features(
        self, encoder, dataloader, class_mu, num_classes
    ):
        _, r_obs = self._compute_initial_stats(
            encoder, dataloader, num_classes, len(class_mu[0])
        )
        return r_obs

    def _control_gradients(
        self, encoder, classifier, diffusion_model, global_step, batch_idx
    ):
        self._normalize_gradients(encoder, classifier, diffusion_model)
        pass

    def _normalize_gradients(self, *models):
        for model in models:
            if model is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.max_grad_norm
                )

    def _get_metrics(self, probs, labels, cls_num_list):
        labels = [l.cpu().item() if isinstance(l, torch.Tensor) else l for l in labels]
        acc = acc_cal(probs, labels, method="top1")
        mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
        return acc, mmf_acc

    def _save_checkpoint(
        self,
        epoch,
        encoder,
        classifier,
        diffusion_model,
        optimizer,
        accuracy,
        model_type="ce",
    ):
        checkpoint = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "diffusion": diffusion_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "acc": accuracy,
        }

        mode = "ce"
        if model_type == "ce":
            path = f"ckpt_strategy_A_{mode}_best_ce.pth"
            print(
                f"Saved best CE checkpoint (epoch {epoch}, acc {accuracy:.4f}) to {path}"
            )
        else:  # model_type == 'pc'
            path = f"ckpt_strategy_A_{mode}_best_pc.pth"
            print(
                f"Saved best Label Shift checkpoint (epoch {epoch}, acc {accuracy:.4f}) to {path}"
            )

        torch.save(checkpoint, path)

    def _compute_train_accuracy(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        train_set: DataLoader,
        train_class_counts: torch.Tensor,
        num_classes: int,
    ) -> float:
        """
        Compute training set accuracy
        """
        encoder.eval()
        classifier.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in train_set:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                features = encoder.forward_no_fc(inputs)

                logits = classifier(features)

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        encoder.train()
        classifier.train()

        return correct / total if total > 0 else 0.0

    def _save_accuracy_history(
        self, avg_stage3_acc: float = None, avg_stage3_ls_acc: float = None
    ):
        """
        Save accuracy history
        """
        if (
            hasattr(self, "encoder")
            and hasattr(self, "classifier")
            and hasattr(self, "diffusion_model")
        ):
            self.monitor.save_best_checkpoints(
                self.encoder, self.classifier, self.diffusion_model, self.model_manager
            )

        self.monitor.log_training_complete(avg_stage3_acc, avg_stage3_ls_acc)
