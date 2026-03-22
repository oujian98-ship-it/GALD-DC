import torch
from typing import Dict


class TrainingConfig:
    def __init__(self, args):
        self.datapath = args.datapath
        self.config = args.config
        self.dataset = args.dataset
        self.imb_factor = args.imb_factor
        self.model_fixed = args.model_fixed
        self.checkpoint = getattr(args, "checkpoint", None)
        self.diffusion_steps = args.diffusion_step
        self.lr = getattr(args, "lr", 0.01)
        self.lambda_ema = getattr(args, "lambda_ema", 0.2)
        self.eta_p = getattr(args, "eta_p", 0.1)
        self.eta_r = getattr(args, "eta_r", 0.1)
        self.lambda_sem = getattr(args, "lambda_sem", 0.01)
        self.gamma_ge = getattr(args, "gamma_ge", 0.15)
        self.generation_interval = getattr(args, "generation_interval", 5)
        self.ddim_steps = getattr(args, "ddim_steps", 100)
        self.use_radius_constraint = getattr(args, "use_radius_constraint", True)
        self.target_radius = getattr(args, "target_radius", 1.0)
        self.tau = getattr(args, "tau", -1)
        self.lambda_cal = getattr(args, "lambda_cal", 0.3)
        self.beta_radius = getattr(args, "beta_radius", 0.2)
        self.eta_m = getattr(args, "eta_m", 0.0)
        self.margin_m = getattr(args, "margin_m", 3.5)
        self.stage3_mode = getattr(args, "stage3_mode", "hybrid")
        self.beta_cons = getattr(args, "beta_cons", 0.3)
        self.gamma_pseudo = getattr(args, "gamma_pseudo", 0.6)
        self.stage1_end_epoch = getattr(args, "stage1_end_epoch", 200)
        self.stage2_epochs = getattr(args, "stage2_epochs", 50)
        self.stage3_epochs = getattr(args, "stage3_epochs", 200)
        self.stage2_end_epoch = self.stage1_end_epoch + self.stage2_epochs
        self.epochs = self.stage2_end_epoch + self.stage3_epochs
        self.enable_dynamic_stage1 = getattr(args, "enable_dynamic_stage1", True)
        self.convergence_window = getattr(args, "convergence_window", 15)
        self.convergence_threshold = getattr(args, "convergence_threshold", 0.005)
        self.min_stage1_epochs = getattr(args, "min_stage1_epochs", 50)
        self.max_loss_weight = 5.0
        self.max_diffusion_loss = 10.0
        self.max_cov_loss = 5.0
        self.max_radius_loss = 50.0
        self.max_L_semantic = 20.0
        self.max_grad_norm = 0.9
        self.max_L_ge = 50.0
        self.feature_clamp_min = -10.0
        self.feature_clamp_max = 10.0
        self.gradient_accumulation_steps = 4
        self.weight_decay = 2e-3
        self.gradient_clipping_enabled = False
        self.adaptive_lr_factor = 0.5
        self.learning_rate_warmup_steps = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_stage3_calibration = getattr(
            args, "enable_stage3_calibration", True
        )
        self.stage3_calibration_strength = getattr(
            args, "stage3_calibration_strength", 0.5
        )
        self.radius_slack = getattr(args, "radius_slack", 0.5)
        self.ema_warmup_epochs = getattr(args, "ema_warmup_epochs", 10)
        self.margin_top_k = getattr(args, "margin_top_k", 3)
        self.enable_lora = getattr(args, "enable_lora", True)
        self.lora_rank = getattr(args, "lora_rank", 4)
        self.lora_alpha = getattr(args, "lora_alpha", 8.0)

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TrainingConfig":
        class Args:
            pass

        args = Args()
        for key, value in config_dict.items():
            setattr(args, key, value)

        return cls(args)

    def to_json(self) -> str:
        import json

        serializable = {}
        for key, value in self.__dict__.items():
            if key == "device":
                serializable[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable[key] = value

        return json.dumps(serializable, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "TrainingConfig":
        import json

        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    def update(self, key: str, value) -> None:
        setattr(self, key, value)

    def log_config(self) -> None:
        import logging

        lines = [
            "=" * 60,
            "GALD-DC Training Configuration",
            "=" * 60,
            f"dataset: {self.dataset}",
            f"imb_factor: {self.imb_factor}",
            f"epochs: {self.epochs}",
            f"learning_rate: {self.lr}",
            f"weight_decay: {self.weight_decay}",
            f"diffusion_steps: {self.diffusion_steps}",
            f"ddim_steps: {self.ddim_steps}",
            "-" * 60,
            "Loss Weights:",
            f"  lambda_ema: {self.lambda_ema}",
            f"  beta_radius: {getattr(self, 'beta_radius', self.lambda_ema)}",
            f"  eta_p (prototype loss): {self.eta_p}",
            f"  eta_r (radius loss): {self.eta_r}",
            f"  eta_m (margin loss): {self.eta_m}",
            f"  lambda_sem: {self.lambda_sem}",
            f"  gamma_ge (gen loss): {self.gamma_ge}",
            "-" * 60,
            "GALD-DC Parameters:",
            f"  tau (head/tail threshold): {self.tau}",
            f"  lambda_cal (calibration factor): {self.lambda_cal}",
            f"  margin_m (margin distance): {self.margin_m}",
            f"  stage3_mode: {self.stage3_mode}",
            f"  beta_cons (consistency weight): {self.beta_cons}",
            f"  gamma_pseudo (pseudo loss weight): {self.gamma_pseudo}",
            "-" * 60,
            "Stage 3 Explicit Calibration:",
            f"  enable_stage3_calibration: {self.enable_stage3_calibration}",
            f"  stage3_calibration_strength: {self.stage3_calibration_strength}",
            "-" * 60,
            "Three-Stage Training:",
            f"  Stage 1 (CE Pre-training): epochs 0-{self.stage1_end_epoch - 1}",
            f"  Stage 2 (Diffusion Training): epochs {self.stage1_end_epoch}-{self.stage2_end_epoch - 1}",
            f"  Stage 3 (Controlled Fine-tuning): epochs {self.stage2_end_epoch}-{self.epochs - 1}",
            "-" * 60,
            f"  use_radius_constraint: {self.use_radius_constraint}",
            f"  generation_interval: {self.generation_interval}",
            "=" * 60,
        ]

        for line in lines:
            print(line)
            logging.info(line)
