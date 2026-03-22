"""
Training monitor module
Responsible for logging, progress tuning, and model saving during training
"""

import os
import logging
from datetime import datetime
from typing import Dict, List


from .config import TrainingConfig


class TrainingMonitor:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_logging()
        self.best_accuracy = 0.0
        self.best_bs_acc = 0.0
        self.accuracies_history = []

    def _setup_logging(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_filename = os.path.join(
            logs_dir,
            f"{self.config.dataset}_{self.config.imb_factor}_{current_time}.log",
        )
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def log_batch_progress(
        self, epoch: int, batch_idx: int, losses: Dict[str, float], stage: int = 3
    ):
        if batch_idx % 100 == 0:
            if stage == 1:
                print(
                    f" Epoch: {epoch}, Batch: {batch_idx}, CE Loss: {losses['real']:.4f}"
                )
            elif stage == 2:
                print(
                    f" Epoch: {epoch}, Batch: {batch_idx}, "
                    f"L_LDM: {losses.get('diffusion', 0):.4f}, "
                    f"L_proto: {losses.get('prototype', 0):.4f}, "
                    f"L_rad: {losses.get('radius', 0):.4f}, "
                    f"L_margin: {losses.get('margin', 0):.4f}"
                )
            else:
                print(
                    f" Epoch: {epoch}, Batch: {batch_idx}, "
                    f"Real: {losses['real']:.4f}, Gen: {losses['gen']:.4f}, "
                    f"Cons: {losses.get('consistency', 0):.4f}"
                )

    def log_epoch_summary(
        self,
        epoch: int,
        avg_losses: Dict[str, float],
        train_accuracy: float = None,
        train_loss: float = None,
        stage: int = 3,
    ):
        if stage == 1:
            msg = f" Epoch {epoch} - CE Loss: {avg_losses['real']:.4f}"
            print(msg)
            logging.info(msg)

        elif stage == 2:
            msg = (
                f"Epoch {epoch} - "
                f"L_LDM: {avg_losses.get('diffusion', 0):.4f}, "
                f"L_proto: {avg_losses.get('prototype', 0):.4f}, "
                f"L_rad: {avg_losses.get('radius', 0):.4f}, "
                f"L_margin: {avg_losses.get('margin', 0):.4f}, "
                f"Total: {avg_losses['total']:.4f}"
            )
            print(msg)
            logging.info(msg)

        else:
            msg = (
                f"Epoch {epoch} Summary - "
                f"Real Loss: {avg_losses['real']:.4f}, "
                f"Gen Loss: {avg_losses['gen']:.4f}, "
                f"Consistency: {avg_losses.get('consistency', 0):.4f}, "
                f"Total: {avg_losses['total']:.4f}"
            )
            print(msg)
            logging.info(msg)

    def log_validation(
        self,
        epoch: int,
        accuracy: float,
        test_loss: float,
        bs_acc: float,
        mmf_acc: List,
        mmf_acc_bs: List,
        mode: str = "Val",
    ):
        method_name = "CE"
        method_display = f"{method_name} " if method_name == "CE" else method_name
        set_name = "Validation" if mode == "Val" else "Test"

        self.accuracies_history.append(
            {
                "epoch": epoch,
                "mode": mode,
                "method": method_name,
                "base_accuracy": accuracy,
                "accuracy": accuracy,
                "bs_acc": bs_acc,
                "base_mmf": mmf_acc,
                "mmf_acc": mmf_acc,
                "bs_mmf": mmf_acc_bs,
                "mmf_acc_bs": mmf_acc_bs,
            }
        )

        print(f"Epoch {epoch} {set_name} Results")
        print(f"{method_display} Results:")
        print(f"  {set_name} Loss:       {test_loss:.4f}")
        print(f"  {set_name} Acc:        {100 * accuracy:.2f}%")
        print(
            f"  MMF Acc:         [Many: {mmf_acc[0]:.2f}%, Med: {mmf_acc[1]:.2f}%, Few: {mmf_acc[2]:.2f}%, Overall: {mmf_acc[3]:.2f}%]"
        )

        print("BS (Balanced Softmax) Results:")
        print(f"  {set_name} Acc:        {100 * bs_acc:.2f}%")
        print(
            f"  MMF Acc:         [Many: {mmf_acc_bs[0]:.2f}%, Med: {mmf_acc_bs[1]:.2f}%, Few: {mmf_acc_bs[2]:.2f}%, Overall: {mmf_acc_bs[3]:.2f}%]\n"
        )

        logging.info(f"Epoch {epoch} {set_name} Results - {method_display}")
        logging.info(
            f"  {set_name} Accuracy: {100 * accuracy:.2f}%, Loss: {test_loss:.4f}"
        )
        logging.info(
            f"  {set_name} MMF: [Many: {mmf_acc[0]:.2f}%, Med: {mmf_acc[1]:.2f}%, Few: {mmf_acc[2]:.2f}%, Overall: {mmf_acc[3]:.2f}%]"
        )
        logging.info(f"  BS (Balanced Softmax) Accuracy: {100 * bs_acc:.2f}%")
        logging.info(
            f"  BS MMF: [Many: {mmf_acc_bs[0]:.2f}%, Med: {mmf_acc_bs[1]:.2f}%, Few: {mmf_acc_bs[2]:.2f}%, Overall: {mmf_acc_bs[3]:.2f}%]"
        )

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(
                f" New best {set_name} {method_display} accuracy: {100 * accuracy:.2f}%\n"
            )
            logging.info(f"New best {set_name} accuracy: {100 * accuracy:.2f}%")

    def get_accuracy_history(self) -> List[Dict]:
        return self.accuracies_history

    def get_best_accuracy(self) -> float:
        return self.best_accuracy

    def save_best_checkpoints(
        self, encoder, classifier, diffusion_model, model_manager
    ):
        if self.accuracies_history:
            best_epoch = max(self.accuracies_history, key=lambda x: x["base_accuracy"])
            best_accuracy = best_epoch["base_accuracy"]
            best_bs_acc = best_epoch["bs_acc"]

            model_manager.save_best_models(
                encoder, classifier, diffusion_model, best_accuracy, best_bs_acc
            )

    def log_training_complete(
        self, avg_stage3_acc: float = None, avg_stage3_bs_acc: float = None
    ):
        if self.accuracies_history:
            method_name = "CE"
            best_epoch = max(self.accuracies_history, key=lambda x: x["base_mmf"][3])
            best_accuracy = best_epoch["base_accuracy"]
            best_bs_acc = best_epoch["bs_acc"]
            best_mmf_acc = best_epoch["base_mmf"]
            best_mmf_acc_bs = best_epoch["bs_mmf"]

            best_bs_epoch = max(self.accuracies_history, key=lambda x: x["bs_acc"])
            best_bs_only = best_bs_epoch["bs_acc"]
            best_mmf_acc_bs_best = best_bs_epoch["bs_mmf"]

            print(f"\n{'='*70}")
            print("Training Complete - Final Results")

            method_display = (
                f"{method_name} + Ours" if method_name == "CE" else method_name
            )

            print(f"Best {method_display}  :")
            print(f"  Accuracy:        {100 * best_accuracy:.2f}%")
            print(
                f"  MMF Acc:         [Many: {best_mmf_acc[0]:.2f}%, Med: {best_mmf_acc[1]:.2f}%, Few: {best_mmf_acc[2]:.2f}%, Overall: {best_mmf_acc[3]:.2f}%]"
            )

            print("Best BS (Balanced Softmax)  :")
            print(f"  Accuracy:        {100 * best_bs_only:.2f}%")
            print(
                f"  MMF Acc:         [Many: {best_mmf_acc_bs_best[0]:.2f}%, Med: {best_mmf_acc_bs_best[1]:.2f}%, Few: {best_mmf_acc_bs_best[2]:.2f}%, Overall: {best_mmf_acc_bs_best[3]:.2f}%]"
            )

            if avg_stage3_acc is not None:
                print(f"  Avg Test Accuracy:  {100 * avg_stage3_acc:.2f}%")
                print(f"  Avg Test BS:        {100 * avg_stage3_bs_acc:.2f}%")

            logging.info(f"Training Complete - Method: {method_display}")
            logging.info(
                f"  Best {method_display} Accuracy: {100 * best_accuracy:.2f}%"
            )
            logging.info(
                f"  Best {method_display} MMF: [Many: {best_mmf_acc[0]:.2f}%, Med: {best_mmf_acc[1]:.2f}%, Few: {best_mmf_acc[2]:.2f}%, Overall: {best_mmf_acc[3]:.2f}%]"
            )
            logging.info(
                f"  Best BS (Balanced Softmax) Accuracy: {100 * best_bs_only:.2f}%"
            )
            logging.info(
                f"  Best BS MMF: [Many: {best_mmf_acc_bs_best[0]:.2f}%, Med: {best_mmf_acc_bs_best[1]:.2f}%, Few: {best_mmf_acc_bs_best[2]:.2f}%, Overall: {best_mmf_acc_bs_best[3]:.2f}%]"
            )

            if avg_stage3_acc is not None:
                logging.info(f"  Average Test Accuracy: {100 * avg_stage3_acc:.2f}%")
                logging.info(
                    f"  Average Test BS Accuracy: {100 * avg_stage3_bs_acc:.2f}%"
                )
