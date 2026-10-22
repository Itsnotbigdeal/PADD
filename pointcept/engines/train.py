"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from packaging import version
from functools import partial
from pathlib import Path
from collections import OrderedDict
from itertools import chain
if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.models.diffusion_kd import (
    AdditiveDiffusionSchedule,
    PointwiseDiffusionDenoiser,
    pointwise_degradation_score,
)
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry
from pointcept.utils.visualization import (
    feat_to_rgb_pca,
    save_point_cloud,
)
TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class PointFeatureMemoryBank:
    """
    点云版特征 Memory Bank：
    - 维护一个大小为 [queue_size, C] 的队列，用于存储教师特征的 L2-normalized 向量。
    - 支持 FIFO 入队和随机采样，用于构造 Memory-based P2P 关系蒸馏。
    """

    def __init__(self, queue_size: int):
        self.queue_size = int(queue_size)
        self.queue: torch.Tensor | None = None
        self.ptr: int = 0
        self.current_size: int = 0

    def _init_queue(self, feat: torch.Tensor):
        """根据首次入队的特征形状和 device 初始化队列张量。"""
        if self.queue is not None:
            return
        assert feat.dim() == 2, "Feature for memory bank must be [N, C]"
        N, C = feat.shape
        device = feat.device
        self.queue = torch.zeros(self.queue_size, C, device=device, dtype=feat.dtype)
        self.ptr = 0
        self.current_size = 0

    def enqueue(self, feats: torch.Tensor):
        """
        feats: [K, C]，已做过 L2-normalize 的教师特征子集。
        采用环形缓冲区的方式进行 FIFO 入队。
        """
        if feats is None:
            return
        if feats.numel() == 0:
            return
        if feats.dim() != 2:
            feats = feats.view(feats.shape[0], -1)

        self._init_queue(feats)
        K = feats.shape[0]
        if K >= self.queue_size:
            # 只保留最后 queue_size 个
            feats = feats[-self.queue_size :]
            K = feats.shape[0]

        end = self.ptr + K
        if end <= self.queue_size:
            self.queue[self.ptr : end] = feats
        else:
            first_len = self.queue_size - self.ptr
            self.queue[self.ptr :] = feats[:first_len]
            remaining = K - first_len
            if remaining > 0:
                self.queue[:remaining] = feats[first_len:]
        self.ptr = (self.ptr + K) % self.queue_size
        self.current_size = min(self.current_size + K, self.queue_size)

    def sample(self, num_samples: int) -> torch.Tensor | None:
        """
        从当前队列中随机采样 num_samples 个特征（有放回），用于构造对比 embedding。
        当队列为空时返回 None。
        """
        if self.queue is None or self.current_size == 0:
            return None
        num_samples = int(num_samples)
        if num_samples <= 0:
            return None
        # 允许有放回采样，保证采样数量固定
        indices = torch.randint(
            low=0,
            high=self.current_size,
            size=(num_samples,),
            device=self.queue.device,
        )
        return self.queue[indices]


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def _diffkd_get(self, key, default=None):
        cfg = self.diffkd_cfg
        if isinstance(cfg, bool):
            return cfg if key == "enabled" else default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.diffkd_cfg = getattr(cfg, "diffkd", False)
        self.diffkd_enabled = self._diffkd_get("enabled", False)
        self.diffkd_training_mode = str(
            self._diffkd_get("training_mode", "joint")
        )
        if self.diffkd_training_mode not in {
            "joint",
            "diffusion_pretrain",
            "student_refine",
        }:
            raise ValueError(
                "diffkd.training_mode must be joint, diffusion_pretrain, or student_refine."
            )
        if self.diffkd_enabled:
            estimator = str(self._diffkd_get("estimator", "teacher_distance"))
            if estimator != "teacher_distance":
                raise ValueError(
                    "The current implementation supports only "
                    "diffkd.estimator='teacher_distance'."
                )
            policy = self._diffkd_get("timestep_policy", "discrepancy")
            if policy not in {"fixed", "discrepancy", "confidence", "shuffle"}:
                raise ValueError("Unknown diffkd.timestep_policy: " + str(policy))
            if policy == "fixed" and not 0 <= int(self._diffkd_get("fixed_timestep", 100)) <= int(self._diffkd_get("max_timesteps", 1000)):
                raise ValueError("diffkd.fixed_timestep outside schedule.")
            if int(self._diffkd_get("reverse_steps", 5)) < 1:
                raise ValueError("diffkd.reverse_steps must be positive.")
        self.diffusion_schedule = None
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.diff_model = self.build_diff_model() if self.diffkd_enabled else None
        
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self._gradient_accumulation_counter = 0

        # xiugai
        self.teacher_model = self.build_teacher_model()
        # xiugai

        # Memory-based P2P KD 配置与 Memory Bank
        self.m_p2p_enable = getattr(self.cfg, "m_p2p_enable", False)
        self.m_p2p_queue_size = getattr(self.cfg, "m_p2p_queue_size", 20000)
        self.m_p2p_sample_size = getattr(self.cfg, "m_p2p_sample_size", 2048)
        self.m_p2p_enqueue_k = getattr(self.cfg, "m_p2p_enqueue_k", 256)
        self.m_p2p_tau = float(getattr(self.cfg, "m_p2p_tau", 0.1))
        self.m_p2p_loss_weight = float(getattr(self.cfg, "m_p2p_loss_weight", 0.1))
        self.m_p2p_bank = PointFeatureMemoryBank(self.m_p2p_queue_size) if self.m_p2p_enable else None

        # Feature visualization
        self.feat_vis_enable = getattr(self.cfg, "feat_vis_enable", False)
        self.feat_vis_max_total = getattr(self.cfg, "feat_vis_max_total", 100)
        self.feat_vis_max_per_epoch = getattr(self.cfg, "feat_vis_max_per_epoch", 4)
        self.feat_vis_saved_count = 0
        self._last_vis_input_dict = None
        self._last_vis_kd_teacher = None
        self._last_vis_kd_student = None
        self._last_vis_kd_student_denoised = None

    def train(self):
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"================ Model Stats ================")
        self.logger.info(f"Total Parameters: {total_params / 1e6:.2f} M")     # 总参数量 (百万)
        self.logger.info(f"Trainable Parameters: {trainable_params / 1e6:.2f} M") # 可训练参数量
        if 'logits' in (self.cfg.kd_loss_type_1, self.cfg.kd_loss_type_2):
            self.logger.info(f"Using Logits-based KD with temperature {self.cfg.kd_temperature} and weight {self.cfg.kd_loss_weight_1}.")
        if 'feat' in (self.cfg.kd_loss_type_1, self.cfg.kd_loss_type_2):
            self.logger.info(f"Using Feature-based KD with weight {self.cfg.kd_loss_weight_2}.")
        if self.diffkd_enabled:
            self.logger.info(f"Using Diffusion KD with weight.")
            noise_w = self._diffkd_get("noise_loss_weight", 1.0)
            align_w = self._diffkd_get("align_loss_weight", 1.0)
            self.logger.info(f"DiffKD noise_w={noise_w}, align_w={align_w}.")
            self.logger.info(f"DiffKD training_mode={self.diffkd_training_mode}.")
            if self.cfg.cwkd:
                T_cw = float(getattr(self.cfg, "T", 4.0))
                self.logger.info(f"DiffKD with cwkd, temperature {T_cw}.")
            if self.m_p2p_enable:
                self.logger.info(
                    f"Using Memory-based P2P KD: queue_size={self.m_p2p_queue_size}, "
                    f"sample_size={self.m_p2p_sample_size}, enqueue_k={self.m_p2p_enqueue_k}, "
                    f"tau={self.m_p2p_tau}, weight={self.m_p2p_loss_weight}."
                )
        if self.feat_vis_enable:
            self.logger.info(
                f"Feature visualization enabled: max_total={self.feat_vis_max_total}, "
                f"max_per_epoch={self.feat_vis_max_per_epoch}."
            )
        self.logger.info(f"=============================================")
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                if self.diff_model is not None and self.diffkd_training_mode != "student_refine":
                    self.diff_model.train()
                elif self.diff_model is not None:
                    self.diff_model.eval()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Only clear gradients on first accumulation step
        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()

        self.teacher_model.eval()
        # xiugai: 教师模型不需要梯度
        with torch.no_grad():
            # with auto_cast(
            #     enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
            # ):
            _, kd_dict_teacher = self.teacher_model(input_dict,kd=True)  # xiugai

        # Forward pass
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict, kd_dict_student = self.model(input_dict,kd=True)  # xiugai
            total_loss = output_dict["loss"]
            if "logits" in (self.cfg.kd_loss_type_1, self.cfg.kd_loss_type_2):
                kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
                kd_loss_logit = kd_loss_fn(
                    torch.log_softmax(kd_dict_student["seg_logits"] / self.cfg.kd_temperature, dim=1),
                    torch.softmax(kd_dict_teacher["seg_logits"] / self.cfg.kd_temperature, dim=1)
                ) * (self.cfg.kd_temperature ** 2)
                total_loss = total_loss + self.cfg.kd_loss_weight_1 * kd_loss_logit
                output_dict["kd_logit_loss"] = kd_loss_logit.detach()
            if "feat" in (self.cfg.kd_loss_type_1, self.cfg.kd_loss_type_2):
                kd_loss_feat = nn.MSELoss()(kd_dict_student["feat"], kd_dict_teacher["feat"])
                total_loss = total_loss + self.cfg.kd_loss_weight_2 * kd_loss_feat
                output_dict["kd_feat_loss"] = kd_loss_feat.detach()

            if self.diffkd_enabled:
                # ================== Diffusion KD ==================
                teacher_feat = kd_dict_teacher["feat"].detach()  # [N, C]
                student_feat = kd_dict_student["feat"]          # [N, C]
                if teacher_feat.shape != student_feat.shape:
                    raise ValueError(
                        "Diffusion KD currently requires aligned teacher/student features, got "
                        f"{tuple(teacher_feat.shape)} and {tuple(student_feat.shape)}."
                    )
                offset = input_dict.get("offset", None)
                max_t = self.diffusion_schedule.num_timesteps

                train_prior = self.diffkd_training_mode in {
                    "joint",
                    "diffusion_pretrain",
                }
                refine_student = self.diffkd_training_mode in {
                    "joint",
                    "student_refine",
                }

                if train_prior:
                    diff_loss = self._train_diffusion_prior(teacher_feat, offset)
                    noise_w = float(self._diffkd_get("noise_loss_weight", 1.0))
                    total_loss = total_loss + noise_w * diff_loss
                    output_dict["diffusion_noise_loss"] = diff_loss.detach()

                student_denoised = None
                if refine_student:
                    degradation_score = pointwise_degradation_score(
                        student_feat, teacher_feat
                    )
                    from pointcept.models.diffusion_kd.assignment import assign_timesteps
                    start_t = assign_timesteps(
                        self.diffusion_schedule, degradation_score,
                        kd_dict_teacher.get("seg_logits"), offset,
                        policy=self._diffkd_get("timestep_policy", "discrepancy"),
                        fixed_timestep=int(self._diffkd_get("fixed_timestep", 100)),
                    )
                    student_denoised = self._denoise_student_feature(
                        student_feat, start_t, offset
                    )

                    T_cw = float(getattr(self.cfg, "T", 4.0))
                    if self.cfg.cwkd:
                        kd_loss_fn_feat = nn.KLDivLoss(reduction="batchmean")
                        log_p_s = torch.log_softmax(student_denoised / T_cw, dim=1)
                        p_t = torch.softmax(teacher_feat / T_cw, dim=1)
                        align_loss = kd_loss_fn_feat(log_p_s, p_t) * (T_cw ** 2)
                    else:
                        align_loss = F.mse_loss(student_denoised, teacher_feat)

                    align_w = float(self._diffkd_get("align_loss_weight", 1.0))
                    total_loss = total_loss + align_w * align_loss
                    output_dict["diffusion_align_loss"] = align_loss.detach()
                    output_dict["diffusion_timestep_mean"] = (
                        start_t.float().mean().detach()
                    )
                    output_dict["diffusion_degradation_mean"] = (
                        degradation_score.mean().detach()
                    )

                    m_p2p_loss = self._compute_memory_p2p_loss(
                        student_denoised, teacher_feat
                    )
                    if m_p2p_loss is not None:
                        total_loss = (
                            total_loss + self.m_p2p_loss_weight * m_p2p_loss
                        )
                        output_dict["m_p2p_loss"] = m_p2p_loss.detach()
                # ==================================================

                # Cache denoised student features for visualization
                if (
                    self.feat_vis_enable
                    and comm.is_main_process()
                    and student_denoised is not None
                ):
                    self._last_vis_kd_student_denoised = {
                        "feat": student_denoised.detach().cpu()
                    }
                else:
                    self._last_vis_kd_student_denoised = None
            else:
                self._last_vis_kd_student_denoised = None

            output_dict["loss"] = total_loss.detach()
            loss = total_loss / self.cfg.gradient_accumulation_steps

         
        # Backward pass
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._gradient_accumulation_counter += 1

        # Perform optimizer step only when enough gradients have accumulated
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            if self.cfg.enable_amp:
                self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_grad is not None:
                    # 合并主模型和diff_model的参数进行梯度裁剪
                    if hasattr(self, 'diff_model') and self.diff_model is not None:
                        all_params = chain(self.model.parameters(), self.diff_model.parameters())
                    else:
                        all_params = self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(
                        all_params, self.cfg.clip_grad
                    )
                self.scaler.step(self.optimizer)

                # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
                # Fix torch warning scheduler step before optimizer step.
                scale = self.scaler.get_scale()
                self.scaler.update()
                if scale <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                if self.cfg.clip_grad is not None:
                    # 合并主模型和diff_model的参数进行梯度裁剪
                    if hasattr(self, 'diff_model') and self.diff_model is not None:
                        all_params = chain(self.model.parameters(), self.diff_model.parameters())
                    else:
                        all_params = self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(
                        all_params, self.cfg.clip_grad
                    )
                self.optimizer.step()
                self.scheduler.step()

            # Reset grad accumulation counter
            self._gradient_accumulation_counter = 0

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

        # Cache last batch for feature visualization (after_epoch)
        if self.feat_vis_enable and comm.is_main_process():
            self._last_vis_input_dict = {
                "coord": input_dict["coord"].detach().cpu(),
                "offset": input_dict["offset"].detach().cpu(),
            }
            self._last_vis_kd_teacher = {"feat": kd_dict_teacher["feat"].detach().cpu()}
            self._last_vis_kd_student = {"feat": kd_dict_student["feat"].detach().cpu()}

    def _train_diffusion_prior(self, teacher_feat, offset):
        max_t = self.diffusion_schedule.num_timesteps
        train_t = torch.randint(
            1,
            max_t + 1,
            (teacher_feat.shape[0],),
            device=teacher_feat.device,
        )
        teacher_noisy, noise = self.diffusion_schedule.q_sample(
            teacher_feat, train_t
        )
        predicted_noise = self.diff_model(
            teacher_noisy,
            train_t.to(teacher_noisy.dtype),
            offset,
        )
        return F.mse_loss(predicted_noise, noise)

    def _denoise_student_feature(self, student_feat, start_t, offset):
        reverse_steps = int(self._diffkd_get("reverse_steps", 5))
        deterministic = bool(self._diffkd_get("deterministic", True))
        denoised = student_feat
        for current_t, next_t in self.diffusion_schedule.reverse_time_pairs(
            start_t, reverse_steps
        ):
            predicted_noise = self.diff_model(
                denoised,
                current_t.to(denoised.dtype),
                offset,
            )
            denoised = self.diffusion_schedule.reverse_step(
                denoised,
                current_t,
                next_t,
                predicted_noise,
                deterministic=deterministic,
            )
        return denoised

    def _compute_memory_p2p_loss(self, student_feat, teacher_feat):
        if not self.m_p2p_enable or self.m_p2p_bank is None:
            return None
        teacher_normalized = F.normalize(teacher_feat, p=2, dim=1)
        student_normalized = F.normalize(student_feat, p=2, dim=1)

        enqueue_count = min(self.m_p2p_enqueue_k, teacher_normalized.shape[0])
        if enqueue_count > 0:
            indices = torch.randperm(
                teacher_normalized.shape[0], device=teacher_normalized.device
            )[:enqueue_count]
            self.m_p2p_bank.enqueue(teacher_normalized[indices])

        memory = self.m_p2p_bank.sample(self.m_p2p_sample_size)
        if memory is None:
            return None
        memory = F.normalize(memory, p=2, dim=1)
        teacher_relation = teacher_normalized @ memory.t()
        student_relation = student_normalized @ memory.t()
        return F.kl_div(
            F.log_softmax(student_relation / self.m_p2p_tau, dim=1),
            F.softmax(teacher_relation / self.m_p2p_tau, dim=1),
            reduction="batchmean",
        )

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

        # Feature visualization: save teacher/student features as PLY per epoch
        if (
            self.feat_vis_enable
            and comm.is_main_process()
            and self.feat_vis_saved_count < self.feat_vis_max_total
            and self._last_vis_input_dict is not None
            and self._last_vis_kd_teacher is not None
            and self._last_vis_kd_student is not None
        ):
            vis_dir = os.path.join(self.cfg.save_path, "feat_vis")
            os.makedirs(vis_dir, exist_ok=True)
            coord = self._last_vis_input_dict["coord"]
            offset = self._last_vis_input_dict["offset"]
            teacher_feat = self._last_vis_kd_teacher["feat"]
            student_feat = self._last_vis_kd_student["feat"]
            student_denoised_feat = (
                self._last_vis_kd_student_denoised["feat"]
                if self._last_vis_kd_student_denoised is not None
                else None
            )
            num_samples = len(offset)
            files_saved_this_epoch = 0
            for i in range(num_samples):
                if files_saved_this_epoch >= self.feat_vis_max_per_epoch:
                    break
                if self.feat_vis_saved_count >= self.feat_vis_max_total:
                    break
                start = int(offset[i - 1]) if i > 0 else 0
                end = int(offset[i])
                coord_i = coord[start:end].numpy()
                feat_t = teacher_feat[start:end]
                feat_s = student_feat[start:end]
                color_t = feat_to_rgb_pca(feat_t)
                color_s = feat_to_rgb_pca(feat_s)
                path_t = os.path.join(
                    vis_dir, f"epoch{self.epoch}_sample{i}_teacher.ply"
                )
                path_s = os.path.join(
                    vis_dir, f"epoch{self.epoch}_sample{i}_student.ply"
                )
                save_point_cloud(coord_i, color_t, path_t, self.logger)
                save_point_cloud(coord_i, color_s, path_s, self.logger)
                self.feat_vis_saved_count += 2
                files_saved_this_epoch += 2
                if student_denoised_feat is not None:
                    if self.feat_vis_saved_count >= self.feat_vis_max_total:
                        continue
                    if files_saved_this_epoch >= self.feat_vis_max_per_epoch:
                        continue
                    feat_sd = student_denoised_feat[start:end]
                    color_sd = feat_to_rgb_pca(feat_sd)
                    path_sd = os.path.join(
                        vis_dir, f"epoch{self.epoch}_sample{i}_student_denoised.ply"
                    )
                    save_point_cloud(coord_i, color_sd, path_sd, self.logger)
                    self.feat_vis_saved_count += 1
                    files_saved_this_epoch += 1

    def build_model(self):
        model = build_model(self.cfg.model)
        
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_diff_model(self):
        feat_dim = getattr(self.cfg.model, "backbone_out_channels", None)
        if feat_dim is None:
            feat_dim = getattr(self.cfg.teacher_model, "backbone_out_channels", None)
        if feat_dim is None:
            feat_dim = getattr(self.cfg, "backbone_out_channels", None)
        if feat_dim is None:
            raise ValueError(
                "Cannot determine the aligned feature dimension for diffusion KD."
            )

        self.diffusion_schedule = AdditiveDiffusionSchedule(
            num_timesteps=int(self._diffkd_get("max_timesteps", 1000)),
            beta_schedule=str(self._diffkd_get("beta_schedule", "linear")),
            beta_start=float(self._diffkd_get("beta_start", 1e-4)),
            beta_end=float(self._diffkd_get("beta_end", 2e-2)),
            gamma_max=float(self._diffkd_get("gamma_max", 1.0)),
        ).cuda()

        diff_model = PointwiseDiffusionDenoiser(
            feat_dim=int(feat_dim),
            time_dim=int(self._diffkd_get("time_dim", 64)),
            hidden_dim=int(self._diffkd_get("hidden_dim", int(feat_dim) * 4)),
            num_heads=int(self._diffkd_get("num_heads", 4)),
            num_blocks=int(self._diffkd_get("num_blocks", 2)),
            mlp_ratio=float(self._diffkd_get("mlp_ratio", 4.0)),
            window_size=int(self._diffkd_get("window_size", 256)),
            drop_path=float(self._diffkd_get("drop_path", 0.1)),
        )
        diff_model = diff_model.cuda()
        if self.diffkd_training_mode == "student_refine":
            for parameter in diff_model.parameters():
                parameter.requires_grad = False
        else:
            diff_model = create_ddp_model(
                diff_model,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        n_parameters = sum(
            parameter.numel()
            for parameter in diff_model.parameters()
            if parameter.requires_grad
        )
        self.logger.info(
            "Diffusion model params: %.2f M; process=additive; gamma_max=%.6f",
            n_parameters / 1e6,
            self.diffusion_schedule.gammas[-1].item(),
        )
        return diff_model

    # xiugai
    def build_teacher_model(self):
        model = build_model(self.cfg.teacher_model)
        # 修改 教师模型 不使用 sync bn
        # if self.cfg.sync_bn:
        #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model.cuda()
        # 加载教师模型权重
        ckpt_path = getattr(self.cfg, "teacher_model_ckpt", None)

        if ckpt_path:
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # 移除module.开头的内容
                    if k.startswith("module."):
                        name = k[7:] 
                    else:
                        name = k
                    new_state_dict[name] = v
                
                state_dict = new_state_dict
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                
                # 打印日志
                self.logger.info(
                    f"Loaded teacher checkpoint: {ckpt_path}, "
                    f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
                )
                
                if len(missing) > 0:
                    self.logger.warning(f"Missing keys details: {missing}")

            else:
                self.logger.warning(f"Teacher checkpoint not found: {ckpt_path}")
        model.cuda()
        return model


    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        # 如果存在 diff_model，需要合并两个模型的参数
        if hasattr(self, 'diff_model') and self.diff_model is not None:
            # 创建一个包装类来合并两个模型的参数
            class ParamWrapper:
                def __init__(self, model, diff_model):
                    self.model = model
                    self.diff_model = diff_model
                
                def parameters(self):
                    """返回合并的参数迭代器"""
                    return chain(self.model.parameters(), self.diff_model.parameters())
                
                def named_parameters(self):
                    """返回合并的命名参数迭代器，diff_model的参数添加'diff_model.'前缀"""
                    # 主模型的参数
                    for name, param in self.model.named_parameters():
                        yield name, param
                    # diff_model的参数，添加前缀以区分
                    for name, param in self.diff_model.named_parameters():
                        yield f"diff_model.{name}", param
            
            wrapper = ParamWrapper(self.model, self.diff_model)
            return build_optimizer(self.cfg.optimizer, wrapper, self.cfg.param_dicts)
        else:
            return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader)
            * self.cfg.eval_epoch
            // self.cfg.gradient_accumulation_steps
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
