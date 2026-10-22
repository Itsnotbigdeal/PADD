"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
import sys
# 获取当前脚本文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设 tools 在根目录下一级，所以是父目录）
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    cfg.kd_loss_type_1 = args.kd_loss_type_1
    cfg.kd_loss_type_2 = args.kd_loss_type_2
    cfg.kd_loss_weight_1 = args.kd_loss_weight_1
    cfg.kd_loss_weight_2 = args.kd_loss_weight_2
    cfg.kd_temperature = args.kd_temperature
    if not hasattr(cfg, "diffkd"):
        cfg.diffkd = dict(enabled=bool(args.diffkd))
    elif args.diffkd:
        if isinstance(cfg.diffkd, bool):
            cfg.diffkd = dict(enabled=True)
        else:
            cfg.diffkd.enabled = True
    if args.diffkd_training_mode is not None:
        if isinstance(cfg.diffkd, bool):
            cfg.diffkd = dict(
                enabled=cfg.diffkd,
                training_mode=args.diffkd_training_mode,
            )
        else:
            cfg.diffkd.training_mode = args.diffkd_training_mode
    cfg.cwkd = args.cwkd
    if not isinstance(cfg.diffkd, bool):
        cfg.diffkd.noise_loss_weight = getattr(args, "noise_w", 1.0)
        cfg.diffkd.align_loss_weight = getattr(args, "align_w", 1.0)
    # Memory-based P2P KD 配置
    cfg.m_p2p_enable = getattr(args, "m_p2p_enable", False)
    cfg.m_p2p_queue_size = getattr(args, "m_p2p_queue_size", 20000)
    cfg.m_p2p_sample_size = getattr(args, "m_p2p_sample_size", 2048)
    cfg.m_p2p_enqueue_k = getattr(args, "m_p2p_enqueue_k", 256)
    cfg.m_p2p_tau = getattr(args, "m_p2p_tau", 0.1)
    cfg.m_p2p_loss_weight = getattr(args, "m_p2p_loss_weight", 0.1)
    # Feature visualization
    cfg.feat_vis_enable = getattr(args, "feat_vis", False)
    cfg.feat_vis_max_total = getattr(args, "feat_vis_max_total", 100)
    cfg.feat_vis_max_per_epoch = getattr(args, "feat_vis_max_per_epoch", 4)
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
