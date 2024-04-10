# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco128.yaml imgsz=640 epochs=100 batch=16
"""

import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg_yaml import creat_args, creat_save_dir
from ultralytics.data.utils import check_cls_dataset, check_dataset
from ultralytics.nn.tasks_model import load_pytorch_model, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_PARAM,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
)

""" 训练的一些参数 保存路径 """
class Engine_Trainer: #分类检测都可以用

    def __init__(self, default_param=DEFAULT_PARAM, overrides=None, _callbacks=None): #成员属性 数据集检查 保存东西  
        # 创建参数
        self.args = creat_args(default_param, overrides) #大参数


        self.check_resume(overrides) #用到的自身的参数
        self.device = select_device(self.args.device, self.args.batch)
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        if RANK == -1:
            print_args(vars(self.args))
        # 设置设备
        if self.device.type in ("cpu", "mps"):
            self.args.workers = 0  # 在 CPU 上加快训练速度，因为时间主要花费在推断上，而不是数据加载

        try:
            if self.args.task_name == "classify":
                self.data_dict = check_cls_dataset(self.args.data_str)
            elif self.args.data_str.split(".")[-1] in ("yaml", "yml") or self.args.task_name in (
                "detect",
                "segment",
                "pose",
                "obb",
            ):
                self.data_dict = check_dataset(self.args.data_str)
                if "yaml_file" in self.data_dict:
                    self.args.data_str = self.data_dict["yaml_file"]  # 用于验证 'yolo train data=url.zip' 的用法
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data_str)}' error ❌ {e}")) from e
        
        # 创建保存目录
        self.save_dir = creat_save_dir(self.args) #保存路径 任务名 名字
        self.args.name = self.save_dir.name  # 更具模式i添加名字
        self.weights_path = self.save_dir / "weights"  # 权重目录


        # 在主进程中创建保存目录
        if RANK in (-1, 0):
            self.args.save_dir = str(self.save_dir) #'runs/detect/train12'
            self.weights_path.mkdir(parents=True, exist_ok=True)  # 创建目录
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # 保存运行参数
        self.last, self.best = self.weights_path / "last.pt", self.weights_path / "best.pt"  # 检查点路径
        self.save_period = self.args.save_period

        self.epochs = self.args.epochs
        self.start_epoch = 0
        self.batch_size = self.args.batch

        self.model = None  # 初始化时可以传入 model，如果没有则为 None
        self.validator = None
        self.model_str = check_stem(self.args.model_str)  # 添加后缀，例如 yolov8n -> yolov8n.pt
        self.trainset, self.testset = self.get_dataset(self.data_dict)
        self.ema = None

        # 优化工具的初始化
        self.lf = None
        self.scheduler = None

        # 每个 epoch 的指标
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]
        self.plots = {}
        # 回调函数
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)


    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def prepare_train(self): #爸爸
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch == -1:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._normally_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            "nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _get_dataloaders_optimizer(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        if world_size > 1:
            self._setup_ddp(world_size)




        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()




        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # 带有这个后缀就冻结
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients 不要求梯度但是是浮点
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True



        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # Automatic Mixed Precision，即自动混合精度 
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)# 自动缩放梯度
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])



        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training



        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)




        # Dataloaders
        distributed_batch_size = self.batch_size // max(world_size, 1) #def get_dataloader(self, dataset_sp, batch_size=16, rank=0, mode="train"): #得到数据加载器
        self.train_loader = self.get_dataloader(self.trainset, batch_size=distributed_batch_size, rank=RANK, mode="train")
        if RANK in (-1, 0):
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(self.testset, batch_size=distributed_batch_size if self.args.task_name == "obb" else distributed_batch_size * 2, rank=-1, mode="val")
            self.validator = self.get_validator() #验证用测试数据集
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels() #tudo



        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing 累计
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs #总迭代次数
        self.optimizer = self.creat_optimizer(model=self.model,name=self.args.optimizer,lr=self.args.lr0,momentum=self.args.momentum,
                                              decay=weight_decay,iterations=iterations)



        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")




        # Calculate the number of batches per epoch
        epoch_num_batch = len(self.train_loader)
        # Calculate the number of warmup iterations
        num_warmup = max(round(self.args.warmup_epochs * epoch_num_batch), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * epoch_num_batch
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])





        self.epoch_time = None
        LOGGER.info(
            f'训练验证图像尺寸是 {self.args.imgsz} \n'
            f'使用 {self.train_loader.num_workers * (world_size or 1)} 个工人加载数据\n'
            f"日志保留在 {colorstr('bold', self.save_dir)}\n"
        )
        return num_warmup,epoch_num_batch

    def start_epoch_training(self, epoch, epoch_num_batch):
        """Start training for a new epoch."""

        self.epoch = epoch

        # Run callbacks for the start of each training epoch
        self.run_callbacks("on_train_epoch_start")

        # Set the model to train mode
        self.model.train() 

        # Initialize progress bar
        pbar = enumerate(self.train_loader) #每一个epoch的数据

        # Set epoch for distributed training
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)

        # Log progress information for the first process
        if RANK in (-1, 0):
            LOGGER.info(self.progress_string())
            # Use tqdm for progress visualization
            pbar = TQDM(enumerate(self.train_loader), total=epoch_num_batch)  

        # Optionally update dataloader attributes
        if epoch == (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()
            self.train_loader.reset()

        # Initialize loss
        self.tloss = None

        # Reset gradients
        self.optimizer.zero_grad()
        return pbar
        
    def _normally_train(self, world_size=1): #训练开始
        """Train completed, evaluate and plot if specified by arguments."""

        num_warmup, epoch_num_batch = self._get_dataloaders_optimizer(world_size)



        self.run_callbacks("on_train_start")
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()


        last_opt_step = -1
        current_epoch = self.start_epoch
        while True:

            pbar = self.start_epoch_training(current_epoch, epoch_num_batch) #当先第几轮，一轮多少批

            for i, batch_labels_list in pbar:
                self.run_callbacks("on_train_batch_start")

                # Warmup
                cumulative_batch = i + epoch_num_batch * current_epoch #i 当前次数索引 ni总次数索引
                if cumulative_batch <= num_warmup:
                    xi = [0, num_warmup]  # x interp   到热身次数 accumulate逐渐接近4
                    self.accumulate = max(1, int(np.interp(cumulative_batch, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            cumulative_batch, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(current_epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(cumulative_batch, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch_labels_list = self.normalized_batch_images(batch_labels_list) #前向传播之前先归一化数据
                    self.loss, self.loss_items = self.model(batch_labels_list) #TUDO 计算损失值 tensor(2202.0635, device='cuda:0') torch.Size([3])
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize 
                if cumulative_batch - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = cumulative_batch

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (f"{current_epoch + 1}/{self.epochs}", mem, *losses, batch_labels_list["cls"].shape[0], batch_labels_list["img"].shape[-1])
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and cumulative_batch in self.plot_idx:
                        self.plot_training_samples(batch_labels_list, cumulative_batch)

                self.run_callbacks("on_train_batch_end")
            """ 批处理结束 """

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in (-1, 0):
                final_epoch = current_epoch + 1 == self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(current_epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (current_epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.current_epoch  # do not move
                    self.stop |= current_epoch >= self.epochs  # stop if exceeded epochs
                self.scheduler.step()
            self.run_callbacks("on_fit_epoch_end")
            torch.cuda.empty_cache()  # clear GPU memory at end of current_epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            current_epoch += 1


        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(
                f"\n{current_epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import pandas as pd  # scope for faster startup

        metrics = {**self.metrics, **{"fitness": self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()}
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": metrics,
            "train_results": results,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

        # Save last and best
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.weights_path / f"epoch{self.epoch}.pt")

    @staticmethod
    def get_dataset(data):

        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model_str, model = self.model_str, None
        ckpt = None
        if str(model_str).endswith(".pt"):
            model, ckpt = load_pytorch_model(model_str)
            model_str = ckpt["model"].yaml
        else:
            model_str = model_str
        self.model = self.get_model(model_str=model_str, model=model, verbose=RANK == -1)  # calls Model(model_str, model)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def normalized_batch_images(self, batch_labels_list):
        """Allows custom preprocessing model inputs and ground truths depending on task_name type."""
        return batch_labels_list

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, model_str=None, model=None, verbose=True):
        """Get model and raise NotImplementedError for loading model_str files."""
        raise NotImplementedError("This task_name trainer doesn't support loading model_str files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_sp, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data_dict["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, cumulative_batch):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = creat_args(ckpt_args)
                self.args.model = str(last)  # reinstate model
                for k in "imgsz", "batch":  # allow arg updates to reduce memory on resume if crashed due to CUDA OOM
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt["epoch"] + 1
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        if self.resume:
            assert start_epoch > 0, (
                f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
                f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
            )
            LOGGER.info(
                f"Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs"
            )
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def creat_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for name_module, module in model.named_modules():
            for name_param, param in module.named_parameters(recurse=False):
                fullname = f"{name_module}.{name_param}" if name_module else name_param
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer
