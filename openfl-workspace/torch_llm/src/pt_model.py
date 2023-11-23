# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import argparse
from typing import Any, Mapping

import horovod.torch as hvd
import numpy as np
import torch
import torch as pt
import torch.nn as nn
import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from tqdm import tqdm
from openfl.utilities.split import split_tensor_dict_for_holdouts
from openfl.federated import PyTorchTaskRunner
from openfl.federated.task.runner_pt import change_tags
from openfl.utilities import Metric, TensorKey
from transformers import AutoModelForSequenceClassification, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from datasets import Dataset, load_dataset, load_metric
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.ptglue_inmemory import GlueMrpcFederatedDataLoader
import subprocess
import json
from logging import getLogger
import time

logger = getLogger(__name__)

class timer:
    def __init__(self) -> None:
        self.clock_time = time.perf_counter()

    def tic(self):
        self.clock_time = time.perf_counter()

    def toc(self):
        return time.perf_counter() - self.clock_time

    def ttoc(self):
        val = self.toc()
        self.tic()
        return val

    def ptoc(self, message=None):
        print(message, self.toc())

    def pttoc(self, message=None):
        print(message, self.ttoc())

class LLMTaskRunner(PyTorchTaskRunner):
    def __init__(
        self,
        data_loader,
        base_model_name="roberta-base",
        device=None,
        metric=None,
        args=None,
        **kwargs,
    ):
        kwargs["data_loader"] = data_loader
        super().__init__(device, **kwargs)
        self.base_model_name = base_model_name
        self.kwargs = kwargs
        self.metric = load_metric("glue", "mrpc")
        self.timer = timer()
        self.timer.tic()
        self._init_model()
        logger.info(f'init model: {self.timer.ttoc()}')
        self._init_optimizer()
        logger.info(f'init optimizer: {self.timer.ttoc()}')
        
        self.save_models = []

    def _init_model(self):
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, return_dict=True
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="lora_only",
        )
        self.model = get_peft_model(model, peft_config)
        self.model.to(self.device)
        if self.kwargs.get("use_horovod"):
            #hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_parameters(self.state_dict(), root_rank=0)

    def _init_optimizer(self):
        ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=0.001)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.data_loader.train_set) * 5,
        )
        if self.kwargs.get("use_horovod"):
            self.optimizer = hvd.DistributedOptimizer(self.optimizer)
        self.initialize_tensorkeys_for_functions()

    def train(self):
        return self.model.train()

    def state_dict(self):
        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return set_peft_model_state_dict(self.model, state_dict)

    def validate(
        self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs
    ):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        if kwargs.get("use_horovod"):
            htim = timer()
            print(f"{hvd.rank()}:{hvd.size()} loading data")
            checkpoint = torch.load(kwargs["state_path"])
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            kwargs.update(checkpoint["kwargs"])
            print(f'{hvd.rank()}:{hvd.size()} hval load: {htim.ttoc()}')
            print(f"{hvd.rank()}:{hvd.size()} setup model")
            self.model.eval()
            self.model.to(self.device)
            val_score = 0
            total_samples = 0

            print(f"start dataloader")
            loader = self.data_loader.get_valid_loader()
            if use_tqdm:
                loader = tqdm(loader, desc="validate")
            print(f'{hvd.rank()}:{hvd.size()} hval dataloader: {htim.ttoc()}')
            print(f"{hvd.rank()}:{hvd.size()} validating")
            samples_run = 0
            with pt.no_grad():
                for sample in loader:
                    samples = sample["input_ids"].shape[0]
                    total_samples += samples
                    output = self.model(**sample)
                    # get the index of the max log-probability
                    logits = output.logits
                    predictions = torch.argmax(logits, dim=-1)
                    self.metric.add_batch(
                        predictions=predictions, references=sample["labels"]
                    )
                    samples_run += len(sample)
                    
            print(f'{hvd.rank()}:{hvd.size()} hval loop: {htim.ttoc()}')
            print(f"{hvd.rank()}:{hvd.size()} get accuracy")
            val_score = np.asarray(self.metric.compute()["accuracy"])
            result = hvd.allreduce(torch.from_numpy(val_score))
            print(f"{hvd.rank()}:{hvd.size()} for local rank", hvd.rank(), val_score)
            
            print(f'{hvd.rank()}:{hvd.size()} samples run++++++++++++++++++++++++++++: {samples_run}')
            if hvd.rank() == 0:
                print(f"{hvd.rank()}:{hvd.size()} mean of the big array is %f" % result)
                torch.save({"output": result}, kwargs["out_path"])
                print(f'{hvd.rank()}:{hvd.size()} hval save: {htim.ttoc()}')
        else:
            self.timer.tic()
            self.save_models.append(input_tensor_dict.copy())
            self.rebuild_model(round_num, input_tensor_dict, validation=True)
            state_path = f"col:{col_name}_rnd:{round_num}_state_validate.pt"
            out_path = f"col:{col_name}_rnd:{round_num}_out_validate.pt"
            data_path = self.data_loader.data_path
            logger.info(f'validation misc: {self.timer.ttoc()}')
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "kwargs": kwargs,
                },
                state_path,
            )
            horovod_kwags = {
                "col_name": col_name,
                "round_num": round_num,
                "input_tensor_dict": None,
                "use_tqdm": use_tqdm,
                "use_horovod": True,
            }
            logger.info(f'validation save: {self.timer.ttoc()}')
            result = subprocess.run(
                [
                    "horovodrun",
                    "-np",
                    "2",
                    "-H","localhost:2,"
                    "python",
                    "src/pt_model.py",
                    "--data_path",
                    str(data_path),
                    "--state_path",
                    state_path,
                    "--batch_size",
                    str(self.data_loader.batch_size),
                    "--kwargs",
                    json.dumps(horovod_kwags),
                    "--func",
                    "validate",
                    "--out_path",
                    out_path,
                ],
                capture_output=True,
            )
            logger.info(f'validation process: {self.timer.ttoc()}')

            val_score = torch.load(out_path)["output"]

            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            origin = col_name
            suffix = "validate"
            if kwargs["apply"] == "local":
                suffix += "_local"
            else:
                suffix += "_agg"
            tags = ("metric",)
            tags = change_tags(tags, add_field=suffix)
            # TODO figure out a better way to pass in metric for this pytorch
            #  validate function
            output_tensor_dict = {
                TensorKey("acc", origin, round_num, True, tags): np.array(val_score)
            }

            # Empty list represents metrics that should only be stored locally
            return output_tensor_dict, {}

    def train_batches(
        self, col_name, round_num, input_tensor_dict, use_tqdm=False, epochs=1, **kwargs
    ):
        """Train batches.

        Train the model on the requested number of batches.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)
            epochs:              The number of epochs to train

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        if kwargs.get("use_horovod"):
            checkpoint = torch.load(kwargs["state_path"])
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            kwargs.update(checkpoint["kwargs"])

            self.train()
            self.to(self.device)
            for epoch in range(epochs):
                self.logger.info(f"Run {epoch} epoch of {round_num} round")
                loader = self.data_loader.get_train_loader()
                if use_tqdm:
                    loader = tqdm.tqdm(loader, desc="train epoch")
                metric = self.train_epoch(loader)
            metric = hvd.allreduce(torch.from_numpy(metric))
            if hvd.rank() == 0:
                print("mean of the big array is %f" % metric)

                if self.model.config.problem_type == "regression":
                    loss_fct = MSELoss()
                elif self.model.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                elif self.model.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                torch.save(
                    {
                        "output": metric,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss_fct_name": loss_fct._get_name(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    },
                    kwargs["out_path"],
                )
        else:
            self.timer.tic()
            self.rebuild_model(round_num, input_tensor_dict)
            # set to "training" mode
            state_path = f"col:{col_name}_rnd:{round_num}_state_train_batches.pt"
            out_path = f"col:{col_name}_rnd:{round_num}_out_train_batches.pt"
            data_path = self.data_loader.data_path
            logger.info(f'train misc: {self.timer.ttoc()}')
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "kwargs": kwargs,
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                },
                state_path,
            )
            horovod_kwags = {
                "col_name": col_name,
                "round_num": round_num,
                "input_tensor_dict": None,
                "use_tqdm": use_tqdm,
                "use_horovod": True,
            }
            logger.info(f'train save: {self.timer.ttoc()}')
            result = subprocess.run(
                [
                    "horovodrun",
                    "-np",
                    "2",
                    "-H","localhost:2,",
                    "python",
                    "src/pt_model.py",
                    "--data_path",
                    str(data_path),
                    "--state_path",
                    state_path,
                    "--batch_size",
                    str(self.data_loader.batch_size),
                    "--kwargs",
                    json.dumps(horovod_kwags),
                    "--func",
                    "train_batches",
                    "--out_path",
                    out_path,
                ],
                capture_output=True,
            )
            logger.info(f'train process: {self.timer.ttoc()}')
            if result.returncode != 0:
                raise RuntimeError(result.stderr)

            checkpoint = torch.load(out_path)
            metric = checkpoint["output"]
            loss_fct_name = checkpoint["loss_fct_name"]
            metric = Metric(name=loss_fct_name, value=np.array(metric))
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            # Output metric tensors (scalar)
            origin = col_name
            tags = ("trained",)
            output_metric_dict = {
                TensorKey(
                    metric.name, origin, round_num, True, ("metric",)
                ): metric.value
            }

            # output model tensors (Doesn't include TensorKey)
            output_model_dict = self.get_tensor_dict(with_opt_vars=True)
            global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
                self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
            )

            # Create global tensorkeys
            global_tensorkey_model_dict = {
                TensorKey(tensor_name, origin, round_num, False, tags): nparray
                for tensor_name, nparray in global_model_dict.items()
            }
            # Create tensorkeys that should stay local
            local_tensorkey_model_dict = {
                TensorKey(tensor_name, origin, round_num, False, tags): nparray
                for tensor_name, nparray in local_model_dict.items()
            }
            # The train/validate aggregated function of the next round will look
            # for the updated model parameters.
            # This ensures they will be resolved locally
            next_local_tensorkey_model_dict = {
                TensorKey(
                    tensor_name, origin, round_num + 1, False, ("model",)
                ): nparray
                for tensor_name, nparray in local_model_dict.items()
            }

            global_tensor_dict = {**output_metric_dict, **global_tensorkey_model_dict}
            local_tensor_dict = {
                **local_tensorkey_model_dict,
                **next_local_tensorkey_model_dict,
            }

            # Update the required tensors if they need to be pulled from the
            # aggregator
            # TODO this logic can break if different collaborators have different
            # roles between rounds.
            # For example, if a collaborator only performs validation in the first
            # round but training in the second, it has no way of knowing the
            # optimizer state tensor names to request from the aggregator because
            # these are only created after training occurs. A work around could
            # involve doing a single epoch of training on random data to get the
            # optimizer names, and then throwing away the model.
            if self.opt_treatment == "CONTINUE_GLOBAL":
                self.initialize_tensorkeys_for_functions(with_opt_vars=True)

            # This will signal that the optimizer values are now present,
            # and can be loaded when the model is rebuilt
            self.training_round_completed = True

            # Return global_tensor_dict, local_tensor_dict
            return global_tensor_dict, local_tensor_dict

    def train_epoch(self, batch_generator) -> Metric:
        """Train single epoch.

        Override this function in order to use custom training.

        Args:
            batch_generator: Train dataset batch generator. Yields (samples, targets) tuples of
            size = `self.data_loader.batch_size`.
        Returns:
            Metric: An object containing name and np.ndarray value.
        """
        losses = []
        for sample in batch_generator:
            self.model.zero_grad()
            output = self.model(**sample)
            loss = output.loss
            loss.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        return np.array(loss)

    def save_native(
        self,
        filepath,
        model_state_dict_key="model_state_dict",
        optimizer_state_dict_key="optimizer_state_dict",
        **kwargs,
    ):
        """
        Save model and optimizer states in a picked file specified by the \
        filepath. model_/optimizer_state_dicts are stored in the keys provided. \
        Uses pt.save().

        Args:
            filepath (string)                 : Path to pickle file to be
                                                created by pt.save().
            model_state_dict_key (string)     : key for model state dict
                                                in pickled file.
            optimizer_state_dict_key (string) : key for optimizer state
                                                dict in picked file.
            kwargs                            : unused

        Returns:
            None
        """
        pickle_dict = {
            model_state_dict_key: get_peft_model_state_dict(self.model),
            optimizer_state_dict_key: self.optimizer.state_dict(),
        }
        pt.save(pickle_dict, filepath)


def get_args():
    """
    Get command-line arguments for a script.

    Parameters:
    - data_path (str): Path to the data.
    - model_path (str): Path to the model.

    Returns:
    - args (Namespace): A namespace containing the parsed arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument(
        "--data_path", type=str, help="Path to the data.", required=True
    )
    parser.add_argument("--out_path", type=str, help="Path to the data.", required=True)
    parser.add_argument(
        "--state_path", type=str, help="Path to the model.", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, help="Path to the model.", required=True
    )
    parser.add_argument("--kwargs", type=str, help="Path to the model.", required=True)
    parser.add_argument("--func", type=str, help="Path to the model.", required=True)

    args = parser.parse_args()
    return args


def main():
    tic = timer()
    args = get_args()
    print("getting GlueMrpcFederatedDataLoader")
    data_loader = GlueMrpcFederatedDataLoader(
        data_path=args.data_path, batch_size=args.batch_size, use_horovod=True
    )
    print(f'inside dataloader: {tic.ttoc()}')
    print("getting taskrunner")
    taskrunner = LLMTaskRunner(data_loader, use_horovod=True)
    print(f'inside llmtaskrunner: {tic.ttoc()}')
    func = getattr(taskrunner, args.func)
    kwargs = json.loads(args.kwargs)
    kwargs.update(
        {
            "data_path": args.data_path,
            "state_path": args.state_path,
            "out_path": args.out_path,
        }
    )
    print("running function")
    p = func(**kwargs)
    print(f'inside process: {tic.ttoc()}')
    return p


if __name__ == "__main__":
    print("in main")
    main()

#horovodrun -np 4 -H localhost:4 python src/pt_model.py --data_path 1 --state_path col:one_rnd:0_state_validate.pt --batch_size 32 --func validate --out_path col:one_rnd:0_out_validate.pt --kwargs '{"col_name": "one", "round_num": "0", "input_tensor_dict": "null", "use_tqdm": "false", "use_horovod": "true"}'

#horovodrun -np 2 -H localhost:1,10.72.4.30:1 python src/pt_model.py --data_path 1 --state_path col:one_rnd:0_state_validate.pt --batch_size 32 --func validate --out_path col:one_rnd:0_out_validate.pt --kwargs '{"col_name": "one", "round_num": "0", "input_tensor_dict": "null", "use_tqdm": "false", "use_horovod": "true"}'