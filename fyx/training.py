from collections import defaultdict
import contextlib

import numpy as np
import torch
import torch.cuda.amp
from tabulate import tabulate
from tqdm import tqdm

from fyx.utils import from_numpy

def looped(generator):
    while True:
        yield from generator

def fit_model(
        model,
        train_dataloader,
        validation_dataloader,
        optimizer,
        loss_fn,
        num_epochs,
        logger,
        callbacks=[],
        steps_per_epoch=None,
        accumulate_n_batches=1,
        mixed_precision=False,
        profile=False,
        profile_path=None
    ):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_dataloader)

    train_iterator = looped(train_dataloader)

    scaler = torch.cuda.amp.GradScaler()

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=10, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) if profile else contextlib.nullcontext() as prof:
        for epoch in tqdm(range(num_epochs)):
            num_batches = steps_per_epoch
            logs = defaultdict(lambda: 0)
            model.train()
            torch.set_grad_enabled(True)
            for callback in callbacks: callback.on_epoch_begin(logs)

            accumulator = defaultdict(lambda: [])

            progress_bar = tqdm(total=num_batches)
            for i in range(num_batches):
                with torch.cuda.amp.autocast() if mixed_precision else contextlib.nullcontext():
                    batch = from_numpy(next(train_iterator))
                    outputs = model(batch)
                    loss_dict = loss_fn(outputs, batch)

                for key, value in loss_dict.items():
                    accumulator[f'train_{key}'].append(value.item() if isinstance(value, torch.Tensor) else value)

                if mixed_precision:
                    scaler.scale(loss_dict['total_loss']).backward()
                else:
                    loss_dict['total_loss'].backward()

                if (i + 1) % accumulate_n_batches == 0:
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                if profile:
                    prof.step()

                for callback in callbacks: callback.on_train_batch_end(logs, outputs, batch)
                progress_bar.set_postfix({
                    **{key: f'{value[-1]:.2f}' for key, value in accumulator.items() if key.startswith('train')},
                    **{key: f'{value:.2f}' for key, value in logs.items() if key.startswith('train')}
                })
                progress_bar.update(1)
                del batch
                del outputs

            optimizer.zero_grad()
            progress_bar.close()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            num_batches = len(validation_dataloader)
            model.eval()
            torch.set_grad_enabled(False)
            progress_bar = tqdm(total=num_batches)
            validation_iterator = iter(validation_dataloader)
            for i in range(num_batches):
                with torch.cuda.amp.autocast() if mixed_precision else contextlib.nullcontext():
                    batch = from_numpy(next(validation_iterator))
                    outputs = model(batch)
                    loss_dict = loss_fn(outputs, batch)

                for key, value in loss_dict.items():
                    accumulator[f'val_{key}'].append(value.item() if isinstance(value, torch.Tensor) else value)
                for callback in callbacks: callback.on_validation_batch_end(logs, outputs, batch)
                progress_bar.set_postfix({
                    **{key: f'{value[-1]:.2f}' for key, value in accumulator.items() if key.startswith('val')},
                    **{key: f'{value:.2f}' for key, value in logs.items() if key.startswith('val')}
                })
                progress_bar.update(1)
                del batch
                del outputs
            progress_bar.close()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            for key in accumulator.keys(): logs[key] = np.mean(accumulator[key])

            for callback in callbacks: callback.on_epoch_end(logs)

            epoch_rows = [['epoch', epoch]]
            for name, value in logs.items():
                epoch_rows.append([name, f'{value:.3f}'])

            logger(tabulate(epoch_rows))
