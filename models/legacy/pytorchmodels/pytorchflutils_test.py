import torch
import numpy as np
import time



def pt_train_epoch_test_performance(torch_nn, train_loader, device, \
    optimizer, loss_fn, num_batches):
    # set to "training" mode
    torch_nn.train()
    
    losses = []
    train_times = []
    data_load_times = []
    last_batch_end = None

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == num_batches:
            break
        batch_train_start = time.time()
        if batch_idx != 0:
            data_load_times.append(batch_train_start - last_batch_end)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = torch_nn(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        batch_train_stop = time.time()
        if batch_idx != 0:
           train_times.append(batch_train_stop -last_batch_end) 
        last_batch_end = batch_train_stop



    return np.mean(losses), data_load_times, train_times

