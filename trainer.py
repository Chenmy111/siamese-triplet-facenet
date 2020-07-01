import torch
import numpy as np
from tqdm import tqdm


def fit(train_loader, val_loader, model, logger, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, epoch, n_epochs, loss_fn, optimizer, cuda, log_interval, metrics)
        scheduler.step()
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        logger.log_value('Train set: Average loss', train_loss)
        for metric in metrics:
            logger.log_value(metric.name(), metric.value())
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Test set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        logger.log_value('Test set: Average loss', val_loss)
        for metric in metrics:
            logger.log_value(metric.name(), metric.value())
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, epoch, n_epochs, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    pbar =tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in pbar:
        # len(target) == 0 是TripletMNIST的情况,返回一个空list
        target = target if len(target) > 0 else None
        # SiameseMNIST,TripletMNIST返回list,BalancedBatchSampler返回torch.Tensor
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        # 处理OnlineTripletLoss返回losses.mean(), len(triplets)的情况
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:


            # message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     batch_idx * len(data[0]), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), np.mean(losses))
            # for metric in metrics:
            #     message += '\t{}: {}'.format(metric.name(), metric.value())
            #
            # print(message)

            print_message = 'Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, n_epochs, batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses)
            )
            for metric in metrics:
                print_message += '\t{}: {}'.format(metric.name(), metric.value())

            pbar.set_description(print_message)

            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0

        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
