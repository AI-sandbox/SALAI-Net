import torch
import torch.nn as nn
from .utils import ancestry_accuracy, ProgressSaver, AverageMeter, ReshapedCrossEntropyLoss,\
    adjust_learning_rate

import time

def train(model, train_loader, valid_loader, args):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = ReshapedCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    init_time = time.time()

    progress_saver = ProgressSaver(args.exp)
    train_loss_meter = AverageMeter()
    best_val_acc = -1
    best_epoch = -1

    lr = args.lr

    print("start")

    init_epoch = 0
    if args.resume:
        progress_saver.load_progress()
        init_epoch, best_val_loss, start_time = progress_saver.get_resume_stats()

        init_time = time.time() - start_time
        model.load_state_dict(torch.load(args.exp + "/models/last_model.pth"))
        optimizer.load_state_dict(
            torch.load(args.exp + "/models/last_optim.pth"))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % init_epoch)

        init_epoch += 1

    model.to(device)

    for n in range(init_epoch, args.num_epochs):
        model.train()
        train_loss_meter.reset()

        if args.lr_decay > 0:
            lr = adjust_learning_rate(args.lr, args.lr_decay, optimizer, n)

        for i, batch in enumerate(train_loader):
            # break

            if args.model == "VanillaConvNet":
                out_base, out = model(batch["vcf"].to(device))
            elif args.model == "LAINet":
                out_base, out = model(batch["vcf"].to(device))

            loss = criterion(out, batch["labels"].to(device))

            if args.base_loss:
                loss_base = criterion(out_base,  batch["labels"].to(device))
                loss = loss + loss_base
            loss.backward()

            if((i+1) % args.update_every) == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss_meter.update(loss.item())

        val_acc, base_val_acc, val_loss, base_val_loss = validate(model, valid_loader, criterion, args)
        train_loss = train_loss_meter.get_average()

        total_time = time.time() - init_time

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = n
            torch.save(model.state_dict(), args.exp + "/models/best_model.pth")

        torch.save(model.state_dict(), args.exp + "/models/last_model.pth")
        torch.save(optimizer.state_dict(), args.exp + "/models/last_optim.pth")

        epoch_data = {
            "epoch": n,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "base_val_loss": base_val_loss,
            "val_acc": val_acc.cpu(),
            "base_val_acc": base_val_acc.cpu(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "time": total_time,
            "lr": lr
        }

        progress_saver.update_epoch_progess(epoch_data)


        print("epoch #", n, ":\tVal acc:", val_acc.item(), "\ttime:", time.time()- init_time)

def validate(model, val_loader, criterion, args):

    with torch.no_grad():

        val_loss = AverageMeter()
        base_val_loss = AverageMeter()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval().to(device)

        acc = torch.tensor(0).float()
        base_acc = torch.tensor(0).float()

        for i, batch in enumerate(val_loader):

            if args.model == "VanillaConvNet":
                out_base, out = model(batch["vcf"].to(device))
            elif args.model == "LAINet":
                out_base, out = model(batch["vcf"].to(device))

            batch["labels"] = batch["labels"].to(device)

            acc = acc + ancestry_accuracy(out, batch["labels"])
            loss = criterion(out, batch["labels"])
            val_loss.update(loss.item())

            if args.base_loss:
                base_acc = base_acc + ancestry_accuracy(out_base,batch["labels"])
                base_loss = criterion(out_base, batch["labels"])
                base_val_loss.update(base_loss.item())


        acc = acc / len(val_loader.dataset)

        if args.base_loss:
            base_acc = base_acc / len(val_loader.dataset)

        return acc, base_acc, val_loss.get_average(), base_val_loss.get_average()

