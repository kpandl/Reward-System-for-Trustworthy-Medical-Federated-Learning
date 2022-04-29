import torch

def BatchIterator(model, phase,
        Data_loader,
        criterion,
        optimizer,
        device,
        differentialprivacy=0):

    print_freq = 1000
    running_loss = 0.0

    for i, data in enumerate(Data_loader):

        imgs, labels = data

        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
            outputs = model(imgs)
        else:

            model.eval()
            with torch.no_grad():
                outputs = model(imgs)

        loss = criterion(outputs, labels)

        if phase == 'train':
            loss.backward()
            optimizer.step()  # update weights

        running_loss += loss * batch_size
        if (i % print_freq == 0):
            print(str(i * batch_size))

    return running_loss