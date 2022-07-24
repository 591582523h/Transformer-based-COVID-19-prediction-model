import torch


def train_epochs(model, optimizer, criterion, data, label, device, lr=0.005, epochs=400):
    model.train()
    data = data.to(device)
    for epoch in range(epochs):
        for i in range(len(data)):
            optimizer.zero_grad()
            output = model(data[:i].unsqueeze(1), device)
            output = output.to(device)
            
            loss = criterion(label[:i].squeeze(), output.squeeze())
            loss.backward()
            optimizer.step()

        print(f'epoch: {epoch:3} loss: {loss.item():10.12f} learning rate: {lr}')
        
        # if epoch == 10 or epoch == 40 or epoch == 100:
        #    lr *= 0.1
        if epoch % 20 == 0:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, f'save_model_14/net_{epoch + 1}_{loss.item():10.12f}.pt')

    return model
