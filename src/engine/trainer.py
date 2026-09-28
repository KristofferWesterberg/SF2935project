
def train_one_step(optimizer,model,data):

    optimizer.zero_grad()
    loss = model.loss_function(data) # Forward pass in loss function
    loss.backwards()
    optimizer.step()
    return loss


def train(train_loader, epcs, model, optimizer, batch_size):
    samples_evaluated = 0
    for epc in range(epcs):
        loss_list = []

        print(f'EPOCH: {epc+1}')

        for i, data in enumerate(train_loader):

            loss = train_one_step(optimizer, model, data)
            samples_evaluated += batch_size
            loss_list.append((samples_evaluated,loss))

            print(f' batch: {i+1}, loss: {loss}')

    return loss_list


