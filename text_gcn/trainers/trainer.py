class Trainer(object):
    """
    Generic Trainer class which trains and tests networks
    """
    def __init__(self, model, loss_func, data_loader):
        self.loss_func = loss_func
        self.data_loader = data_loader
        self.model = model

    def train(self, epochs, optimizer):
        self.model.train()
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for iteration, (graph_batch, label) in enumerate(self.data_loader):
                prediction = self.model(graph_batch)
                label = label.type_as(prediction)
                loss = self.loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (iteration + 1)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

    def test(self):
        pass
