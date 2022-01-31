import matplotlib.pyplot as plt
from torch import nn,optim

from PIL import Image
import tqdm, torch
import numpy as np

def train_model(model, dl_train, dl_test, cuda=False, lr=0.001, epochs=10, show_plots=False):
    if show_plots:
        plt.ion()
        plt.show(block=False)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if cuda:
        model = model.cuda()


    losses = []
    accuracies = []

    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        for x,y in iter(dl_train):
            if cuda:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()

            tag_scores =  model(x)
            loss = loss_function(tag_scores.transpose(1, 2),y)

            loss.backward()
            optimizer.step()

        acc = []
        los = []
        ## evaluation
        with torch.no_grad():
            for x,y in iter(dl_test):
                if cuda:
                    x, y = x.cuda(), y.cuda()

                tag_scores =  model(x)

                loss = loss_function(tag_scores.transpose(1, 2),y)
                los.append(loss.cpu().item())

                acc.append(
                    (tag_scores.argmax(dim=2)==y)[y!=-100].float()
                )

        acc = torch.cat(acc).mean().item()
        los = np.array(los).mean()

        losses.append(los)
        accuracies.append(acc)
        #show epoch results
        #pbar.set_description(f"Loss:{los}\tAccurancy:{acc}")
        if show_plots:
            plt.clf()
            plt.subplot(121)
            plt.title("Test loss")
            plt.plot(losses)

            plt.subplot(122)
            plt.title("Test accuracy")
            plt.plot(accuracies)

            plt.draw()
            plt.pause(0.001)

    if show_plots:
        plt.ioff()
        fig = plt.gcf()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.clf()
        return losses, accuracies, img

    return losses, accuracies
