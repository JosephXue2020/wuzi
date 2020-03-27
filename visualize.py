# coding = 'utf8
# date: 2019.11.18


import time
import csv
import matplotlib.pyplot as plt
import neural_network as nn


def read_from_file(directory):
    losses = []
    with open(directory, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            losses.append(row)
    return losses


directory = nn.model_loader.losses_path()
while True:
    losses = read_from_file(directory)
    losses = losses[:]
    loss_last_episode = [item[0] for item in losses]
    loss_sample = [item[1] for item in losses]
    plt.plot(loss_last_episode, marker='x', color='r')
    plt.plot(loss_sample, marker='o', color='b')
    plt.show()
    time.sleep(500)
