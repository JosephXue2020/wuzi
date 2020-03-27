# coding = 'utf8'
# date: 2019.11.01


from collections import deque
import json
import csv
import os
import numpy as np

import board_wuzi
import neural_network as nn
import mcts


class TrainModel:
    def __init__(self, size, channel, n_in_line, buffer_size, search_num, episode_num, batch_size,
                 training_time_per_episode, temperature):
        self.size = size
        self.channel = channel
        self.n_in_line = n_in_line
        self.temperature = temperature
        self.kwargs = {'size': self.size, 'channel': self.channel, 'n_in_line': self.n_in_line}
        board = board_wuzi.Board(reset=True, **self.kwargs)  # 初始化空棋盘
        self.nn = nn.CNN(board_size=self.size, channel=self.channel)
        self.mctree = mcts.MCTree(board, self.nn, self.temperature)
        self.buffer_size = buffer_size
        self.search_num = search_num
        self.episode_num = episode_num
        self.batch_size = batch_size
        self.training_time_per_episode = training_time_per_episode
        self.model_loader = nn.ModelLoader()
        self.training_data = deque(maxlen=self.buffer_size)
        if os.path.exists(self.model_loader.training_data_path()):
            self.training_data.extend(self.load_training_data())
        self.training_data_cache = []  # 保存self-play数据

    def run(self):
        for episode in range(self.episode_num):
            episode_data = []
            self.nn.predict_session_open()
            while True:  # 自我下棋，直到分出胜负
                for i in range(self.search_num):
                    self.mctree.simulation()
                print('The number of sub-children is: ', len(self.mctree.tree.get_children()))
                data_dict, winner = self.mctree.get_play()
                self.mctree.tree.board.graphic()
                print('The self-play makes one step.')
                episode_data.append(data_dict)
                if winner:
                    print('Self-play complete for times: {0}'.format(episode))
                    print('The winner is: ', winner)

                    # 加入末态到episode_data（似乎无必要）
                    # board = self.mctree.tree.get_board()
                    # state = board.state
                    # z = 0
                    # player = board.get_player()
                    # prob = np.zeros([self.size, self.size])
                    # last_move = board.last_move
                    # prob[last_move[0], last_move[1]] = 1.
                    # terminal_data = {'state': state, 'prob': prob, 'z': z, 'player': player}
                    # episode_data.append(terminal_data)

                    break
            self.nn.predict_session_close()
            for item in episode_data:
                if item['player'] == winner:
                    item['z'] = 1.
                elif item['player'] != winner and winner:
                    item['z'] = -1.
                self.training_data.append(item)
                self.training_data_cache.append(item)
            if episode == episode_num -1:
                self.validation(episode_data)
            data_loader = nn.DataLoader(self.training_data, self.batch_size)
            self.nn.train(data_loader, self.training_time_per_episode)
            self.mctree.reset()

    def save_training_data(self):
        data_path = self.model_loader.training_data_path()
        if os.path.exists(data_path):
            data_saved = self.load_training_data()
        else:
            data_saved = []
        data_saved.extend(self.training_data_cache)
        data_all = data_saved
        with open(data_path, 'w') as f:
            json.dump(data_all, f, cls=NumpyEncoder)
        self.training_data_cache = []

    def load_training_data(self):
        data_path = self.model_loader.training_data_path()
        with open(data_path, 'r') as f:
            data = json.load(f)
            return data

    def save_losses(self, data):
        losses_path = self.model_loader.losses_path()
        with open(losses_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def validation(self, episode_data):
        # 返回两个值，最近一个episode的平均loss和全部数据采样的平均loss
        data_loader_episode = nn.DataLoader(episode_data)
        loss_last_episode = self.nn.validation(data_loader_episode)
        if os.path.exists(self.model_loader.training_data_path()):
            data_all = self.load_training_data()
            data_loader_sample = nn.DataLoader(data_all, self.batch_size)
            loss_sample = self.nn.validation(data_loader_sample)
        else:
            loss_sample = loss_last_episode
        self.save_losses([loss_last_episode, loss_sample])
        return loss_last_episode, loss_sample


# json不能存储ndarray类型数据，下面的类将数据转换为list以便存储
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    size = 11
    n_in_line = 5
    channel = 5
    time = 10
    temperature = 1.
    buffer_size = size ** 2 * 150
    search_num = 600
    episode_num = 10
    batch_size = 30
    training_time_per_episode = 400
    while True:
        train_model = TrainModel(size, channel, n_in_line, buffer_size, search_num, episode_num, batch_size,
                     training_time_per_episode, temperature)
        train_model.run()
        train_model.save_training_data()
