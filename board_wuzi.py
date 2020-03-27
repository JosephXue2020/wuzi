# coding = 'utf8'
# date: 2019.11.01


import numpy as np


'''
Construct the class of board. We always prefer list more than array for the data structure, e.g.self.legal_next_moves().
'''


class Board:
    def __init__(self, state=None, last_move=None, reset=False, **kwargs):
        self.size = kwargs.get('size', 11)
        self.channel = kwargs.get('channel')
        self.n_in_line = kwargs.get('n_in_line')
        if reset:
            self.reset()
        else:
            self.state = state
        self.last_move = last_move or self.get_last_move()
        self.legal_next_moves = self.get_legal_next_moves()
        self._players = ['black', 'white']
        self.winner = self.get_a_winner()

    def reset(self):
        self.state = np.zeros([self.size, self.size, self.channel])

    def get_player(self):
        # 返回当前走子方
        if self.state[0, 0, -1] == 1:
            return self._players[0]
        if self.state[0, 0, -1] == 0:
            return self._players[1]

    def get_last_move(self):
        # 通过比较state第三个维度的0、2channel，得到最后落子点。board类可缓存self.last_move，以节约计算
        state_layer_zero = self.state[:, :, 0]
        state_layer_two = self.state[:, :, 2]
        last_move = np.where(state_layer_zero != state_layer_two)
        last_move = np.array(last_move).T
        if last_move.shape[0] > 1:
            raise ValueError('Some error happened on board.')
        elif last_move.shape[0] == 0:
            return None
        else:
            last_move = list(last_move[0, :])
            return last_move

    def get_legal_next_moves(self):
        # 没有走子的地方都可以走子，返回2维列表
        position_taken = self.state[:, :, 0] + self.state[:, :, 1]
        position_spare = np.where(position_taken == 0)
        position_spare = np.array(position_spare).transpose()
        position_spare = list(position_spare)
        position_spare = [list(item) for item in position_spare]
        return position_spare

    def get_legal_next_states(self):
        legal_next_moves = self.get_legal_next_moves()
        # 返回新数组，不能影响原state，需深拷贝
        next_state = np.empty(self.size, self.size, self.channel)
        next_state[:, :, 1:-1] = self.state[:, :, :-2]
        # 处理最后一层，更新走子方
        if self.state[0, 0, -1] == 0:
            next_state[:, :, -1] = 1
        if self.state[0, 0, -1] == 1:
            next_state[:, :, -1] = 0
        # 处理第0层
        next_states = []
        for move in legal_next_moves:
            next_state_copy = next_state.copy()
            next_state_copy[move[0], move[1]] = 1
            next_states.append(next_state_copy)
        return next_states

    def move_check(self, move):
        if move in self.legal_next_moves:
            return True
        else:
            return False

    def get_next_state_by_move(self, move):
        # 先检查move是否合法
        if not self.move_check(move):
            raise ValueError('The move is illegal.')
        # 返回新数组，不能影响原state，则需深拷贝
        state_shape = self.state.shape
        next_state = np.empty(state_shape)
        next_state[:, :, 1:-1] = self.state[:, :, :-2]
        # 处理第0层
        state_layer_one = self.state[:, :, 1].copy()
        state_layer_one[move[0], move[1]] = 1
        next_state[:, :, 0] = state_layer_one
        # 处理最后一层，更新走子方
        next_state_copy = next_state.copy()
        if self.state[0, 0, -1] == 0:
            next_state_copy[:, :, -1] = 1
        if self.state[0, 0, -1] == 1:
            next_state_copy[:, :, -1] = 0
        return next_state_copy

    # 输赢已分，则返回胜方；未分，返回None
    def get_a_winner(self):
        player = self.get_player()
        move = self.last_move
        # 空集对应空棋盘
        if not move:
            return None
        # 走子位置从move中读出
        m, n = move[0], move[1]

        # 连续走子位置求和大于等于self.n_in_line，则获胜
        # 横向，m固定
        line_horizontal = self.state[m, :, 0]
        count = 1
        cursor = n
        while cursor - 1 >= 0:
            if line_horizontal[cursor-1] == 0:
                break
            else:
                count += 1
                cursor -= 1
        cursor = n
        while cursor + 1 < self.size:  # 注意没有等号
            if line_horizontal[cursor+1] == 0:
                break
            else:
                count += 1
                cursor += 1
        if count >= self.n_in_line:
            return player

        # 纵向，n固定
        line_vertical = self.state[:, n, 0]
        count = 1
        cursor = m
        while cursor-1 >= 0:
            if line_vertical[cursor-1] == 0:
                break
            else:
                count += 1
                cursor -= 1
        cursor = m
        while cursor+1 < self.size:
            if line_vertical[cursor+1] == 0:
                break
            else:
                count += 1
                cursor += 1
        if count >= self.n_in_line:
            return player

        # 西北到东南，mn均不固定
        count = 1
        cursor_vertical = m
        cursor_horizontal = n
        while cursor_vertical - 1 >= 0 and cursor_horizontal - 1 >= 0:
            if self.state[cursor_vertical-1, cursor_horizontal-1, 0] == 0:
                break
            else:
                count += 1
                cursor_vertical -= 1
                cursor_horizontal -= 1
        cursor_vertical = m
        cursor_horizontal = n
        while cursor_vertical + 1 < self.size and cursor_horizontal + 1 < self.size:
            if self.state[cursor_vertical+1, cursor_horizontal+1, 0] == 0:
                break
            else:
                count += 1
                cursor_vertical += 1
                cursor_horizontal += 1
        if count >= self.n_in_line:
            return player

        # 西南到东北，mn均不固定
        count = 1
        cursor_vertical = m
        cursor_horizontal = n
        while cursor_vertical + 1 < self.size and cursor_horizontal - 1 >= 0:
            if self.state[cursor_vertical+1, cursor_horizontal-1, 0] == 0:
                break
            else:
                count += 1
                cursor_vertical += 1
                cursor_horizontal -= 1
        cursor_vertical = m
        cursor_horizontal = n
        while cursor_vertical - 1 >= 0 and cursor_horizontal + 1 < self.size:
            if self.state[cursor_vertical-1, cursor_horizontal+1, 0] == 0:
                break
            else:
                count += 1
                cursor_vertical -= 1
                cursor_horizontal += 1
        if count >= self.n_in_line:
            return player

        # 平局返回tie
        if not self.legal_next_moves:
            return 'tie'

        # 以上情况都不是，则未分出胜负，返回None
        return None

    # 简单图示落子位置
    def graphic(self):
        # 将当前state合成一个组数，1代表黑子，-1代表白子，可直接读出盘面
        factor = 2 * self.state[0, 0, -1] - 1
        board_graph = (self.state[:, :, 0] - self.state[:, :, 1]) * factor

        print('player_1 为黑子，以 X 表示')
        print('player_2 为白子，以 O 表示')
        print('{0:2}'.format(' '), end='')
        for n in range(self.size):
            print("{0:4}".format(n), end='')
        print('\r\n')
        for m in range(self.size):
            print("{0:<4d}".format(m), end='')
            for k in range(self.size):
                value = board_graph[m, k]
                if [m, k] == self.get_last_move():  # 以颜色对最后一步走子加以区分
                    if value == 1:
                        print('X*'.center(4), end='')  # 加了格式，center(4)不起作用了
                    elif value == -1:
                        print('O*'.center(4), end='')
                    else:
                        print('-'.center(4), end='')
                else:
                    if value == 1:
                        print('X'.center(4), end='')
                    elif value == -1:
                        print('O'.center(4), end='')
                    else:
                        print('-'.center(4), end='')
            print('\r\n')

    # 最后走子用颜色展示
    def graphic_color(self):
        # 将当前state合成一个组数，1代表黑子，-1代表白子，可直接读出盘面
        factor = 2 * self.state[0, 0, -1] - 1
        board_graph = (self.state[:, :, 0] - self.state[:, :, 1]) * factor

        print('player_1 为黑子，以 X 表示')
        print('player_2 为白子，以 O 表示')
        print('{0:2}'.format(' '), end='')
        for n in range(self.size):
            print("{0:4}".format(n), end='')
        print('\r\n')
        for m in range(self.size):
            print("{0:<4d}".format(m), end='')
            for k in range(self.size):
                value = board_graph[m, k]
                if [m, k] == self.get_last_move():  # 以颜色对最后一步走子加以区分
                    if value == 1:
                        print('\033[1;35;0m X  \033[0m', end='')  # 加了格式，center(4)不起作用了
                    elif value == -1:
                        print('\033[1;35;0m O  \033[0m', end='')
                    else:
                        print('-'.center(4), end='')
                else:
                    if value == 1:
                        print('X'.center(4), end='')
                    elif value == -1:
                        print('O'.center(4), end='')
                    else:
                        print('-'.center(4), end='')
            print('\r\n')
