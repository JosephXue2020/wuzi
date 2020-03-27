# coding = 'utf8'
# date: 2019.11.01


'''
还有可以改进的地方，即predict value时候数据可增强。
'''


import datetime
import math
import numpy as np
import board_wuzi
import neural_network as nn


# 以类的调用实现树形结构
class Node:
    def __init__(self, board, p=None, v=None):
        self.board = board
        self.V = v  # 当前节点state的value function
        self.P = p  # P是当前state和所有action的先验概率，二维数组
        self.Q = 0  # Q是父节点state在所选action作用下的action value function。注意于P区别
        self.N = 0
        self.W = 0
        self._children = []

    def get_children(self):
        return self._children

    def insert_child(self, node):
        self._children.append(node)

    def get_board(self):
        return self.board


class MCTree:
    def __init__(self, board, neural_network, temperature=1.):
        self.nn = neural_network
        state = board.state
        self.nn.predict_session_open()
        p, v = self.nn.predict(state)
        self.nn.predict_session_close()
        self.tree = Node(board, p, v)
        self.size = board.size
        self.channel = board.channel
        self.n_in_line = board.n_in_line
        self.kwargs = {'size': self.size, 'channel': self.channel, 'n_in_line': self.n_in_line}
        self.routine = []  # 定义一个list，元素是节点node，记录selection和simulation经过的路径，用以back_propagation
        self.c = 1.  # puct中exploration项的系数
        self.temperature_base = temperature
        self.step = 0

    def reset(self):
        board = board_wuzi.Board(reset=True, **self.kwargs)
        state = board.state
        self.nn.predict_session_open()
        p, v = self.nn.predict(state)
        self.nn.predict_session_close()
        self.tree = Node(board, p, v)
        self.step = 0

    def simulation(self):
        node = self.tree  # 每次从根节点开始
        self.routine = []
        expanded_node = self.selection(node)
        self.back_propagation(expanded_node)

    # 调用selection，返回维护的树将要增加的叶节点
    def selection(self, node):
        self.routine.append(node)
        board = node.get_board()
        legal_next_moves = board.legal_next_moves

        # 先判断是不是终点，如果是，则返回该节点，不拓展
        winner = board.winner
        if winner:
            return node

        # 按pucb选择拓展节点
        in_tree_nodes = node.get_children()
        in_tree_last_moves = []
        for in_tree_node in in_tree_nodes:
            if in_tree_node.board.winner:
                expanded_node = self.selection(in_tree_node)
                return expanded_node
            in_tree_last_move = in_tree_node.board.last_move
            in_tree_last_moves.append(in_tree_last_move)
        out_tree_last_moves = [move for move in legal_next_moves if move not in in_tree_last_moves]
        p_parent = node.P
        selected_move, selected_node = self.pucb_max(p_parent, in_tree_last_moves, in_tree_nodes, out_tree_last_moves)
        if selected_node:
            expanded_node = self.selection(selected_node)
            return expanded_node
        else:
            selected_state = board.get_next_state_by_move(selected_move)
            selected_board = board_wuzi.Board(selected_state, selected_move, **self.kwargs)
            p_, v_ = self.nn.predict(selected_state)
            expanded_node = Node(selected_board, p=p_, v=v_)
            # 判断终态，更新value
            if selected_board.winner and selected_board.winner == selected_board.get_player():
                expanded_node.V = 1.
            elif selected_board.winner == 'tie':
                expanded_node.V = 0.
            node.insert_child(expanded_node)
            self.routine.append(expanded_node)  # 注意：这个非常重要，新插入树的节点也要更新，否则影响收敛速度
            return expanded_node

    def pucb_max(self, p, in_tree_moves, in_tree_nodes, out_tree_moves):  # P是父节点属性
        moves = in_tree_moves + out_tree_moves
        length_out_tree = len(out_tree_moves)
        length_total = len(moves)
        out_tree_nodes = [None] * length_out_tree
        nodes = in_tree_nodes + out_tree_nodes

        # 对policy引入noise，加强exploration
        flag = bool(np.random.randint(0, 5))  # 先roll一发，决定是否引入noise
        if not flag:
            epsilon = 0.1
        else:
            epsilon = 0.
        p = p * (1 - epsilon)
        p_list = [p[move[0], move[1]] for move in moves]
        idx = list(range(length_total))
        idx_selected = np.random.choice(idx)
        p_list[idx_selected] += epsilon

        n_in_tree = []
        q_in_tree = []
        for node in in_tree_nodes:
            n = node.N
            n_in_tree.append(n)
            q = node.Q
            q_in_tree.append(q)
        n = n_in_tree + [0.] * length_out_tree
        q = q_in_tree + [0.] * length_out_tree

        n_sum = sum(n) + 0.001  # 小量剔除了全为0时，完全的随机选择（实际是按顺序从头选）
        pucbs = []
        for i in range(length_total):
            pucb = q[i] + self.c * p_list[i] * math.sqrt(n_sum) / (1 + n[i])
            pucbs.append(pucb)
        pucb_max = max(pucbs)
        idx_max = pucbs.index(pucb_max)
        return moves[idx_max], nodes[idx_max]

    def back_propagation(self, leaf_node):
        v = leaf_node.V
        leaf_player = leaf_node.board.get_player()
        for node in self.routine:
            node.N += 1.
            player = node.board.get_player()
            if player == leaf_player:
                node.W += v
                node.Q = node.W / node.N
            else:
                node.W -= v
                node.Q = node.W / node.N

    def get_temperature(self):
        if self.step < self.size**2/8:
            temperature = self.temperature_base
        else:
            temperature = self.temperature_base / 2.
        return temperature

    def get_play(self):
        state = self.tree.board.state
        prob = np.zeros([self.size, self.size])
        prob_flat = []
        children = self.tree.get_children()
        for node in children:
            value = math.pow(node.N, 1 / self.get_temperature())
            move = node.board.last_move
            prob[move[0], move[1]] = value
            prob_flat.append(value)
        prob = prob / np.sum(prob)  # 归一化
        z = 0.  # 初始化
        player = self.tree.board.get_player()
        data_dict = {'state': state, 'prob': prob, 'z': z, 'player': player}
        prob_flat_norm = [item / sum(prob_flat) for item in prob_flat]
        idx = np.arange(len(prob_flat_norm))
        idx_selected = np.random.choice(idx, p=prob_flat_norm)
        node_selected = children[idx_selected]
        self.tree = node_selected
        self.step += 1
        winner = self.tree.board.get_a_winner()
        return data_dict, winner


# 以下是对mcts速度的测试
if __name__ == '__main__':
    pass
