# coding = 'utf8'
# date: 2019.11.13


import datetime
import random
import board_wuzi
import neural_network as nn
import mcts


class Game:
    def __init__(self, size, channel, n_in_line, search_time, search_num, temperature):
        self.size = size
        self.channel = channel
        self.n_in_line = n_in_line
        self.kwargs = {'size': self.size, 'channel': self.channel, 'n_in_line': self.n_in_line}
        self.time_delta = datetime.timedelta(seconds=search_time)
        self.search_num = search_num
        self.temperature = temperature
        self.reset()

    def reset(self):
        board = board_wuzi.Board(reset=True, **self.kwargs)
        self.nn = nn.CNN(board_size=self.size, channel=self.channel)
        self.mctree = mcts.MCTree(board, self.nn, self.temperature)

    def run(self):
        print('The game is starting.')
        print('Input a number to identify the game type, 0 for ai vs. human, 1 for ai vs. itself, '
              '2 for current ai vs. compared ai: ')
        game_type = int(input())
        if game_type == 0:
            self.ai_vs_human()
        if game_type == 1:
            self.ai_vs_self()
        if game_type == 2:
            self.ai_vs_ai()

    def human_play(self, move):
        node = self.mctree.tree
        board = node.get_board()
        children = node.get_children()
        children_moves = []
        for child in children:
            sub_board = child.get_board()
            sub_last_move = sub_board.last_move
            children_moves.append(sub_last_move)
        if move in children_moves:
            idx = children_moves.index(move)
            sub_node = children[idx]
            self.mctree.tree = sub_node
            winner = sub_node.board.get_a_winner()
            return winner
        else:
            next_state = board.get_next_state_by_move(move)
            sub_board = board_wuzi.Board(state=next_state, last_move=move, **self.kwargs)
            p, v = self.nn.predict(next_state)
            sub_node = mcts.Node(sub_board, p, v)
            self.mctree.tree = sub_node
            winner = sub_board.get_a_winner()
            return winner

    def ai_play(self):
        begin = datetime.datetime.now()
        count = 0
        while datetime.datetime.now()-begin < self.time_delta and count < self.search_num:
            self.mctree.simulation()
        win_prob = self.mctree.tree.Q
        print('The winning probability for the human player is:', win_prob)
        _, winner = self.mctree.get_play()
        return winner

    def ai_vs_human(self):
        self.nn.predict_session_open()
        human_player = input('Choose black or white player as you wish:')
        self.mctree.tree.board.graphic()
        if human_player == 'black':
            while True:
                while True:
                    position = input('Please input a place to move:')
                    position = position.split(',')
                    move = [int(item) for item in position]
                    board = self.mctree.tree.board
                    move_check = board.move_check(move)
                    if move_check:
                        break
                    else:
                        print('The place of your move is already taken. Try another again.')
                winner = self.human_play(move)
                board = self.mctree.tree.board
                board.graphic()
                if winner:
                    print('The game is over. The winner is {0}'.format(winner))
                    break
                winner = self.ai_play()
                board = self.mctree.tree.board
                board.graphic()
                if winner:
                    print('The game is over. The winner is {0}'.format(winner))
                    break

        if human_player == 'white':
            while True:
                winner = self.ai_play()
                board = self.mctree.tree.board
                board.graphic()
                if winner:
                    print('The game is over. The winner is {0}'.format(winner))
                    break
                while True:
                    position = input('Please input a place to move:')
                    position = position.split(',')
                    move = [int(item) for item in position]
                    board = self.mctree.tree.board
                    move_check = board.move_check(move)
                    if move_check:
                        break
                    else:
                        print('The place of your move is already taken. Try another again.')
                winner = self.human_play(move)
                board = self.mctree.tree.board
                board.graphic()
                if winner:
                    print('The game is over. The winner is {0}'.format(winner))
                    break

        self.nn.predict_session_close()

    def ai_vs_self(self):
        self.nn.predict_session_open()
        print('The ai is playing against itself.')
        self.mctree.tree.board.graphic()
        while True:
            winner = self.ai_play()
            board = self.mctree.tree.board
            board.graphic()
            if winner:
                print('The game is over. The winner is {0}'.format(winner))
                break
        self.nn.predict_session_open()

    def ai_vs_ai(self):
        print('The ai is playing against a compared ai.')
        print('Which ai makes the first move is randomly chosen.')
        dice = random.randint(0,2)
        flag = bool(dice)
        if not flag:
            print('The current ai will play first.')
        else:
            print('The compared ai will play first.')
        self.mctree.tree.board.graphic()
        while True:
            self.nn.predict_session_open(flag=flag)
            winner = self.ai_play()
            board = self.mctree.tree.board
            board.graphic()
            if winner:
                print('The game is over. The winner is {0}'.format(winner))
                break
            self.nn.predict_session_close()
            # 轮到另一方走子
            self.nn.predict_session_open(flag=not flag)
            winner = self.ai_play()
            board = self.mctree.tree.board
            board.graphic()
            if winner:
                print('The game is over. The winner is {0}'.format(winner))
                break
            self.nn.predict_session_close()


if __name__ == '__main__':
    size = 11
    n_in_line = 5
    channel = 5
    search_time = 10
    search_num = 600
    temperature = 0.2
    game = Game(size, channel, n_in_line, search_time, search_num, temperature)
    game.run()
