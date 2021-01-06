from open_spiel.python import rl_environment
import numpy as np
import pyspiel

class PigGame(object):
    def __init__(self, num_players=3, win_val=100, rng=np.random.default_rng()):
        assert(num_players > 0)

        self.win_val = win_val

        self.num_players = num_players
        self.player_totals = [0.] * self.num_players

        self.rng = rng

        self.curr_player = 0
        self.curr_total = 0

    def current_player(self):
        return self.curr_player

    def print_current_game_state(self):
        print('player_totals:', self.player_totals)
        print('curr_player:', self.curr_player)
        print('curr_total:', self.curr_total)

    def currrent_player_state(self):
        pass

    def step(self, decision):
        # If current player decides to roll...
        if decision == 1:
            sample = self.rng.integers(1, 6, endpoint=True)
            if sample != 1:
                self.curr_total += sample
                # Returns True if we stay on the same player's turn. None used
                # for last value since game is not over yet, and there is no
                # winner.
                return True, None
            else:
                # Reset current total and increment to next player turn.
                self.curr_total = 0
                self.curr_player = (self.curr_player + 1) % self.num_players
                # Returns False since since we go to the next players turn.
                # None used for last value since game is not over yet, and
                # there is no winner.
                return False, None
        # If current player decides to hold...
        else:
            self.player_totals[self.curr_player] += self.curr_total

            winning_player = None
            if self.player_totals[self.curr_player] >= self.win_val:
                winning_player = self.curr_player

            return False, winning_player


if __name__ == '__main__':
    game = PigGame()
    game.print_current_game_state()
    game.step(1)
    game.print_current_game_state()
