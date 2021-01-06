import numpy as np


class PigGame(object):
    def __init__(self, num_players=3, win_val=100, rng=np.random.default_rng()):
        assert(num_players > 0)

        self.win_val = win_val

        self.num_players = num_players
        self.player_totals = np.array([0.] * self.num_players)

        self.rng = rng

        self.curr_player = 0
        self.curr_total = 0

    def current_player(self):
        return self.curr_player

    def reset(self):
        self.player_totals = np.array([0.] * self.num_players)
        self.curr_player = 0
        self.curr_total = 0

    def print_current_game_state(self):
        print('player_totals:', self.player_totals)
        print('curr_player:', self.curr_player)
        print('curr_total:', self.curr_total)

    def current_player_state(self):
        # Vectorization has other player totals in order of game (in case some
        # strategy based on order is possible). These values are preceeded
        # by the current total from the die for the current player.
        reordered = np.roll(self.player_totals, -self.curr_player)

        return np.insert(reordered, 0, self.curr_total)

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

            # Check if player has won with this increment.
            winning_player = None
            if self.player_totals[self.curr_player] >= self.win_val:
                winning_player = self.curr_player

            # Reset total and increment to next player.
            self.curr_total = 0
            self.curr_player = (self.curr_player + 1) % self.num_players

            return False, winning_player


if __name__ == '__main__':
    game = PigGame()
    game.print_current_game_state()
    game.step(1)
    game.print_current_game_state()
    print('player_state:', game.current_player_state())
    game.step(0)
    game.print_current_game_state()
