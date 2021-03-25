
# Representation
import numpy as np
import random
import gym
import pygame
# from . import graphics

# TODO(jjma): Look into turning this into OpenAI gym.
# TODO(jjma): Make number of players/goal size tunable?
# TODO(jjma): Implement cooperation (shared goals, etc) or enable this in reward.
class MarkovSoccer(object):
    # Game constants. Do not change.
    BOARD_LENGTH = 6
    NUM_PLAYERS = 4

    # Board Labels
    INVALID_POS = -1
    EMPTY = 0
    BALL = 5
    GOAL_START_NUM = 10

    # Rewards settings.
    OWN_GOAL_LOSER = -1.
    OWN_GOAL_OTHER = 0.
    NORMAL_GOAL_WINNER = 1.
    NORMAL_GOAL_LOSER = -1.
    NORMAL_GOAL_OTHER = -0.2

    # Render colors.
    COLORS = ['red', 'green', 'blue', 'orange', 'black']

    def __init__(
        self
    ):
        # Randomly generate coordinates for the ball and each player.
        length = MarkovSoccer.BOARD_LENGTH
        self.possible_coords = [
            np.array([x+1, y+1]) for x in range(length) for y in range(length)
        ]
        # TODO(jjma): Currently, hard coded, need to generalize this.
        self.goal_positions = [
            [np.array((1, 3)), np.array((1, 4))],
            [np.array((3, 6)), np.array((4, 6))],
            [np.array((6, 3)), np.array((6, 4))],
            [np.array((3, 1)), np.array((4, 1))]
        ]

        # Enumerate moves as defined in action space.
        self.moves = [
            np.array([0, 0]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, -1])
        ]

        self._board_reset()
        self.render_window = None

    def reset(self):
        self._board_reset()
        return self._compute_observations()

    def step(self, actions):
        """
        Apply actions to game environment. In order, actions are integers
        where 0 = stand, 1 = up, 2 = right, 3 = down, 4 = left.
        """
        # Execute actions in a randomized order.
        order = list(range(MarkovSoccer.NUM_PLAYERS))
        random.shuffle(order)

        dones = [False for  _ in range(MarkovSoccer.NUM_PLAYERS)]
        rewards = np.array([0. for _ in range(MarkovSoccer.NUM_PLAYERS)])

        for i in order:
            # Default Case: Calculate new position based on action taken.
            old_pos = self.item_to_coord[i+1]
            new_pos = old_pos + self.moves[actions[i]]

            # If any moves will change state.
            if actions[i] > 0:
                # Case 1: Player tries to move into boundary: no pos update.
                if self.board[tuple(new_pos)] == MarkovSoccer.INVALID_POS:
                    new_pos = old_pos

                # Case 2: Picking up an ungrabbed ball (position should still update).
                elif self.board[tuple(new_pos)] == 5:
                    self.has_ball[i] = True

                # Case 3: Colliding with another player.
                elif (self.board[tuple(new_pos)] >= 1 and self.board[tuple(new_pos)] <= 4):
                    # Position does not change, as defined in Markov Soccer.
                    other_player_idx = self.board[tuple(new_pos)] - 1

                    # Only update state if other player has ball; steal it.
                    if self.has_ball[other_player_idx]:
                        # Ball changes hands.
                        self.has_ball[other_player_idx] = False
                        self.has_ball[i] = True

                        # Update item tracker dictionary for ball.
                        self.item_to_coord[MarkovSoccer.BALL] = old_pos

                    # No change in movement.
                    new_pos = old_pos

                # Case 4: Player has ball and moves into a goal.
                elif self.has_ball[i] and self.board[tuple(new_pos)] > MarkovSoccer.GOAL_START_NUM:
                    # Check if own goal, reward differently.
                    # TODO(jjma): Add cooperation ability here and tune reward.
                    if self.board[tuple(new_pos)] % 10 == i + 1:
                        rewards = np.array([MarkovSoccer.OWN_GOAL_OTHER for _ in range(MarkovSoccer.NUM_PLAYERS)])
                        rewards[i] = MarkovSoccer.OWN_GOAL_LOSER
                    else:
                        rewards = np.array([MarkovSoccer.NORMAL_GOAL_OTHER for _ in range(MarkovSoccer.NUM_PLAYERS)])
                        rewards[i] = MarkovSoccer.NORMAL_GOAL_WINNER
                        loser_idx = self.board[tuple(new_pos)] % 10 - 1
                        rewards[loser_idx] = MarkovSoccer.NORMAL_GOAL_LOSER

                    # Mark game as done.
                    dones = [True for  _ in range(MarkovSoccer.NUM_PLAYERS)]
                    break

                # Case 5: Player tries to move into a goal without ball.
                elif self.board[tuple(new_pos)] > MarkovSoccer.GOAL_START_NUM:
                    new_pos = old_pos

                # Update position on board and in item dictionary if pos updated.
                if not np.array_equal(new_pos, old_pos):
                    # Wipe old position and set new game board position.
                    self.board[tuple(old_pos)] = MarkovSoccer.EMPTY
                    self.board[tuple(new_pos)] = i + 1

                    # Update reference in helper dictionary.
                    self.item_to_coord[i+1] = new_pos

                    # If current player has ball, update ball location as well
                    if self.has_ball[i]:
                        self.item_to_coord[MarkovSoccer.BALL] = new_pos

        # Computer observations for learning algorithms to use.
        obs = self._compute_observations()
        return obs, rewards, dones, None


    def _board_reset(self):
        """
        Reset game and board internally.
        """
        # Internal board information.
        length = MarkovSoccer.BOARD_LENGTH
        self.board = np.zeros((length+2, length+2), dtype=np.int8)
        self.item_to_coord = {}
        self.has_ball = [False for _ in range(MarkovSoccer.NUM_PLAYERS)]

        # Set coordindates in the map as defaults.
        positions = random.sample(self.possible_coords, 5)
        for i, pos in enumerate(positions):
            self.item_to_coord[i+1] = pos
            self.board[tuple(pos)] = i+1

        # Set boundary to either invalid positions or goal positions.
        for i in range(length + 2):
            if i == length // 2 or i == length // 2 + 1:
                self.board[0][i] = MarkovSoccer.GOAL_START_NUM + 1
                self.board[i][length+1] = MarkovSoccer.GOAL_START_NUM + 2
                self.board[length+1][i] = MarkovSoccer.GOAL_START_NUM + 3
                self.board[i][0] = MarkovSoccer.GOAL_START_NUM + 4

            else:
                self.board[i][0] = MarkovSoccer.INVALID_POS
                self.board[0][i] = MarkovSoccer.INVALID_POS
                self.board[i][length + 1] = MarkovSoccer.INVALID_POS
                self.board[length + 1][i] = MarkovSoccer.INVALID_POS


    def _compute_observations(self):
        """
        For each player, compute distance from ball, from respective goal,
        and from other players.
        """
        ball_pos = self.item_to_coord[MarkovSoccer.BALL]
        states = []

        # Calculate states for each player.
        for i in range(MarkovSoccer.NUM_PLAYERS):
            curr_pos = self.item_to_coord[i+1]
            state = []

            # Compute distance of other goals from current player
            # in order of clockwise goals.
            for j in range(1, MarkovSoccer.NUM_PLAYERS):
                other = (i+j) % MarkovSoccer.NUM_PLAYERS
                for goal_pos in self.goal_positions[other]:
                    state.append(goal_pos - curr_pos)

            # Append ball offset from current player.
            state.append(ball_pos - curr_pos)

            # Computer offset of other players, in order of clockwise goals.
            for j in range(1, MarkovSoccer.NUM_PLAYERS):
                other = (i+j) % MarkovSoccer.NUM_PLAYERS
                other_pos = self.item_to_coord[other+1]
                state.append(other_pos - curr_pos)
            states.append(state)

        states_np = np.array(states)

        # Order state vectors in order of current player then clockwise players.
        observations = []
        for i in range(MarkovSoccer.NUM_PLAYERS):
            obs = np.roll(states_np, -i, axis=0).flatten()
            observations.append(obs)

        return observations

    def render(self):
        if self.render_window is None:
            self.render_window = graphics.GraphWin(width=400, height=400)

        # Clear previous drawing.
        self.render_window.delete("all")

        for i in range(MarkovSoccer.BOARD_LENGTH + 2):
            for j in range(MarkovSoccer.BOARD_LENGTH + 2):
                # Draw borders of game.
                if self.board[i][j] == -1:
                    sq = graphics.Rectangle(
                        graphics.Point(50*j, 50*i),
                        graphics.Point(50*(j+1), 50*(i+1))
                    )
                    sq.setFill('black')
                    sq.draw(self.render_window)

                # Draw players and matching goals.
                elif self.board[i][j] > MarkovSoccer.GOAL_START_NUM:
                    player_goal = graphics.Rectangle(
                        graphics.Point(50*j, 50*i),
                        graphics.Point(50*(j+1), 50*(i+1))
                    )
                    color = MarkovSoccer.COLORS[self.board[i][j] % 10 - 1]
                    player_goal.setFill(color)
                    player_goal.setOutline(color)
                    player_goal.draw(self.render_window)

                # Draw players.
                elif self.board[i][j] >= 1 and self.board[i][j] <= 4:
                    player = graphics.Circle(
                        graphics.Point(50*(j + 0.5), 50*(i + 0.5)),
                        20
                    )
                    color = MarkovSoccer.COLORS[self.board[i][j] % 10 - 1]
                    player.setOutline(color)
                    player.setWidth(5)
                    player.draw(self.render_window)

        # Draw ball on canvas.
        ball_x, ball_y = self.item_to_coord[MarkovSoccer.BALL]
        ball = graphics.Circle(
            graphics.Point(50*(ball_y + 0.5), 50*(ball_x + 0.5)),
            10
        )
        ball_color = MarkovSoccer.COLORS[MarkovSoccer.BALL - 1]
        ball.setFill(ball_color)
        ball.setOutline(ball_color)
        ball.draw(self.render_window)


    def close(self):
        if self.render_window:
            self.render_window.close()

if __name__ == '__main__':
    env = MarkovSoccer()
    print(MarkovSoccer.COLORS)
    while (True):
        env.render()
        input_strs = input("0 = stand, 1 = up, 2 = right, 3 = down, 4 = left: ").split(" ")
        input_vals = [int(s) for s in input_strs]
        assert len(input_vals) == 4

        obs, rews, dones, _ = env.step(input_vals)

        if all(dones):
            break
    print(rews)
    env.close()
