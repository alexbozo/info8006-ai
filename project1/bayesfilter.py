import numpy as np

from pacman_module.game import Agent, Directions, manhattanDistance
from pacman_module.util import PriorityQueue


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """
        # Variables initialization
        width = walls.width
        height = walls.height
        t_matrix = np.zeros((width, height,
                             width, height))
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        fear_values = {
            "fearless": 0.0,
            "afraid": 1.0,
            "terrified": 3.0
        }
        fear = fear_values.get(self.ghost, 1.0)
        weight_away = 2.0 ** fear

        # Build the transition matrix for each position in the grid
        for i in range(width):
            for j in range(height):
                if walls[i][j]:
                    continue

                current_distance = manhattanDistance((i, j), position)

                # Find all accessible positions from the current position
                legal_moves = []
                for move in moves:
                    (x, y) = i + move[0], j + move[1]
                    if 0 <= x < width and 0 <= y < height and not walls[x][y]:
                        legal_moves.append((x, y))

                if not legal_moves:
                    continue

                # Compute the weights for each accessible position
                weights = []
                for (x, y) in legal_moves:
                    succ_dist = manhattanDistance((x, y), position)
                    if succ_dist >= current_distance:
                        w = weight_away
                    else:
                        w = 1.0
                    weights.append(w)

                # Normalize the weights for each accessible position
                total_weight = sum(weights)
                for (x, y), w in zip(legal_moves, weights):
                    t_matrix[i, j, x, y] = w / total_weight

        return t_matrix

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        pass

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)

        pass

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _astar(self, start, goal, walls):
        """Performs A* search using the Manhattan distance heuristic to find
        the shortest path from start to goal.

        Arguments:
            start: The starting position (x, y).
            goal: The goal position (x, y).
            walls: The W x H grid of walls.

        Returns:
            The first action to take towards the goal, or Directions.STOP if
            no path is found.
        """
        if start == goal:
            return Directions.STOP

        # Variables initialization
        actions = [
            (Directions.NORTH, (0, 1)),
            (Directions.SOUTH, (0, -1)),
            (Directions.EAST, (1, 0)),
            (Directions.WEST, (-1, 0))
        ]
        fringe = PriorityQueue()
        fringe.push((start, [], 0), manhattanDistance(start, goal))
        closed = set()

        while not fringe.isEmpty():
            _, (curr_pos, path, g_cost) = fringe.pop()
            # Return the first action if the goal is reached
            if curr_pos == goal:
                return path[0] if path else Directions.STOP
            # Skip this position if it is in the closed set
            if curr_pos in closed:
                continue
            # Position has been visited, add it to the closed set
            closed.add(curr_pos)

            # Explore neighbors and add them to the fringe if they are valid
            for action, (dx, dy) in actions:
                next_x, next_y = (curr_pos[0] + dx, curr_pos[1] + dy)
                if (
                    0 <= next_x < walls.width
                    and 0 <= next_y < walls.height
                    and not walls[next_x][next_y]
                    and (next_x, next_y) not in closed
                ):
                    next_g = g_cost + 1
                    h = manhattanDistance((next_x, next_y), goal)
                    fringe.push(((next_x, next_y),
                                 path + [action],
                                 next_g), next_g + h)

        return Directions.STOP

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        best_target = None
        global_max_prob = -1.0

        # Iterate over each ghost's belief state to find the most likely
        # position of the most certain ghost
        for idx, belief in enumerate(beliefs):
            if eaten[idx]:
                continue

            # Find the position with the highest probability for this ghost
            ghost_max_prob = -1.0
            ghost_target = None
            for i in range(walls.width):
                for j in range(walls.height):
                    if belief[i, j] > ghost_max_prob:
                        ghost_max_prob = belief[i, j]
                        ghost_target = (i, j)

            # Select the ghost with the highest certainty
            if ghost_max_prob > global_max_prob:
                global_max_prob = ghost_max_prob
                best_target = ghost_target

        # If no ghost left or no valid target found
        if best_target is None:
            return Directions.STOP

        return self._astar(position, best_target, walls)

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
