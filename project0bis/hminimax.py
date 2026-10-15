from pacman_module.game import Agent, Directions
import pacman_module.util


class PacmanAgent(Agent):
    """Pacman agent based on H-Minimax."""

    def __init__(self):
        super().__init__()
        self.max_depth = 4
        self.dist = None

    def _init_dist(self, state):
        walls = state.getWalls()
        open_cells = [
            (x, y) for x in range(walls.width) for y in range(walls.height)
            if not walls[x][y]
        ]
        self.dist = {}
        for src in open_cells:
            queue = [src]
            visited = {src: 0}
            for curr in queue:
                d = visited[curr]
                cx, cy = curr
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxt = (cx + dx, cy + dy)
                    if nxt in open_cells and nxt not in visited:
                        visited[nxt] = d + 1
                        queue.append(nxt)
            for dst, d in visited.items():
                self.dist[(src, dst)] = d

    def maze_dist(self, p1, p2):
        return self.dist.get((p1, p2), abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.
        Arguments:
            state: a game state. See API or class `pacman.GameState`.
        Returns:
            A legal move as defined in `game.Directions`.
        """
        if self.dist is None:
            self._init_dist(state)
        _, action = self.minimax(state, 0, float('-inf'), float('inf'), 0)

        return action

    def evaluation(self, state, depth):
        if state.isWin():
            return 10000 - depth
        if state.isLose():
            return -10000 + depth

        score = state.getScore()
        food_list = state.getFood().asList()
        pacman_position = state.getPacmanPosition()
        ghost_positions = state.getGhostPositions()

        # Distance to the nearest ghost
        min_ghost_dist = float("inf")
        for g_pos in ghost_positions:
            g_cell = (g_pos[0], g_pos[1])
            d = self.maze_dist(pacman_position, g_cell)
            min_ghost_dist = min(min_ghost_dist, d)

        if min_ghost_dist <= 1:
            score -= 500
        elif min_ghost_dist == 2:
            score -= 50

        food_distances = []

        if food_list:
            for f in food_list:
                distance = self.maze_dist(pacman_position, f)
                food_distances.append(distance)
            score -= min(food_distances) * 2
            score -= len(food_list) * 20

        return score

    def minimax(self, state, agent_index, alpha, beta, depth):
        """H-Minimax algorithm for Pacman.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: current depth in the game tree.
            agent_index: index of the current agent (0 for Pacman, >0
            for Ghosts).

        Returns:
            A tuple (score, action) where score is the minimax score and
            action is the
            best action for Pacman to take at this state.
        """

        # Base case
        if state.isWin():
            return 10000 - depth, Directions.STOP
        if state.isLose():
            return -10000 + depth, Directions.STOP

        # Maximum depth reached
        if depth == self.max_depth:
            return self.evaluation(state, depth), Directions.STOP

        next_agent = (agent_index + 1) % state.getNumAgents()
        if next_agent == 0:
            next_depth = depth + 1
        else:
            next_depth = depth

        # Pacman's turn (MAX)
        if agent_index == 0:
            successors = state.generatePacmanSuccessors()
            if not successors:
                return self.evaluation(state, depth), Directions.STOP

            best_score = float("-inf")
            best_action = Directions.STOP

            for successor, action in successors:
                score, _ = self.minimax(
                                        successor,
                                        next_agent,
                                        alpha,
                                        beta,
                                        next_depth
                                        )
                if score > best_score:
                    best_score = score
                    best_action = action

                if best_score >= beta:
                    return best_score, best_action
                else:
                    alpha = max(alpha, best_score)

            return best_score, best_action

        # Ghosts turn (MIN)
        else:
            successors = state.generateGhostSuccessors(agent_index)
            if not successors:
                return self.evaluation(state, depth), Directions.STOP

            best_score = float("inf")
            best_action = Directions.STOP

            for successor, action in successors:
                score, _ = self.minimax(
                                        successor,
                                        next_agent,
                                        alpha,
                                        beta,
                                        next_depth
                                        )
                if score < best_score:
                    best_score = score
                    best_action = action

                if best_score <= alpha:
                    return best_score, best_action
                else:
                    beta = min(beta, best_score)

            return best_score, best_action
