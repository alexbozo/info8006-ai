from pacman_module.game import Agent, Directions
import pacman_module.util

class PacmanAgent(Agent):
    """Pacman agent controlled by the arrow keys."""

    def __init__(self):
        super().__init__()
        self.max_depth = 2

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.
        Arguments:
            state: a game state. See API or class `pacman.GameState`.
        Returns:
            A legal move as defined in `game.Directions`.
        """
        _, action = self.minimax(state, 0, 0)
        return action

    def evaluation(self, state):
        score = state.getScore()

        # favoriser une nourriture proche
        score -= state.manhattanDistance(pacman_module.util.manhattanDistance.getPacmanPosition(), state.getFood().asList()[0]) * 0.1
        return score

    
    def minimax(self, state, depth, agent_index):
        """Minimax algorithm for Pacman.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: current depth in the game tree.
            agent_index: index of the current agent (0 for Pacman, 1 for Ghosts).

        Returns:
            A tuple (score, action) where score is the minimax score and action is the
            best action for Pacman to take at this state.
        """

        # Base case
        if state.isWin() or state.isLose():
            return state.getScore(), Directions.STOP

        # Maximum depth reached
        if depth == self.max_depth:
            return self.evaluation(state), Directions.STOP

        legal_actions = state.getLegalActions(agent_index)
        next_agent = (agent_index + 1) % state.getNumAgents()

        if next_agent == 0: # Pacman
            next_depth = depth + 1
        else:               # Ghosts
            next_depth = depth

        # Pacman's turn
        if agent_index == 0:
            best_score = float("-inf")
            best_action = Directions.STOP

            for action in legal_actions:
                successor = state.generateSuccessor(agent_index, action)
                score, _ = self.minimax(successor, next_depth,next_agent)

                if score > best_score:
                    best_score = score
                    best_action = action

            return best_score, best_action

        # Ghosts turn
        else:
            best_score = float("inf")

            for action in legal_actions:
                successor = state.generateSuccessor(agent_index, action)
                score, _ = self.minimax(successor,next_depth, next_agent)

                if score < best_score:
                    best_score = score

            return best_score, Directions.STOP