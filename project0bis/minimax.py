from pacman_module.game import Agent, Directions
import pacman_module.util

class PacmanAgent(Agent):
    """Pacman agent based on Minimax."""

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.
        Arguments:
            state: a game state. See API or class `pacman.GameState`.
        Returns:
            A legal move as defined in `game.Directions`.
        """
        _, action = self.minimax(state, set(), 0, float('-inf'), float('inf'), 0)
        return action
    
    def minimax(self, state, path_visited, agent_index, alpha, beta, depth):
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
        if state.isWin():
            return 10000 - depth, Directions.STOP
        if state.isLose():
            return -10000 + depth, Directions.STOP

        state_key = (
            state.getPacmanPosition(),
            state.getFood(),
            tuple(state.getGhostPositions()),
        )
        if state_key in path_visited:
            return -10000 + depth, Directions.STOP

        new_visited = path_visited | {state_key}
        next_agent = (agent_index + 1) % state.getNumAgents()
        if next_agent == 0:
            next_depth = depth + 1
        else:
            next_depth = depth

        # Pacman's turn (MAX)
        if agent_index == 0:
            successors = state.generatePacmanSuccessors()
            if not successors:
                return -10000 + depth, Directions.STOP
            
            best_score = float("-inf")
            best_action = Directions.STOP

            for successor, action in successors:
                score, _ = self.minimax(successor, new_visited, next_agent, alpha, beta, next_depth)
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
                return 10000 - depth, Directions.STOP
            
            best_score = float("inf")
            best_action = Directions.STOP
            
            for successor, action in successors:
                score, _ = self.minimax(successor, new_visited, next_agent, alpha, beta, next_depth)
                if score < best_score:
                    best_score = score
                    best_action = action
            
                if best_score <= alpha: 
                    return best_score, best_action
                else:
                    beta = min(beta, best_score)

            return best_score, best_action