from pacman_module.game import Agent, Directions
import pacman_module.util

class PacmanAgent(Agent):
    """Pacman agent controlled by Minimax."""

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.
        Arguments:
            state: a game state. See API or class `pacman.GameState`.
        Returns:
            A legal move as defined in `game.Directions`.
        """
        _, action = self.minimax(state, set(), 0, float('-inf'), float('inf'))
        return action
    
    def minimax(self, state, path_visited, agent_index, alpha, beta):
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
        
        state_key = (
            state.getPacmanPosition(),
            state.getFood(),
            tuple(state.getGhostPositions())
        )
        if state_key in path_visited:
            return state.getScore(), Directions.STOP
    
        new_visited = path_visited | {state_key}

        legal_actions = state.getLegalActions(agent_index)
        next_agent = (agent_index + 1) % state.getNumAgents()
    
        # Pacman's turn
        if agent_index == 0:
            best_score = float("-inf")
            best_action = Directions.STOP

            for action in legal_actions:
                successor = state.generateSuccessor(agent_index, action)
                score, _ = self.minimax(successor, new_visited ,next_agent, alpha, beta)

                if score > best_score:
                    best_score = score
                    best_action = action
                
                if best_score >= beta: 
                    return best_score, best_action
                else :
                    alpha = max(alpha, best_score)

            return best_score, best_action

        # Ghosts turn
        else:
            best_score = float("inf")
            best_action = Directions.STOP

            for action in legal_actions:
                successor = state.generateSuccessor(agent_index, action)
                score, _ = self.minimax(successor, new_visited, next_agent, alpha, beta)

                if score < best_score:
                    best_score = score
                    best_action = action
                    
                if best_score <= alpha:
                    return best_score, best_action
                else :
                    beta = min(beta, best_score)

            return best_score, best_action