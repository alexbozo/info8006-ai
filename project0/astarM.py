from pacman_module.game import Agent, Directions
import pacman_module.util
from pacman_module.util import PriorityQueue
import pacman_module.pacman

def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules())
        
    )

def heuristic(state):
    """Returns the distance between Pacman and the farthest food using the Manhattan Distance.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        The distance between Pacman and the farthest food.
    """

    # Variables initialization
    pacman_pos = state.getPacmanPosition()
    food_list = state.getFood().asList()

    # Base case 
    if(len(food_list) == 0):
        return 0

    # Computing the distance for each food element
    distances = []
    for food in food_list:
        distances.append(pacman_module.util.manhattanDistance(pacman_pos, food))

    # Choose the farthest one
    heuristic = max(distances)
    
    return heuristic

class PacmanAgent(Agent):
    """Pacman agent based on A*."""

    def __init__(self):
        super().__init__()
        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        if self.moves == None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """

        # Variables initialization
        fringe = PriorityQueue()
        start_g_cost = 0
        start_f_cost = start_g_cost + heuristic(state)

        fringe.push((state, [], start_g_cost), start_f_cost)
        closed = set()

        # Looping and popping
        while True:
            # No solution found
            if fringe.isEmpty():
                return []

            #remove the state with the lowest f_cost 
            f_cost, (currentState, path, g_cost) = fringe.pop()

            # Win (if Pacman has eaten all food and capsules)
            if currentState.isWin():
                return path

            # Already visited ?
            current_key = key(currentState)
            if current_key in closed:
                continue

            # Add to the visited set
            closed.add(current_key)

            # Loop on all the possible next states and their respective actions from the current state
            for successor, action in currentState.generatePacmanSuccessors():
                new_g_cost = g_cost + 1
                new_f_cost = new_g_cost + heuristic(successor)
                
                # Add the successors to the fringe. It will be the pop() that will choose which of the successors is better and where to explore then.
                fringe.push((successor, path + [action], new_g_cost), new_f_cost)