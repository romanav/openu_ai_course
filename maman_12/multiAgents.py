# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import search
from searchAgents import AnyFoodSearchProblem, PositionSearchProblem
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

    def getAction(self, gameState):
        """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if action == 'Stop':
            return -100

        if not len(successorGameState.getFood().asList()):
            return 0  # if we ate all  food we need to return something to make run finish

        food = search.breadthFirstSearch(AnyFoodSearchProblem(successorGameState))  # find closest food path

        if not [i for i in newScaredTimes if i != 0]:  # if we have any ghost that not scared get
            ghost = search.breadthFirstSearch(AnyGhostSearchProblem(successorGameState))  # find closest ghost path
            if len(ghost) <= 1:  # if we too close to ghost => don't go there
                return -999999
        if len(currentGameState.getFood().asList()) > len(
                successorGameState.getFood().asList()):  # in case we are on food cell
            return 0
        return -len(food)  # return distance to food with minus sings, close food get higher score


class AnyGhostSearchProblem(PositionSearchProblem):
    """
    Same as any food, but find any ghost
    """

    def __init__(self, gameState):
        # Store the food for later reference
        self.food = [i.getPosition() for i in gameState.getGhostStates()]

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        if state in self.food:
            return True
        return False


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
        "*** YOUR CODE HERE ***"
        search = MiniMaxSearch(self.depth)
        return search.decision(gameState)


class MiniMaxSearch(object):

    def __init__(self, depth):
        self._max_depth = depth

    def decision(self, game_state):
        current_depth = 0
        legal_moves = game_state.getLegalActions()
        to_return = []
        for move in legal_moves:
            to_return.append((move, self._min_value(game_state.generatePacmanSuccessor(move), current_depth)))

        max_val = max(to_return, key=lambda x: x[1])[1]
        return random.choice([i[0] for i in to_return if i[1] == max_val])

    def _max_value(self, game_state, current_depth):

        current_depth += 1
        if self._is_terminal_state(game_state, current_depth):
            return game_state.getScore()

        min_values =[]
        for move in game_state.getLegalActions(0):
            min_values.append(self._min_value(game_state.generatePacmanSuccessor(move), current_depth))
        return max(min_values)

    def _min_value(self, game_state, current_depth):
        current_depth += 1
        if self._is_terminal_state(game_state, current_depth):
            return game_state.getScore()

        max_values = []
        for move in game_state.getLegalActions(1):
            max_values.append(self._max_value(game_state.generateSuccessor(1, move), current_depth))

        return min(max_values)

    def _is_terminal_state(self, game_state, current_depth):
        if current_depth == self._max_depth:
            return True
        return game_state.isLose() or game_state.isWin()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
  """

    def getAction(self, gameState):
        """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
