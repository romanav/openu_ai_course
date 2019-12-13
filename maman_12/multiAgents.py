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
        return self._max_value(game_state, 0, True)[1]

    def _max_value(self, game_state, current_depth, return_move=False):
        values = []
        if self._is_max_terminal_state(game_state, current_depth):
            for move in game_state.getLegalActions(0):
                score = game_state.generatePacmanSuccessor(move).getScore()
                if return_move:
                    score = score, move
                values.append(score)
            if not values:
                return game_state.getScore()

        else:
            for move in game_state.getLegalActions(0):
                if move != 'Stop':
                    min_value = self._min_value(game_state.generatePacmanSuccessor(move), current_depth + 1, 1)
                    if return_move:
                        min_value = min_value, move
                    values.append(min_value)

        if return_move:
            return max(values, key=lambda x: x[0])

        return max(values)

    def _min_value(self, game_state, current_depth, agent_id):
        values = []
        if self.is_min_terminal_state(game_state, current_depth, agent_id):
            return game_state.getScore()

        else:
            if self._is_this_final_ghosts_to_check(game_state, agent_id):
                for move in game_state.getLegalActions(agent_id):
                    max_value = self._max_value(game_state.generateSuccessor(agent_id, move), current_depth + 1)
                    values.append(max_value)

            else:
                for move in game_state.getLegalActions(agent_id):
                    min_value = self._min_value(game_state.generateSuccessor(agent_id, move), current_depth,
                                                agent_id + 1)
                    values.append(min_value)

        return min(values)

    def _is_this_final_ghosts_to_check(self, game_state, agent_id):
        return not self._is_ghost_id_exist(game_state, agent_id + 1)

    def _is_ghost_id_exist(self, game_state, agent_id):
        ghost_count = game_state.getNumAgents()
        return agent_id != 0 and agent_id < ghost_count

    def is_min_terminal_state(self, game_state, current_depth, agent_id):
        if game_state.isLose() or game_state.isWin():
            return True
        if current_depth == self._max_depth and not self._is_ghost_id_exist(game_state, agent_id + 1):
            return True
        return False

    def _is_max_terminal_state(self, game_state, current_depth):
        if current_depth == self._max_depth:
            return True
        return game_state.isLose() or game_state.isWin()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
  """

    def getAction(self, game_state):
        return self._max_value(game_state, 0, (-float("inf"), float("inf")), True)[1]

    def _max_value(self, game_state, current_depth, alpha_beta, return_move=False):
        if self._is_max_terminal_state(game_state, current_depth):
            return game_state.getScore()

        alpha, beta = alpha_beta
        v = -float("inf")
        to_return = []

        for move in game_state.getLegalActions(0):
            if move != 'Stop':
                v = max(v, self._min_value(game_state.generateSuccessor(0, move), current_depth + 1, (alpha, beta)))
                to_return.append((v, move))
                if v >= beta:
                    if return_move:
                        return [(val,mov) for val, mov in to_return if val == v][0]
                    return v
                alpha = max(alpha, v)
        if return_move:
            return [(val, mov) for val, mov in to_return if val == v][0]
        return v

    def _min_value(self, game_state, current_depth, alpha_beta):
        if self._is_max_terminal_state(game_state, current_depth):
            return game_state.getScore()

        alpha, beta = alpha_beta
        v = float("inf")

        for move in game_state.getLegalActions(1):
            v = min(v, self._max_value(game_state.generateSuccessor(1, move), current_depth + 1, (alpha, beta)))
            if v <= alpha:
                return v
            beta = max(beta, v)
        return v

    def _is_max_terminal_state(self, game_state, current_depth):
        if current_depth == self.depth:
            return True
        return game_state.isLose() or game_state.isWin()


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
