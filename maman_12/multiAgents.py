# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


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
            return -float("inf")

        ghost_dist = []

        # get all manhattan distances to ghosts
        for ghostState in newGhostStates:
            position = ghostState.getPosition()
            if manhattanDistance(newPos, position) <= 1:  # to close to ghost, run!!!
                return -float("inf")
            ghost_dist.append(manhattanDistance(successorGameState.getPacmanPosition(), position))

        # get all distances to food
        food_dist = []
        for food_pos in oldFood.asList():
            food_dist.append(manhattanDistance(successorGameState.getPacmanPosition(), food_pos))

        # closest ghost / closest food - we want to be far from ghost and close to food
        return min(ghost_dist) * 1.0 / (min(food_dist) + 0.1)


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
        score, action = self._max_value(gameState, 1, True)
        return action

    def _max_value(self, game_state, depth, return_action=False):
        if self._is_game_finished(game_state):
            return self.evaluationFunction(game_state)

        v = -float("inf"), None

        for action in game_state.getLegalActions(0):
            if action != 'Stop':
                next_state = game_state.generateSuccessor(0, action)
                v = max(v, (self._min_value(next_state, depth), action), key=lambda x: x[0])

        if return_action:
            return v
        return v[0]

    def _min_value(self, game_state, depth, agent_id=1):

        if self._is_game_finished(game_state):
            return self.evaluationFunction(game_state)

        if agent_id == self._get_ghosts_count(game_state) and depth == self.depth:
            v = float("inf")
            for action in game_state.getLegalActions(agent_id):
                v = min(v, self.evaluationFunction(game_state.generateSuccessor(agent_id, action)))
            return v

        v = float("inf")
        for action in game_state.getLegalActions(agent_id):
            if agent_id == self._get_ghosts_count(game_state):
                v = min(v, self._max_value(game_state.generateSuccessor(agent_id, action), depth + 1))
            else:
                v = min(v, self._min_value(game_state.generateSuccessor(agent_id, action), depth, agent_id + 1))
        return v

    def _is_game_finished(self, game_state):
        if game_state.isLose() or game_state.isWin():
            return True
        return False

    def _get_ghosts_count(self, game_state):
        return game_state.getNumAgents() - 1


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        score, move = self._max_value(game_state, 1, (-float("inf"), float("inf")), True)
        return move

    def _max_value(self, game_state, depth, alpha_beta, return_move=False):

        if self._is_game_finished(game_state):
            return self.evaluationFunction(game_state)

        alpha, beta = alpha_beta

        v = -float("inf"), None

        for action in game_state.getLegalActions(0):
            if action != 'Stop':
                next_state = game_state.generateSuccessor(0, action)
                v = max(v, (self._min_value(next_state, depth, (alpha, beta)), action), key=lambda x: x[0])
                if v[0] >= beta:
                    return v if return_move else v[0]
                alpha = max(alpha, v[0])
        return v if return_move else v[0]

    def _min_value(self, game_state, depth, alpha_beta, agent_id=1):
        if self._is_game_finished(game_state):
            return self.evaluationFunction(game_state)

        alpha, beta = alpha_beta
        v = float("inf")

        if agent_id == self._get_ghosts_count(game_state) and depth == self.depth:
            for action in game_state.getLegalActions(agent_id):
                v = min(v, self.evaluationFunction(game_state.generateSuccessor(agent_id, action)))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        for action in game_state.getLegalActions(agent_id):
            next_state = game_state.generateSuccessor(agent_id, action)
            if agent_id == self._get_ghosts_count(game_state):
                v = min(v, self._max_value(next_state, depth + 1, (alpha, beta)))
            else:
                v = min(v, self._min_value(next_state, depth, (alpha, beta), agent_id + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def _is_game_finished(self, game_state):
        if game_state.isLose() or game_state.isWin():
            return True
        return False

    def _get_ghosts_count(self, game_state):
        return game_state.getNumAgents() - 1


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
        score, action = self._max_value(gameState, 1, True)
        return action

    def _max_value(self, game_state, depth, return_action=False):
        if self._is_game_finished(game_state):
            return self.evaluationFunction(game_state)

        v = -float("inf"), None

        for action in game_state.getLegalActions(0):
            if action != 'Stop':
                next_state = game_state.generateSuccessor(0, action)
                v = max(v, (self._ghost_average(next_state, depth), action), key=lambda x: x[0])

        if return_action:
            return v
        return v[0]

    def _ghost_average(self, game_state, depth):
        values = self._get_scores(game_state, depth)
        avg = sum(values) * 1.0 / len(values)
        return avg

    def _get_scores(self, game_state, depth, agent_id=1):

        if self._is_game_finished(game_state):
            return [self.evaluationFunction(game_state)]

        values = []
        if agent_id == self._get_ghosts_count(game_state) and depth == self.depth:
            for action in game_state.getLegalActions(agent_id):
                values.append(self.evaluationFunction(game_state.generateSuccessor(agent_id, action)))
            return values

        for action in game_state.getLegalActions(agent_id):
            next_state = game_state.generateSuccessor(agent_id, action)
            if agent_id == self._get_ghosts_count(game_state):
                values.append(self._max_value(next_state, depth+1))
            else:
                values += self._get_scores(next_state, depth, agent_id+1)

        return values

    def _is_game_finished(self, game_state):
        if game_state.isLose() or game_state.isWin():
            return True
        return False

    def _get_ghosts_count(self, game_state):
        return game_state.getNumAgents() - 1


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
