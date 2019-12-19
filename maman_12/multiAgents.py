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
        score, move = self._max_value(gameState, 0, True)
        return move


    def _max_value(self, game_state, current_depth, return_move=False):
        values = []
        if self._is_max_terminal_state(game_state, current_depth):

            for move in game_state.getLegalActions(0):
                score = game_state.generatePacmanSuccessor(move).getScore()
                if return_move:
                    score = score, move
                values.append(score)
            if not values:
                return self.evaluationFunction(game_state)

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
        if current_depth == self.depth and not self._is_ghost_id_exist(game_state, agent_id + 1):
            return True
        return False

    def _is_max_terminal_state(self, game_state, current_depth):
        if current_depth == self.depth:
            return True
        return game_state.isLose() or game_state.isWin()


class AlphaBetaAgent(MultiAgentSearchAgent):
    _nodes_open = 0
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        # self._nodes_open = 0
        # try:
        return self._max_value(game_state, 0, (-float("inf"), float("inf")), True)[1]

    # finally:
    #     print "node open: " + str(self._nodes_open)

    def _max_value(self, game_state, current_depth, alpha_beta, return_move=False):
        self._nodes_open += 1
        if self._is_max_terminal_state(game_state, current_depth):
            values = []
            for move in game_state.getLegalActions(0):
                score = game_state.generatePacmanSuccessor(move).getScore()
                if return_move:
                    score = score, move
                values.append(score)
            if not values:
                return game_state.getScore()
            else:
                if return_move:
                    return max(values, key=lambda x: x[0])
                else:
                    return max(values)

        alpha, beta = alpha_beta
        v = -float("inf")
        to_return = []

        for move in game_state.getLegalActions(0):
            if move != 'Stop':
                v = max(v, self._min_value(game_state.generateSuccessor(0, move), current_depth + 1, (alpha, beta), 1))
                to_return.append((v, move))
                if v >= beta:
                    if return_move:
                        return [(val, mov) for val, mov in to_return if val == v][0]
                    return v
                alpha = max(alpha, v)
        if return_move:
            return [(val, mov) for val, mov in to_return if val == v][0]
        return v

    def _min_value(self, game_state, current_depth, alpha_beta, agent_id):
        self._nodes_open += 1
        if self._is_min_terminal_state(game_state, current_depth, agent_id):
            return game_state.getScore()

        alpha, beta = alpha_beta
        v = float("inf")

        for move in game_state.getLegalActions(agent_id):
            if not self._is_this_final_ghosts_to_check(game_state, agent_id):
                v = min(v, self._min_value(game_state.generateSuccessor(agent_id, move), current_depth, (alpha, beta),
                                           agent_id + 1))
            else:
                v = min(v,
                        self._max_value(game_state.generateSuccessor(agent_id, move), current_depth + 1, (alpha, beta)))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def _is_this_final_ghosts_to_check(self, game_state, agent_id):
        return not self._is_ghost_id_exist(game_state, agent_id + 1)

    def _is_ghost_id_exist(self, game_state, agent_id):
        ghost_count = game_state.getNumAgents()
        return agent_id != 0 and agent_id < ghost_count

    def _is_max_terminal_state(self, game_state, current_depth):
        if current_depth == self.depth:
            return True
        return game_state.isLose() or game_state.isWin()

    def _is_min_terminal_state(self, game_state, current_depth, agent_id):
        if game_state.isLose() or game_state.isWin():
            return True
        if current_depth == self.depth and not self._is_ghost_id_exist(game_state, agent_id + 1):
            return True
        return False


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
        return self._max_value(gameState, 0, True)[1]

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
                    min_value = self._ghost_average(game_state.generatePacmanSuccessor(move), current_depth + 1)
                    if return_move:
                        min_value = min_value, move
                    values.append(min_value)

        if return_move:
            return max(values, key=lambda x: x[0])

        return max(values)

    def _ghost_average(self, game_state, depth):
        values = self._get_scores(game_state, depth, 1)
        avg = sum(values) * 1.0 / len(values)
        return avg

    def _get_scores(self, game_state, depth, agent):

        if self._is_min_terminal_state(game_state, depth, agent):
            return [game_state.getScore()]

        values = []

        for move in game_state.getLegalActions(agent):
            if self._is_this_final_ghosts_to_check(game_state, agent):
                val = self._max_value(game_state.generateSuccessor(agent, move), depth + 1)
                values.append(val)
            else:
                values += self._get_scores(game_state.generateSuccessor(agent, move), depth, agent + 1)

        return values

    def _is_max_terminal_state(self, game_state, current_depth):
        if current_depth == self.depth:
            return True
        return game_state.isLose() or game_state.isWin()

    def _is_this_final_ghosts_to_check(self, game_state, agent_id):
        return not self._is_ghost_id_exist(game_state, agent_id + 1)

    def _is_ghost_id_exist(self, game_state, agent_id):
        ghost_count = game_state.getNumAgents()
        return agent_id != 0 and agent_id < ghost_count

    def _is_min_terminal_state(self, game_state, current_depth, agent_id):
        if game_state.isLose() or game_state.isWin():
            return True
        if current_depth == self.depth and not self._is_ghost_id_exist(game_state, agent_id + 1):
            return True
        return False


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
