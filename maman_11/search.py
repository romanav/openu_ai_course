# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
import game


class SearchProblem:
    """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).

  You do not need to change anything in this class, ever.
  """

    def getStartState(self):
        """
     Returns the start state for the search problem
     """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
       state: Search state

     Returns True if and only if the state is a valid goal state
     """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
       state: Search state

     For a given state, this should return a list of triples,
     (successor, action, stepCost), where 'successor' is a
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental
     cost of expanding to that successor
     """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
      actions: A list of actions to take

     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    return GraphSearch(problem, util.Stack()).search()

def breadthFirstSearch(problem):
    return GraphSearch(problem, util.Queue()).search()

def uniformCostSearch(problem):
    return UniformedCostSearch(problem).search()


class GraphSearch(object):

    def __init__(self,  problem, fringe):
        self.fringe = fringe
        self.problem = problem
        self.bfs_path = []
        self.closed_set = set()

    def search(self):
        self.fringe.push(self.problem.getStartState())
        path = {self.problem.getStartState(): None}

        while True:
            self.assert_fringe_empty()

            node = self.fringe.pop()

            if self.problem.isGoalState(node):
                return self.extract_path_from_walking_history(node, path)
            else:
                if node not in self.closed_set:
                    self.closed_set.add(node)
                    for successors in self.problem.getSuccessors(node):
                        if successors[0] not in self.closed_set:
                            self.fringe.push(successors[0])
                            path[successors[0]] = node, successors[1]

    def extract_path_from_walking_history(self, goal, path):
        if path[goal] is None:
            self.bfs_path.reverse()
            print ("Path Size: " + str(len(self.bfs_path)))
            return self.bfs_path
        self.bfs_path.append(path[goal][1])
        return self.extract_path_from_walking_history(path[goal][0], path)

    def assert_fringe_empty(self):
        if self.fringe.isEmpty():
            raise Exception("Search Failure, Fringe Empty")


class UniformedCostSearch(GraphSearch):
    """
    We override graph search to utilize existing methods for UCS
    """

    def __init__(self, problem):
        super(UniformedCostSearch, self).__init__(problem, FringePriorityQueue())

    def search(self):
        self.fringe.push((self.problem.getStartState(), 0))
        path = {self.problem.getStartState(): None}

        while True:
            self.assert_fringe_empty()

            node, distance = self.fringe.pop()

            if self.problem.isGoalState(node):
                return self.extract_path_from_walking_history(node, path)
            else:
                if node not in self.closed_set:
                    self.closed_set.add(node)
                    for successors in self.problem.getSuccessors(node):
                        if successors[0] not in self.closed_set:
                            new_distance = distance + successors[2]
                            self.fringe.push((successors[0], new_distance))
                            path[successors[0]] = node, successors[1]


class FringePriorityQueue(util.PriorityQueueWithFunction):
    """
    This class override PriorityQueue class.
    In reason it implemented as a Queue we cannot update item in it,
    But we can check if item was already extracted and it means that it was updated with better priority
    """

    def __init__(self):
        util.PriorityQueueWithFunction.__init__(self, lambda x: x[1])
        self.removed_items = set()

    def pop(self):
        item = util.PriorityQueueWithFunction.pop(self)
        while item in self.removed_items:
            item = util.PriorityQueueWithFunction.pop(self)
        return item



def nullHeuristic(state, problem=None):
    """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
