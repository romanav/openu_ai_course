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


def nullHeuristic(state, problem=None):
    """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
    return 0


def manhattan_heuristic(state, problem=None):
    """Method to save some writings here, from state to the goal of prblem"""
    return util.manhattanDistance(state[0], problem.goal)


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
    return HeuristicGraphSearch(problem, nullHeuristic).search()


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return HeuristicGraphSearch(problem, heuristic).search()


class GraphSearch(object):

    def __init__(self, problem, fringe):
        self.fringe = fringe  # frontier data structure
        self.problem = problem
        self.bfs_path = []  # We save path from start to goal here
        self.closed_set = set()  # Store visited nodes

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
                            path[successors[0]] = node, successors[1]  # store path

    def extract_path_from_walking_history(self, goal, path):
        """
        WE extracting path from visited node in format of directions
        :param goal:  - our gaol state
        :param path:  - visited node set
        :return:
        """
        if path[goal] is None:
            self.bfs_path.reverse()
            # print ("Path Size: " + str(len(self.bfs_path)))
            return self.bfs_path
        self.bfs_path.append(path[goal][1])
        return self.extract_path_from_walking_history(path[goal][0], path)

    def assert_fringe_empty(self):
        if self.fringe.isEmpty():
            raise Exception("Search Failure, Fringe Empty")


class HeuristicGraphSearch(GraphSearch):
    """
    We override graph search to utilize existing methods for UCS
    WE just want to not duplicate the code
    """

    def __init__(self, problem, heuristic):
        super(HeuristicGraphSearch, self).__init__(problem, FringePriorityQueue())
        self.heuristic = heuristic

    def search(self):
        self.fringe.push((self.problem.getStartState(), 0, 0))
        path = {self.problem.getStartState(): None}

        while True:
            self.assert_fringe_empty()

            node, heuristic_distance, distance = self.fringe.pop()

            if self.problem.isGoalState(node):
                return self.extract_path_from_walking_history(node, path)
            else:
                if node not in self.closed_set:
                    self.closed_set.add(node)
                    for successors in self.problem.getSuccessors(node):
                        heuristic_distance = distance + successors[2] + self.heuristic(successors[0], self.problem)
                        if successors[0] not in self.closed_set and not self.fringe.isContain(successors):
                            self.fringe.push((successors[0], heuristic_distance, distance + successors[2]))
                            path[successors[0]] = node, successors[1]
                        elif self.fringe.isContain(successors): # update node with better heuristics values if needed
                            is_replaced = self.fringe.push(
                                (successors[0], heuristic_distance, distance + successors[2]))
                            if is_replaced: # handle path changes when frontier updated
                                path[successors[0]] = node, successors[1]


class FringePriorityQueue(object):
    """
    This is implementation of priority data set for the A* and UniformCost search algorithms
    """

    def __init__(self):
        self.get_cost = lambda x: x[1]
        self.get_name = lambda x: x[0]
        self.get_distance = lambda x: x[2]
        self.heap = []

    def push(self, item):
        replaced_in_heap = False
        for i in self.heap:  # update heuristics value if new is smaller
            node = i[1]
            if self.get_name(node) == self.get_name(item):
                if self.get_distance(item) < self.get_distance(node):
                    self.heap.remove(i)
                    game.heapq.heapify(self.heap)
                    replaced_in_heap = True

        game.heapq.heappush(self.heap, (self.get_cost(item), item))
        return replaced_in_heap

    def isContain(self, item):
        for i in self.heap:
            node = i[1]
            if self.get_name(node) == self.get_name(item):
                return True
        return False

    def pop(self):
        return game.heapq.heappop(self.heap)[1]

    def isEmpty(self):
        return len(self.heap) == 0


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
