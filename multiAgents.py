# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foods = newFood.asList()
        plus =0
        temp = []
        for food in foods:
            temp.append(manhattanDistance(newPos, food))
        if temp:
            minimum = min(temp)
        else:
            return successorGameState.getScore()
        plus +=minimum

        for ghost in successorGameState.getGhostPositions():
            distance = manhattanDistance(newPos, ghost)
            item = [item for item in newScaredTimes]
            if min(item) > 0:#if we have eaten power food (or we are near one)
                plus += 50 #give points because we make ghosts to go away
            elif distance < 2:#incase of ghost being near, if we had < 1 the ghosts next move could be deadly for us
                plus -= 100
        return successorGameState.getScore()+ pow(plus, -1)#antistrofh timh toy plus

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.max_step(gameState, 0)[0] #starting with max becasue pacman is max (tuple: 0: action, 1: evaluation value)
        util.raiseNotDefined()


    def minimax(self, gameState, agent, depth):#this function is a "controller" to check if we are done and to run the correct next steps for the max_value and min_value
        if (depth == self.depth * gameState.getNumAgents()) or gameState.isWin() or gameState.isLose(): #checking if we need to finish the repercussion and return evaluation number
            return self.evaluationFunction(gameState) #because of the fact that we use depth we have to find the desire depth for every agent so we need to have as many repercussions as self.depth * numAgents
        elif agent == 0:
            return self.max_step(gameState, depth)[1] #returns the value instead of the action cause we dont need it into our max_value or min_value fuc
        else:
            return self.min_step(gameState, agent, depth)[1] #adding agent cause in max we always use pacman (index 0) for ghosts we have from one to numAgents-1

    def max_step(self, gameState, depth):
        best = []
        best.append((None, float('-Inf'))) #initialize the list with a very small number (found representation of infinite online)
        for action in gameState.getLegalActions(0):
            best.append((action, self.minimax(gameState.generateSuccessor(0, action), (depth+1) % gameState.getNumAgents(), depth+1))) #for every action run minmax to see if we have finished or what the next step is and higher the depth by one
            #we need (depth +1) % numAgents because we need to go from 0 to numAgents but for every depth that we are in (as much as you go higher in depth, the output of depth%numAgents will be from 0 to numAgents)
        maximum = (None, float('-Inf')) #find out the maximum value out of all, tried to use max() but couldnt make it work so used the old-fashion way
        for item in best:
            if item[1] > maximum[1]:
                maximum = item
        return maximum

    def min_step(self, gameState, agent, depth): #similar with max_value but added agent because we are going through many ghost-agents
        best = []
        best.append((None, float('+Inf')))
        for action in gameState.getLegalActions(agent):
            best.append((action, self.minimax(gameState.generateSuccessor(agent, action), (depth+1) % gameState.getNumAgents(), depth+1)))
        minimum = (None, float('+Inf'))
        for item in best:
            if item[1] < minimum[1]: #and changed to minimum
                minimum = item
        return minimum


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
