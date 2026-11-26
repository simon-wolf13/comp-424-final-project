
"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time
import math

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"

        # Set up Tree
        

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """

        # PLAN : Use random_agent.py for default policy, we can improve on default policy later. 



        start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.
        return random_move(board, reserves, player)

class MCTS:

    def __init__(self, state, ITERATIONS):
        self.root =  self.Node(state)
        self.ITERATIONS = ITERATIONS

    def select_node(self):
        current = self.root
        while(not current.isTerminal()):

            if current.untried_moves:
                return current.expand()
            else:
                current = max(current.children, key=lambda c: c.get_node_score())
        return current

    def backpropagate(self, node, result, winner):

        # Award points to nodes depending on if terminal node is winning for that player
        if winner == 0:
            reward = 0.5
        else:
            if winner == node.state.player:
                reward = 1
            else:
                reward = 0


        # walk up to root (only node which should have no parent) 
        while (node is not None):
            node.wins += reward
            node.visits += 1
            # we need to alternate the reward at every step since players alternate turns
            reward = 1 - reward
            node = node.parent
    
    def select_best_move(self):
        """Selects the best child of the root"""
        # We don't use score here since we are uninterested in exploration 
        return max(self.root.children, key=lambda c: (c.score)/(c.visits))

    
    def Tree_policy(self):
        curr = self.root
        for i in range(self.ITERATIONS):
            curr = self.select_node(self.root)
            result = self.random_rollout(curr)
            self.backpropagate(curr, result)

        best_move = self.select_best_move()

        return best_move

    def random_rollout(self, curr):
        """plays random moves for both sides until we reach the end of a game, returns the winning player's number"""
        state = curr.state.copy()
        while(not state.isTerminal()):
            move = random_move(state.board, state. reserves, state.player)
            state.apply_move(move)
        endgame = check_endgame(state.board, state.reserves, state.player, state.ply)
        return endgame[1]


class Node:
    """A node representing a state in the Monte Carlo search tree."""
    C = math.sqrt(2)

    def __init__(self, state):
        self.state = state
        self.parent = None
        self.children = []
        self.score = 0
        self.wins = 0
        #We keep a list of untried moves in the node so we can pop when expanding 
        self.untried_moves = get_valid_moves(state.board, state.reserves, state.player)

    def get_node_score(self):
        reward = self.wins / self.score
        score = reward + self.C * math.sqrt(math.log(self.parent.visits))/ math.log(self.score)
        return score


    def expand(self):
        """Creates a new node with an untried move from state, returns it"""
        move = self.untried_moves.pop()
        new_state = self.state.copy()
        new_state.apply_move(move)
        new_node = Node(new_state)
        # Flip players 
        new_node.parent = self
        self.children.append(new_node)
        return(new_node)


    def isTerminal(self):
        """returns True if this node corresponds to the end of the game"""
        return self.state.isTerminal()
class State:
    def __init__(self,board, reserves, player, opponent, ply):
        self.board = board
        self.reserves = reserves
        self.player = player
        self.opponent = opponent
        self.ply = 0

    def copy(self):
        # built in copy state method
        board_copy, reserves_copy = copy_state(self)
        new_state = State(board_copy, reserves_copy, self.player, self.opponent, self.ply)

    def apply_move(self, move):
        if (not move == None):
            execute_move(self.board, self.reserves, self.player, move)
        tmp = self.player
        self.player = self.opponent
        self.opponent = tmp
        self.ply +=1

    def isTerminal(self):
        """returns True if this state corresponds to the end of the game"""
        return check_endgame(self.board, self.reserves, self.player, self.ply)[0]





