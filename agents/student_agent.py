
"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time
import math

PRUNING_TIME = 0.5
TIME_BUDGET = 1.6


def reward_function(winner):
    if winner == PLAYER_1:
        return 1
    elif winner == PLAYER_2:
        return 0
    else:
        return 0.5

    
@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        ITERATIONS = 100
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.myturns = 0

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
        ply = 2* self.myturns + (player - 1)
        self.myturns += 1
        try:
            root_state = State(board, reserves, player, opponent, ply)
            forced_mv, safe_moves = self.root_pruning(root_state, start)
            if forced_mv is not None:
                return forced_mv
            if safe_moves:
                mcts = MCTS(root_state, safe_moves).Tree_policy(time_limit=start + TIME_BUDGET)
                return mcts
        except Exception as e:
            tb = e.__traceback__how
            while tb.tb_next is not None:
                tb = tb.tb_next
            print("step failed:", type(e).__name__, e, "at line", tb.tb_lineno)
        return random_move(board, reserves, player)


    def root_pruning(self, state, start):
        """Prune the root node the MCTS tree by removing children that are losing moves for the current player."""
        moves = get_valid_moves(state.board, state.reserves, state.player)
        if not moves:
            return None, []
        if len(moves) == 1:
            return moves[0], [] #only one valid option

        candidate_moves = []
        for mv in moves:
            child = state.copy()
            child.apply_move(mv)
            
            is_winner = check_winner(child.board,state.player)
            if is_winner ==state.player:
                return mv, []
            if is_winner==state.opponent:
                continue

            candidate_moves.append((mv, child))

        if not candidate_moves:
            return moves[0], []

        safe = []
        for i, (mv, child) in enumerate(candidate_moves):
            if time.perf_counter() - start > PRUNING_TIME:
                safe.extend(candidate_moves[i:])
                break
            if not self.opponent_wins_next(child):
                safe.append((mv, child))

        if not safe:
            return candidate_moves[0][0], []

        return None, safe


    def opponent_wins_next(self, state):
        """Check if the opponent can win in their next move."""
        p = state.player #from perspective of the opponent
        for opponent_move in get_valid_moves(state.board, state.reserves, p):
            b,r  = copy_state(state.board, state.reserves)
            execute_move(b, r, p, opponent_move)
            if check_winner(b,p) == p:
                return True
        return False

        
class MCTS:

    def __init__(self, root_state, root_moves):
        self.root =  Node(root_state)
        self.rng = np.random.default_rng() 
        self.root.untried_moves = [] 
        for mv, child in root_moves:
            self.root.children.append(Node(child, parent=self.root, move=mv))




    def Tree_policy(self, time_limit):
        iterations = 0  
        while time.perf_counter() < time_limit:
            curr = self.select_node()
            result = self.random_rollout(curr)
            self.backpropagate(curr, winner=result)
            iterations += 1
    
        best_move = self.select_best_move()
        print(" iterations", iterations)

        return best_move

    def select_node(self):
        current = self.root
        while not current.over:
            current.get_untried_moves()
            if current.untried_moves:
                return current.expand()
            else:
                current = max(current.children, key=lambda c: c.get_node_score())
        return current

    def backpropagate(self, node, winner):

        "'Award points to nodes depending on if terminal node is winning for that player'"

        reward = reward_function(winner)
        mover = node.state.opponent #opponent of the node's player is the one who made the last move
        reward_node = reward if mover == PLAYER_1 else 1 - reward

        while node is not None:
            node.visits += 1
            node.wins += reward_node
            reward_node = 1 - reward_node  # Switch perspective between players
            node = node.parent
    
    def select_best_move(self):
        """Selects the best child of the root"""
        # We don't use score here since we are uninterested in exploration 
        return max(self.root.children, key=lambda c: (c.visits)).move

    
  

    def random_rollout(self, curr):
        """plays random moves for both sides until we reach the end of a game, returns the winning player's number"""
        state = curr.state.copy()
        while(not state.isTerminal()[0]):
            move = random_move(state.board, state. reserves, state.player, self.rng)
            state.apply_move(move)
        endgame = check_endgame(state.board, state.reserves, state.player, state.ply)
        return endgame[1]


class Node:
    """A node representing a state in the Monte Carlo search tree."""
    C =0.7

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.score = 0
        self.wins = 0
        self.visits = 0
        self.rng = np.random.default_rng()

        over, winner, _ = state.isTerminal()
        self.winner = winner
        self.over = over
        self.untried_moves = None #only generate untried moves when we need to expand the node

    def get_untried_moves(self):
        if self.untried_moves is None:
            moves = get_valid_moves(self.state.board, self.state.reserves, self.state.player)
            self.untried_moves = list(moves) if moves else [None]



    def get_node_score(self):
        if self.visits ==0:
            return float('inf') #unvisited nodes are prioritized
        reward = self.wins / self.visits
        score = reward + self.C * math.sqrt((math.log(self.parent.visits))/(self.visits))
        return score


    def expand(self):
        """Creates a new node with an untried move from state, returns it"""
        i = int(self.rng.integers(0, len(self.untried_moves)))
        move = self.untried_moves.pop(i)
        new_state = self.state.copy()
        new_state.apply_move(move)
        new_node = Node(new_state, parent=self, move=move)
        self.children.append(new_node)
        return new_node


    def isTerminal(self):
        """returns True if this node corresponds to the end of the game"""
        return self.state.isTerminal()

    
class State:
    def __init__(self,board, reserves, player, opponent, ply):
        self.board = board
        self.reserves = reserves
        self.player = player
        self.opponent = opponent
        self.ply = ply

    def copy(self):
        # built in copy state method
        board_copy, reserves_copy = copy_state(self.board, self.reserves)
        new_state = State(board_copy, reserves_copy, self.player, self.opponent, self.ply)
        return new_state

    def apply_move(self, move):
        if (not move == None):
            execute_move(self.board, self.reserves, self.player, move)
        tmp = self.player
        self.player = self.opponent
        self.opponent = tmp
        self.ply +=1

    def isTerminal(self):
        """returns True if this state corresponds to the end of the game"""
        return check_endgame(self.board, self.reserves, self.player, self.ply)






        
            
               



        

        
