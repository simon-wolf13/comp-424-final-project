
"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, render_text)
from constants import *
import numpy as np
import time
import math

PRUNING_TIME = 0.5
TIME_BUDGET = 1.9
PRUNING = False
RNG = np.random.default_rng()

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
        self.mcts = MCTS()

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
            current_state = State(board, reserves, player, opponent, ply)
            if PRUNING:
                forced_mv, safe_moves = self.root_pruning(current_state, start)
            else:
                forced_mv, safe_moves = None, get_valid_moves(board, reserves, player)

            if forced_mv is not None:
                return forced_mv
            if safe_moves:
                if (self.mcts.root == None):
                    # If root is none we start tree from scratch
                    self.mcts.root = Node(current_state)
                else:
                    print("Current state : \n"  , current_state)
                    print("Root state: \n"  , self.mcts.root.state)
                    print("Root move: " , self.mcts.root.move)
                    print("Root children : ", len(self.mcts.root.children))
                    print("Tree discarded") if not self.mcts.find_state(current_state) else print("Tree preserved")
                # mcts = MCTS(root_state, safe_moves).Tree_policy(time_limit=start + TIME_BUDGET)
                move = self.mcts.get_move(safe_moves, time_limit = start + TIME_BUDGET)
                return move
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

    def __init__(self):
        # root init is done in "step"
        self.root =  None
        self.rng = np.random.default_rng() 
        # self.root.untried_moves = [] 
        #for mv, child in root_moves:
        #    self.root.children.append(Node(child, parent=self.root, move=mv))


    def get_move(self, safe_moves, time_limit):
        #TODO : implement move pruning (safe moves are being passed, but not being used currently). Need to remove every child node of the root from the tree before calling TP
        #HACK : passing an argument for max iter forces that many iterations, keep at 0 for time constraint
        return self.Tree_policy(time_limit, max_iter = 0)


    def Tree_policy(self, time_limit, max_iter = 0):
        # Keep the iteration logic, we need it to debug, don't pass an arg for submission
        iterations = 0
        while (time.perf_counter() < time_limit or iterations < max_iter):
            curr = self.select_node()
            result = self.random_rollout(curr)
            self.backpropagate(curr, winner=result)
            iterations += 1
    
        print("tree size", self.root.count_nodes())
        best_move_node = self.select_best_move()
        self.set_new_root(best_move_node)
        print(" iterations", iterations)

        return best_move_node.move

    def set_new_root(self, node):
        """Sets the chosen node (must be child of root) as the new root"""
        assert node.parent is self.root
        self.root = node
        node.parent = None

    def find_state(self, current_state) -> bool:
        """Finds state amongst root children and sets it, If we have not anticipated the move, discard current tree and start from scratch
        Returns true if state was found, false if we discard"""
        for child in self.root.children:
            if child.state == current_state:
                self.set_new_root(child)
                return True
        self.root = Node(current_state)
        return False

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
        return max(self.root.children, key=lambda c: (c.visits))

    
    def truncated_rollout(self, curr):
        #TODO : Implement and test
        """Rollout strategy with eval function after a couple of moves played. Eval function needs to account for : 
        pieces in a row
        Covered pieces
        Pieces in reserve
        """
        raise NotImplemented

    def random_rollout(self, curr):
        """plays random moves for both sides until we reach the end of a game, returns the winning player's number"""
        state = curr.state.copy()
        for i in range(100 - state.ply):
            # Bad, random move checks all available moves, very expensive
            move = random_move(state.board, state. reserves, state.player, self.rng)

            if move is None:
                return state.opponent # Player to move has no legal moves => they lost

            state.apply_move(move)
            winner = check_winner(state.board, state.opponent)
            if winner != 0:
                return winner

        endgame = check_endgame(state.board, state.reserves, state.player, state.ply)
        return endgame[1]


class Node:
    """A node representing a state in the Monte Carlo search tree."""
    C =0.7

    def __init__(self, state, parent=None, move=None):
        self.state : State = state
        self.parent  : Node = parent
        self.move = move
        self.children :list [Node]= []
        self.score = 0
        self.wins = 0
        self.visits = 0
        # TODO : Get rid of this, a new rng for every node is crazy expensive, use one for MCTS and pass maybe, or a global one. Just not a new one every time.
        self.rng = RNG

        self._winner = None
        self._over = None
        self.untried_moves = None #only generate untried moves when we need to expand the node

    @property
    def over(self) -> bool: 
        if self._over == None:
            self._over, self._winner , _ = check_endgame(self.state.board, self.state.reserves, self.state.player, self.state.ply)
        return self._over
    @property
    def winner(self) -> int:
        if self._winner == None:
            self._over, self._winner , _ = check_endgame(self.state.board, self.state.reserves, self.state.player, self.state.ply)
        return self._winner

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

    def count_nodes(self):
        if not self.children: return 1
        children = 0
        for child in self.children:
            children += child.count_nodes()
        return 1 + children


    
class State:
    def __init__(self,board, reserves, player, opponent, ply):
        self.board = board
        self.reserves = reserves
        self.player = player
        self.opponent = opponent
        self.ply = ply

    def __eq__(self, other):
        """Checks if two State objects are functionaly identical"""
        # Needed for checking if we are in the right state when changing the root
        if not isinstance(other, State):
            raise NotImplemented
        return (
            self.player == other.player
            and np.array_equal(self.board, other.board)
            and np.array_equal(self.reserves, other.reserves)
        )
    def __str__(self):
        return render_text(self.board, self.reserves)
    
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






        
            
               



        

        
