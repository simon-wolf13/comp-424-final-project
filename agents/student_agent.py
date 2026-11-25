"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time
import math

EXPOLORATION_C = math.sqrt(2) #exploration constant can be tuned to make the agent better
SAFE_CHECK = 0.4
TIME_BUDGET = 1.7
SEARCH_DEPTH = 6


class BoardNode:

    def __init__(self, board,reserves, moved_last, move_next, move, parent, ply):
        self.board = board
        self.reserves = reserves
        self.parent = parent
        self.moved_last = moved_last
        self.move_next = move_next
        self.move = move
        self.children = []
        self.untried = list(get_valid_moves(board, reserves, move_next))
        self.vists = 0
        self.wins = 0.0
        self.ply  = ply






@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.rng = np.random.default.rng()
        self.turns_played = 0



    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.
        ply = 2*self.turns_played + (player-1)
        self.turns_played += 1
        try:
            mv = MCTS_Search(board, reserves, player, opponent, ply, start)
            if mv is not None:
                return mv
        except Exception:
            pass
        return random_move(board, reserves, player)

    def MCTS_Search(self, board, reserves, player, opponent, ply, start):
        moves = get_valid_moves(board, reserves, player)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0] #only one valid option

        candidate_moves = []
        for mv in moves:
            b, r = copy_state(board, reserves)
            execute_move(b, r, player, mv)
            is_winner = check_winner(b,player)
            if is_winner ==player:
                return mv
            if is_winner==opponent:
                continue

            candidate_moves.append((mv, b,r))

        if len(candidate_moves) ==0:
            return moves[0]

        #Immediately remove any moves that lead to opponent win in their next turn

        safe_move = [] 
        for i, (m,b,r) in enumerate(candidate_moves):
            if time.perf_counter() - start > SAFE_CHECK:
                safe_move.extend(candidate_moves[i:])
                break
            if not self.opponent_wins_next(b, r, opponent):
                safe_move.append((m,b,r))

        if len(safe_move) ==0:
            return candidate_moves[0]

        root = BoardNode(board, reserves, opponent, player, None, None, ply)
        root.untried = []

        for mv, b, r in enumerate(safe_move):
            root.children(b, r, player, opponent, mv, root, ply+1)

        end_time = start + TIME_BUDGET
        while time.perf_counter() < end_time:
            self.iterate(root)

        best = max(root.children, key=lambda ch: ch.n)
        return best.move


   




        
            
               

    def ucb_child(self, node):
        vists = math.log(max(node.visits), 1)
        best_node, best_v = None, -1.0
        for ch in node.children:
            if ch.n == 0: #ln(0) is undefined, so we just return the first unvistied child
                return ch
            value = ch.wins / ch.n + EXPOLORATION_C * math.sqrt(math.log(vists)/ch.n)
            if value > best_v:
                best_node = ch
                best_v = value
        return best_node

    def opponent_wins_next(self,board, reserves, opponent):
        for opp_move in get_valid_moves(board, reserves, opponent):
            b, r = copy_state(board, reserves)
            execute_move(b,r, opponent, opp_move)
            winner = check_winner(b, opponent)
            if winner == opponent:
                return True
        return False


        

        
