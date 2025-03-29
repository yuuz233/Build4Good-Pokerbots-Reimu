'''
Improved pokerbot with hand strength evaluation and opponent modeling.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import eval7
import random


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.opponent_stats = {
            'preflop': {'raises': 0, 'calls': 0, 'folds': 0},
            'flop': {'raises': 0, 'calls': 0, 'folds': 0},
            'turn': {'raises': 0, 'calls': 0, 'folds': 0}
        }
        self.hand_strengths = []
        # OPTIMIZED FROM BATCH RESULTS (1.0-1.2 aggression, 0.3-0.35 bluff)
        self.aggression_factor = 1.0
        self.aggression_base = 1.1  # Linear interpolation between 1.0 and 1.2
        self.bluff_threshold = 0.325  # Midpoint of 0.3 and 0.35
        self.bluff_base = 0.3
        self.bluff_scale = 0.5
        self.pot_odds_multiplier = 1.6  # Between best 1.2 and runner-up 2.0
        # REIMU'S PARAMETERS
        self.strength_iters = 200
 
    def _calculate_strength(self, hole, board, iters=200):
        wins = 0
        for _ in range(iters):
            # Create new deck for each simulation
            deck = eval7.Deck()
            hole_cards = [eval7.Card(card) for card in hole]
            
            # Remove known cards
            for card in hole_cards + board:
                try:
                    deck.cards.remove(card)
                except ValueError:
                    pass  # Handle community cards already removed
            
            deck.shuffle()
            
            # Deal 3 cards for opponent (3-card hold'em)
            opponent_hole = deck.deal(3)
            # Deal community cards (total 5 including existing board)
            remaining_community = deck.deal(5 - len(board))
            
            our_hand = hole_cards + board + remaining_community
            opp_hand = opponent_hole + board + remaining_community
            
            our_value = eval7.evaluate(our_hand[:5])
            opp_value = eval7.evaluate(opp_hand[:5])
            
            if our_value > opp_value:
                wins += 1
                
        return wins / iters

    def _update_stats(self, street, action):
        action_type = 'raises' if isinstance(action, RaiseAction) else \
                    'calls' if isinstance(action, CallAction) else \
                    'folds' if isinstance(action, FoldAction) else 'checks'
        if action_type in self.opponent_stats[street]:
            self.opponent_stats[street][action_type] += 1

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        street = 'preflop' if round_state.street == 0 else \
                'flop' if round_state.street == 3 else 'turn'
        if round_state.button != active:  # Track opponent's previous action
            if round_state.pips[1-active] > round_state.pips[active]:
                self._update_stats(street, RaiseAction(0))
            else:
                self._update_stats(street, CheckAction())

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        # Add connection cleanup logic
        try:
            # Reset any network-related state
            self.opponent_stats = {stage: {'raises':0, 'calls':0, 'folds':0} 
                                 for stage in ['preflop', 'flop', 'turn']}
            self.aggression_factor = 1.0
            # Mimic all_in_bot's simple round termination
            if terminal_state.previous_state.street == 5:  # River reached
                return CheckAction()
        except Exception as e:
            pass  # Silent cleanup like all_in_bot

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()
        street = 'preflop' if round_state.street == 0 else \
                'flop' if round_state.street == 3 else 'turn'
                
        my_cards = round_state.hands[active]
        board = [str(card) for card in round_state.deck[:round_state.street]]
        strength = self._calculate_strength(my_cards, [eval7.Card(c) for c in board])
        
        # Calculate opponent fold probability
        total_actions = sum(self.opponent_stats[street].values())
        fold_prob = self.opponent_stats[street]['folds'] / total_actions if total_actions > 0 else 0.3
        
        # Dynamic aggression adjustment
        self.aggression_factor = 1.0 + (0.5 - fold_prob) * 2
        
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            pot_size = sum(round_state.pips)
            
            # Adjust raise size based on strength and opponent tendencies
            raise_size = min_raise + int((max_raise - min_raise) * strength * self.aggression_factor)
            raise_size = max(min_raise, min(raise_size, max_raise))
            
            # Bluff with probability based on opponent fold tendency
            if strength < self.bluff_threshold and random.random() < (self.bluff_base + fold_prob*self.bluff_scale):
                return RaiseAction(raise_size)
                
            if strength > 0.6:
                return RaiseAction(raise_size)
                
        if CallAction in legal_actions:
            # Calculate pot odds
            continue_cost = round_state.pips[1-active] - round_state.pips[active]
            pot_odds = continue_cost / (sum(round_state.pips) + continue_cost)
            
            if strength > pot_odds * self.pot_odds_multiplier:  # Positive expected value
                return CallAction()
                
        if CheckAction in legal_actions and strength > 0.4:
            return CheckAction()
            
        return FoldAction() if FoldAction in legal_actions else CheckAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
