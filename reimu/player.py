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
        self.aggression_factor = 1.0  # Dynamic aggression adjustment

    def _calculate_strength(self, hole, board, iters=200):
        deck = eval7.Deck()
        hole_cards = [eval7.Card(card) for card in hole]
        
        for card in hole_cards + board:
            deck.cards.remove(card)
            
        wins = 0
        for _ in range(iters):
            deck.shuffle()
            opponent_hole = deck.deal(3)
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
        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        pass

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
            if strength < 0.4 and random.random() < (0.3 + fold_prob*0.5):
                return RaiseAction(raise_size)
                
            if strength > 0.6:
                return RaiseAction(raise_size)
                
        if CallAction in legal_actions:
            # Calculate pot odds
            continue_cost = round_state.pips[1-active] - round_state.pips[active]
            pot_odds = continue_cost / (sum(round_state.pips) + continue_cost)
            
            if strength > pot_odds * 1.2:  # Positive expected value
                return CallAction()
                
        if CheckAction in legal_actions and strength > 0.4:
            return CheckAction()
            
        return FoldAction() if FoldAction in legal_actions else CheckAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
