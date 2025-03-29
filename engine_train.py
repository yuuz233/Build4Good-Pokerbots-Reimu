'''
6.9630 MIT POKERBOTS GAME ENGINE
DO NOT REMOVE, RENAME, OR EDIT THIS FILE
'''
from collections import namedtuple
from threading import Thread
from queue import Queue
import time
import math
import json
import subprocess
import socket
import eval7
import sys
import os
import random
import itertools  # Add to top
import glob

sys.path.append(os.getcwd())
from config import *

FoldAction = namedtuple('FoldAction', [])
CallAction = namedtuple('CallAction', [])
CheckAction = namedtuple('CheckAction', [])
# we coalesce BetAction and RaiseAction for convenience
RaiseAction = namedtuple('RaiseAction', ['amount'])
TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

STREET_NAMES = ['Flop', 'Turn']
DECODE = {'F': FoldAction, 'C': CallAction, 'K': CheckAction, 'R': RaiseAction}
CCARDS = lambda cards: ','.join(map(str, cards))
PCARDS = lambda cards: '[{}]'.format(' '.join(map(str, cards)))
PVALUE = lambda name, value: ', {} ({})'.format(name, value)
STATUS = lambda players: ''.join([PVALUE(p.name, p.bankroll) for p in players])

# Socket encoding scheme:
#
# T#.### the player's game clock
# P# the- player's index
# H**,** the player's hand in common format
# F a fold action in the round history
# C a call action in the round history
# K a check action in the round history
# R### a raise action in the round history
# B**,**,**,**,** the board cards in common format
# O**,** the opponent's hand in common format
# D### the player's bankroll delta from the round
# Q game over
#
# Clauses are separated by spaces
# Messages end with '\n'
# The engine expects a response of K at the end of the round as an ack,
# otherwise a response which encodes the player's action
# Action history is sent once, including the player's actions


class RoundState(namedtuple('_RoundState', ['button', 'street', 'pips', 'stacks', 'hands', 'deck', 'previous_state'])):
    '''
    Encodes the game tree for one round of poker.
    '''

    def get_delta(self, winner_index: int) -> int:
        '''Returns the delta after rules are applied.

        Args:
            winner_index (int): Index of the winning player. Must be 0 (player A),
                1 (player B), or 2 (split pot).

        Returns:
            int: The delta value after applying rules.
        '''
        assert winner_index in [0, 1, 2]

        delta = 0
        if winner_index == 2:
            # Case of split pots
            assert(self.stacks[0] == self.stacks[1]) # split pots only happen on the river + equal stacks
            delta = STARTING_STACK - self.stacks[0]
        else:
            # Case of one player winning
            if winner_index == 0:
                delta = STARTING_STACK - self.stacks[1]
            else:
                delta = self.stacks[0] - STARTING_STACK

        # if delta is not an integer, round it down or up depending on who's in position
        if abs(delta - math.floor(delta)) > 1e-6:
            delta = math.floor(delta) if self.button % 2 == 0 else math.ceil(delta)
        return int(delta)

    def showdown(self) -> TerminalState:
        '''
        Compares the players' hands and computes the final payoffs at showdown.

        Evaluates both players' hands (hole cards + community cards) and determines
        the winner. The payoff (delta) is calculated based on:
        - The winner of the hand
        - The current pot size

        Returns:
            TerminalState: A terminal state object containing:
                - List of deltas (positive for winner, negative for loser)
                - Reference to the previous game state
        
        Note:
            This method assumes both players have equal stacks when reaching showdown,
            which is enforced by an assertion.
        '''
        score0 = eval7.evaluate(self.deck.peek(4) + self.hands[0])
        score1 = eval7.evaluate(self.deck.peek(4) + self.hands[1])
        assert(self.stacks[0] == self.stacks[1])
        if score0 > score1:
            delta = self.get_delta(0)
        elif score0 < score1:
            delta = self.get_delta(1)
        else:
            # split the pot
            delta = self.get_delta(2)
        
        return TerminalState([int(delta), -int(delta)], self)

    def legal_actions(self):
        '''
        Returns a set which corresponds to the active player's legal moves.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        if continue_cost == 0:
            # we can only raise the stakes if both players can afford it
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction, FoldAction} if bets_forbidden else {CheckAction, RaiseAction, FoldAction}
        # continue_cost > 0
        # similarly, re-raising is only allowed if both players can afford it
        raises_forbidden = (continue_cost == self.stacks[active] or self.stacks[1-active] == 0)
        return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        '''
        Returns a tuple of the minimum and maximum legal raises.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        max_contribution = min(self.stacks[active], self.stacks[1-active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)

    def proceed_street(self):
        '''
        Resets the players' pips and advances the game tree to the next round of betting.
        '''
        if self.street == 4:
            return self.showdown()
        return RoundState(1, self.street + 2, [0, 0], self.stacks, self.hands, self.deck, self)

    def proceed(self, action):
        '''
        Advances the game tree by one action performed by the active player.

        Args:
            action: The action being performed. Must be one of:
                - FoldAction: Player forfeits the hand
                - CallAction: Player matches the current bet
                - CheckAction: Player passes when no bet to match
                - RaiseAction: Player increases the current bet

        Returns:
            Either:
            - RoundState: The new state after the action is performed
            - TerminalState: If the action ends the hand (e.g., fold or final call)

        Note:
            The button value is incremented after each action to track whose turn it is.
            For FoldAction, the inactive player is awarded the pot.
            For CallAction on button 0, both players post blinds.
            For CheckAction, advances to next street if both players have acted.
            For RaiseAction, updates pips and stacks based on raise amount.
        '''
        active = self.button % 2
        if isinstance(action, FoldAction):
            delta = self.get_delta((1 - active) % 2) # if active folds, the other player (1 - active) wins
            return TerminalState([delta, -delta], self)
        if isinstance(action, CallAction):
            if self.button == 0:  # sb calls bb
                return RoundState(1, 0, [BIG_BLIND] * 2, [STARTING_STACK - BIG_BLIND] * 2, self.hands, self.deck, self)
            # both players acted
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1-active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            state = RoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self)
            return state.proceed_street()
        if isinstance(action, CheckAction):
            if (self.street == 0 and self.button > 0) or self.button > 1:  # both players acted
                return self.proceed_street()
            # let opponent act
            return RoundState(self.button + 1, self.street, self.pips, self.stacks, self.hands, self.deck, self)
        # isinstance(action, RaiseAction)
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        contribution = action.amount - new_pips[active]
        new_stacks[active] -= contribution
        new_pips[active] += contribution
        return RoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self)


class Player():
    '''
    Handles subprocess and socket interactions with one player's pokerbot.
    '''

    def __init__(self, name, path, config):
        self.name = name
        self.path = path
        self.game_clock = STARTING_GAME_CLOCK
        self.bankroll = 0
        self.commands = None
        self.bot_subprocess = None
        self.socketfile = None
        self.bytes_queue = Queue()
        self.parameters = config
        self.marisa_args = config.get('marisa_args', [])

    def build(self):
        '''
        Loads the commands file and builds the pokerbot.
        '''
        try:
            with open(self.path + '/commands.json', 'r') as json_file:
                commands = json.load(json_file)
            if ('build' in commands and 'run' in commands and
                    isinstance(commands['build'], list) and
                    isinstance(commands['run'], list)):
                self.commands = commands
            else:
                print(self.name, 'commands.json missing command')
        except FileNotFoundError:
            print(self.name, 'commands.json not found - check PLAYER_PATH')
        except json.decoder.JSONDecodeError:
            print(self.name, 'commands.json misformatted')
        if self.commands is not None and len(self.commands['build']) > 0:
            try:
                proc = subprocess.run(self.commands['build'],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      cwd=self.path, timeout=BUILD_TIMEOUT, check=False)
                self.bytes_queue.put(proc.stdout)
            except subprocess.TimeoutExpired as timeout_expired:
                error_message = 'Timed out waiting for ' + self.name + ' to build'
                print(error_message)
                self.bytes_queue.put(timeout_expired.stdout)
                self.bytes_queue.put(error_message.encode())
            except (TypeError, ValueError):
                print(self.name, 'build command misformatted')
            except OSError:
                print(self.name, 'build failed - check "build" in commands.json')

    def run(self):
        '''
        Runs the pokerbot and establishes the socket connection.
        '''
        if self.commands is not None and len(self.commands['run']) > 0:
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                with server_socket:
                    server_socket.bind(('', 0))
                    server_socket.settimeout(CONNECT_TIMEOUT)
                    server_socket.listen()
                    port = server_socket.getsockname()[1]
                    proc = subprocess.Popen(
                        self.commands['run'] + [str(port)] + self.marisa_args,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        cwd=self.path)
                    self.bot_subprocess = proc
                    # function for bot listening
                    def enqueue_output(out, queue):
                        try:
                            for line in out:
                                if self.path == r"./player_chatbot":
                                    print(line.strip().decode("utf-8"))
                                else:
                                    queue.put(line)
                        except ValueError:
                            pass
                    # start a separate bot listening thread which dies with the program
                    Thread(target=enqueue_output, args=(proc.stdout, self.bytes_queue), daemon=True).start()
                    # block until we timeout or the player connects
                    client_socket, _ = server_socket.accept()
                    with client_socket:
                        if self.path == r"./player_chatbot":
                            client_socket.settimeout(PLAYER_TIMEOUT)
                        else:
                            client_socket.settimeout(CONNECT_TIMEOUT)
                        sock = client_socket.makefile('rw')
                        self.socketfile = sock
                        print(self.name, 'connected successfully')
            except (TypeError, ValueError):
                print(self.name, 'run command misformatted')
            except OSError:
                print(self.name, 'run failed - check "run" in commands.json')
            except socket.timeout:
                print('Timed out waiting for', self.name, 'to connect')

    def stop(self):
        '''
        Closes the socket connection and stops the pokerbot.
        '''
        if self.socketfile is not None:
            try:
                self.socketfile.write('Q\n')
                self.socketfile.close()
            except socket.timeout:
                print('Timed out waiting for', self.name, 'to disconnect')
            except OSError:
                print('Could not close socket connection with', self.name)
        if self.bot_subprocess is not None:
            try:
                if self.path == r"./player_chatbot":
                    outs, _ = self.bot_subprocess.communicate(timeout=PLAYER_TIMEOUT)
                else:
                    outs, _ = self.bot_subprocess.communicate(timeout=CONNECT_TIMEOUT)
                self.bytes_queue.put(outs)
            except subprocess.TimeoutExpired:
                print('Timed out waiting for', self.name, 'to quit')
                self.bot_subprocess.kill()
                outs, _ = self.bot_subprocess.communicate()
                self.bytes_queue.put(outs)
        with open(self.name + '.txt', 'wb') as log_file:
            bytes_written = 0
            for output in self.bytes_queue.queue:
                try:
                    bytes_written += log_file.write(output)
                    if bytes_written >= PLAYER_LOG_SIZE_LIMIT:
                        break
                except TypeError:
                    pass

    def query(self, round_state, player_message, game_log):
        '''
        Requests one action from the pokerbot over the socket connection.

        This method handles communication with the bot, sending the current game state
        and receiving the bot's chosen action. It enforces game clock constraints and
        validates that the received action is legal.

        Args:
            round_state (RoundState or TerminalState): The current state of the game.
            player_message (list): Messages to be sent to the player bot, including game state
                information like time remaining, player position, and cards.
            game_log (list): A list to store game events and error messages.

        Returns:
            Action: One of FoldAction, CallAction, CheckAction, or RaiseAction representing
            the bot's chosen action. If the bot fails to provide a valid action, returns:
                - CheckAction if it's a legal move
                - FoldAction if check is not legal

        Notes:
            - The game clock is decremented by the time taken to receive a response
            - Invalid or illegal actions are logged but not executed
            - Bot disconnections or timeouts result in game clock being set to 0
            - At the end of a round, only CheckAction is considered legal
        '''
        legal_actions = round_state.legal_actions() if isinstance(round_state, RoundState) else {CheckAction}
        if self.socketfile is not None and self.game_clock > 0.:
            clause = ''
            try:
                player_message[0] = 'T{:.3f}'.format(self.game_clock)
                message = ' '.join(player_message) + '\n'
                del player_message[1:]  # do not send redundant action history
                start_time = time.perf_counter()
                self.socketfile.write(message)
                self.socketfile.flush()
                clause = self.socketfile.readline().strip()
                end_time = time.perf_counter()
                if ENFORCE_GAME_CLOCK and self.path != r"./player_chatbot":
                    self.game_clock -= end_time - start_time
                if self.game_clock <= 0.:
                    raise socket.timeout
                action = DECODE[clause[0]]
                if action in legal_actions:
                    if clause[0] == 'R':
                        amount = int(clause[1:])
                        min_raise, max_raise = round_state.raise_bounds()
                        if min_raise <= amount <= max_raise:
                            return action(amount)
                    else:
                        return action()
                game_log.append(self.name + ' attempted illegal ' + action.__name__)
            except socket.timeout:
                error_message = self.name + ' ran out of time'
                game_log.append(error_message)
                print(error_message)
                self.game_clock = 0.
            except OSError:
                error_message = self.name + ' disconnected'
                game_log.append(error_message)
                print(error_message)
                self.game_clock = 0.
            except (IndexError, KeyError, ValueError):
                game_log.append(self.name + ' response misformatted: ' + str(clause))
        return CheckAction() if CheckAction in legal_actions else FoldAction()

    def _convert_parameters_to_args(self):
        """Convert parameters to command-line arguments"""
        if not self.parameters:  # Handle base bot case
            return []
        return [
            f"--aggression={self.parameters['aggression']}",
            f"--bluff_threshold={self.parameters['bluff_threshold']}",
            f"--pot_multiplier={self.parameters['pot_multiplier']}"
        ]


class Game():
    '''
    Manages logging and the high-level game procedure.
    '''

    def __init__(self):
        self.log = ['Build4Good Pokerbots - ' + PLAYER_1_NAME + ' vs ' + PLAYER_2_NAME]
        self.player_messages = [[], []]
        self.preflop_bets = {PLAYER_1_NAME: 0, PLAYER_2_NAME: 0}
        self.flop_bets = {PLAYER_1_NAME: 0, PLAYER_2_NAME: 0}
        self.turn_bets = {PLAYER_1_NAME: 0, PLAYER_2_NAME: 0}

    def log_round_state(self, players, round_state):
        '''
        Incorporates RoundState information into the game log and player messages.
        '''
        if round_state.street == 2:
            self.preflop_bets = {players[0].name: STARTING_STACK-round_state.stacks[0],
                                     players[1].name: STARTING_STACK-round_state.stacks[1]}
        elif round_state.street == 4:
            self.flop_bets = {players[0].name: STARTING_STACK-round_state.stacks[0]-self.preflop_bets[players[0].name],
                                players[1].name: STARTING_STACK-round_state.stacks[1]-self.preflop_bets[players[1].name]}
        else:
            self.turn_bets = {players[0].name: STARTING_STACK-round_state.stacks[0]-self.flop_bets[players[0].name]-self.preflop_bets[players[0].name],
                                players[1].name: STARTING_STACK-round_state.stacks[1]-self.flop_bets[players[1].name]-self.preflop_bets[players[1].name]}
            
        
        if round_state.street == 0 and round_state.button == 0:
            self.log.append('{} posts the blind of {}'.format(players[0].name, SMALL_BLIND))
            self.log.append('{} posts the blind of {}'.format(players[1].name, BIG_BLIND))
            self.log.append('{} dealt {}'.format(players[0].name, PCARDS(round_state.hands[0])))
            self.log.append('{} dealt {}'.format(players[1].name, PCARDS(round_state.hands[1])))
            self.player_messages[0] = ['T0.', 'P0', 'H' + CCARDS(round_state.hands[0])]
            self.player_messages[1] = ['T0.', 'P1', 'H' + CCARDS(round_state.hands[1])]
        elif round_state.street > 0 and round_state.button == 1:
            board = round_state.deck.peek(round_state.street)
            self.log.append(STREET_NAMES[round_state.street // 2 - 1] + ' ' + PCARDS(board) +
                            PVALUE(players[0].name, STARTING_STACK-round_state.stacks[0]) +
                            PVALUE(players[1].name, STARTING_STACK-round_state.stacks[1]))
            self.log.append(f"Current stacks: {round_state.stacks[0]}, {round_state.stacks[1]}")
            compressed_board = 'B' + CCARDS(board)
            self.player_messages[0].append(compressed_board)
            self.player_messages[1].append(compressed_board)

    def log_action(self, name, action, bet_override):
        '''
        Incorporates action information into the game log and player messages.
        '''
        if isinstance(action, FoldAction):
            phrasing = ' folds'
            code = 'F'
        elif isinstance(action, CallAction):
            phrasing = ' calls'
            code = 'C'
        elif isinstance(action, CheckAction):
            phrasing = ' checks'
            code = 'K'
        else:  # isinstance(action, RaiseAction)
            phrasing = (' bets ' if bet_override else ' raises to ') + str(action.amount)
            code = 'R' + str(action.amount)
        self.log.append(name + phrasing)
        self.player_messages[0].append(code)
        self.player_messages[1].append(code)

    def log_terminal_state(self, players, round_state):
        '''
        Incorporates TerminalState information into the game log and player messages.
        '''
        previous_state = round_state.previous_state
        if not self.log[-1].endswith(' folds'):
            self.log.append('{} shows {}'.format(players[0].name, PCARDS(previous_state.hands[0])))
            self.log.append('{} shows {}'.format(players[1].name, PCARDS(previous_state.hands[1])))
            self.player_messages[0].append('O' + CCARDS(previous_state.hands[1]))
            self.player_messages[1].append('O' + CCARDS(previous_state.hands[0]))
        self.log.append('{} awarded {}'.format(players[0].name, round_state.deltas[0]))
        self.log.append('{} awarded {}'.format(players[1].name, round_state.deltas[1]))
        self.player_messages[0].append('D' + str(round_state.deltas[0]))
        self.player_messages[1].append('D' + str(round_state.deltas[1]))

    def run_round(self, players):
        '''
        Runs one round of poker.
        '''
        deck = eval7.Deck()
        deck.shuffle()
        hands = [deck.deal(3), deck.deal(3)]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        round_state = RoundState(0, 0, pips, stacks, hands, deck, None)
        while not isinstance(round_state, TerminalState):
            self.log_round_state(players, round_state)
            active = round_state.button % 2
            player = players[active]
            action = player.query(round_state, self.player_messages[active], self.log)
            bet_override = (round_state.pips == [0, 0])
            self.log_action(player.name, action, bet_override)
            round_state = round_state.proceed(action)
        self.log_terminal_state(players, round_state)
        for player, player_message, delta in zip(players, self.player_messages, round_state.deltas):
            player.query(round_state, player_message, self.log)
            player.bankroll += delta

    def run(self):
        '''
        Runs one game of poker.
        '''
        print(f'🏆 Competition Started | Total Rounds: {NUM_ROUNDS}')
        start_time = time.time()
        
        players = [
            Player(PLAYER_1_NAME, PLAYER_1_PATH, {}),  # Empty params for base bot
            Player(PLAYER_2_NAME, PLAYER_2_PATH, {
                'aggression': 1.2,
                'bluff_threshold': 0.35,  # Fixed parameter name
                'pot_multiplier': 1.5
            })
        ]
        for player in players:
            player.build()
        for player in players:
            player.run()
        for round_num in range(1, NUM_ROUNDS + 1):
            elapsed = time.time() - start_time
            remaining = (NUM_ROUNDS - round_num) * (elapsed / round_num) if round_num > 0 else 0
            print(f'\n🌀 Round {round_num}/{NUM_ROUNDS} | Elapsed: {elapsed:.1f}s | Est. Remain: {remaining:.1f}s')
            
            self.log.append('')
            self.log.append('Round #' + str(round_num) + STATUS(players))
            self.run_round(players)
            self.log.append('Winning counts at the end of the round: ' + STATUS(players))

            # Add progress bar
            progress = round_num / NUM_ROUNDS
            bar_length = 40
            bar = '█' * int(bar_length * progress) + '-' * (bar_length - int(bar_length * progress))
            print(f'[{bar}] {progress:.0%}')

            players = players[::-1]
        self.log.append('')
        self.log.append('Final' + STATUS(players))
        for player in players:
            player.stop()
        name = GAME_LOG_FILENAME + '.txt'
        print('Writing', name)
        with open(name, 'w') as log_file:
            log_file.write('\n'.join(self.log))


class BatchTester:
    """
    Handles parameter batch testing with different bot configurations
    """
    def __init__(self):
        self.param_grid = {
            'aggression': [1.0, 1.2, 1.4],
            'bluff_threshold': [0.3, 0.35, 0.4],
            'pot_multiplier': [1.2, 1.5, 2.0]
        }
        self.results = []
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def parse_gamelog(self):
        """Extract final balances from gamelog.txt"""
        with open('gamelog.txt', 'r') as f:
            final_line = f.readlines()[-1].strip()
        parts = final_line.split(', ')
        return {p.split(' ')[0]: int(p.split('(')[1].strip(')')) for p in parts[1:]}

    def run_batch(self):
        param_combinations = list(itertools.product(*self.param_grid.values()))
        total = len(param_combinations)
        
        # Load existing results
        existing = glob.glob(f"{self.results_dir}/result_*.json")
        self.results = [json.load(open(f)) for f in existing]
        completed_params = {tuple(r['params'].values()) for r in self.results}
        
        for idx, combo in enumerate(param_combinations):
            params = dict(zip(self.param_grid.keys(), combo))
            if tuple(combo) in completed_params:
                print(f"⏩ Skipping completed param set {idx+1}/{total}")
                continue

            print(f'\n🚀 Testing Param Set {idx+1}/{total}: {params}')
            try:
                # Pass parameters via command line to Marisa
                marisa_args = [
                    f"--aggression={params['aggression']}",
                    f"--bluff_threshold={params['bluff_threshold']}", 
                    f"--pot_multiplier={params['pot_multiplier']}"
                ]
                
                # Run game with current parameters
                players = [
                    Player(PLAYER_1_NAME, "./reimu", {}),
                    Player(PLAYER_2_NAME, "./marisa", {'marisa_args': marisa_args})
                ]
                
                game = Game()
                game.run()
                
                # Save result immediately
                result = self.parse_gamelog()
                result_entry = {
                    'params': params,
                    'balance': result['Marisa'],
                    'win': result['Marisa'] > 0
                }
                self.results.append(result_entry)
                self._save_result(result_entry, idx+1)
                
            except Exception as e:
                print(f"❌ Failed param set {idx+1}: {str(e)}")
                self._save_error(params, idx+1, str(e))

    def _save_result(self, result, idx):
        filename = f"{self.results_dir}/result_{idx}.json"
        with open(filename, 'w') as f:
            json.dump(result, f)
            
    def _save_error(self, params, idx, error):
        filename = f"{self.results_dir}/error_{idx}.json"
        with open(filename, 'w') as f:
            json.dump({
                'params': params,
                'error': error,
                'timestamp': time.time()
            }, f)

    def show_results(self):
        if not self.results:
            print("❌ No results available - check results/ directory")
            return
            
        best = max(self.results, key=lambda x: x['balance'])
        print(f'\n🔥 Best Params (Balance: {best["balance"]}):')
        print(json.dumps(best['params'], indent=2))
        
        # Generate summary report
        with open('batch_summary.txt', 'w') as f:
            f.write(f"Best params: {best['params']}\n")
            f.write(f"Best balance: {best['balance']}\n")
            f.write("\nAll results:\n")
            for res in sorted(self.results, key=lambda x: -x['balance']):
                f.write(f"{res['params']} => {res['balance']}\n")


if __name__ == '__main__':
    if '--batch' in sys.argv:  # Add batch mode flag
        BatchTester().run_batch()
        BatchTester().show_results()
    else:
        Game().run()
