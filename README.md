# Build4Good Pokerbots

Welcome to the Build4Good 2025 Pokerbot Challenge. Today you are tasked with building a bot to play a variant of poker.

## Poker Variant Description

Today we are playing B4G Hold'em, which is based on the popular [No-Limit Texas Hold'em](https://redchippoker.com/beginners-guide-to-no-limit-holdem/), however, the rounds of betting are different.

Player will initially be given 3 cards (instead of 2 as in Hold'em). After the first round of betting, 2 cards will be dealt on the flop. After another round of betting, the final 2 cards will be dealt, followed by a final round of betting. The standard [Texas Hold'em hand rules](https://www.cardplayer.com/rules-of-poker/hand-rankings) will be used to decide the winner. Noteably, this is one less round of betting than in Texas Hold'em.

## Pokerbot Tournament

A match of B4G Hold'em consists of 5,000 rounds played between two players. In every round, each player is allocated a stack of 500 chips before the cards are dealt. The big blind (BB) and small blind (SB) will be 10 and 5 chips, respectively, and will alternate each hand. They players chips reset back to 500 after each round. The change in a player's stack at the end of the round is used to update the player's bankroll, which starts at 0. The player with the highest cumulative bankroll after the last round is played wins the match.

Your poker bot will have a **time limit of 180 seconds per match**. This time limit does not reset after each hand. Once your bot runs out of time, you will automatically fold all remaining hands. 

At the end of Build4Good, you will submit the code for your pokerbot and all competetors' bot's will compete in a March-Madness style single-elimination bracket.

## Code Structure

The engine.py file contains the main code to simulate matches between bots, you should not edit this file. If you want to change any parameters of the matches, you can edit config.py

Each bot is implemented in a seperate folder, python_skeleton/ gives an example of how these folders need to be structured. Inside the folder for each bot, you should have the skeleton/ folder, which contains basic code required to run your bot. **DO NOT** edit any of the files in skeleton/, but you do need to copy the skeleton/ folder into the folder for any new bots you create.

The only code you need to edit is the player.py class inside the bot folder. You will need to implement the methods __init__, handle_new_round, handle_round_over, and get_action. You can store variables that will be kept between rounds as members of your class. To run your bot, edit the path in config.py. You may create additional files if needed, but your final submission must be less than 10 MB in size.

There is a special bot, player_chatbot, that is provided which allows you to play against your own bot using a command line interface. This can be used for debugging purposes. 

You can use numpy/numba, but any other external Python libraries **are not allowed**!

If your pokerbot attempts to tamper with the game engine/judging system in any way, your team will be **immidiately disqualified**.  

## Dependencies
 - python>=3.5
 - eval7 (pip install eval7)
 - openai (optional, pip install openai)

## Submission

More info to come later!

## Acknowledgements

The Build4Good 2025 Pokerbot Challenge is sponsored by Susquehanna International Group and Hudson River Trading, in addition to all of the general Build4Good sponsors.

The code for the Build4Good Pokerbot Challenge was forked from the [MIT Pokerbots codebase](https://github.com/mitpokerbots/engine-2025/tree/master).
