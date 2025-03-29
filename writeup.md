# Reimu: A Monte Carlo-Based Adaptive Poker Bot

## Elevator Pitch
Reimu is an intelligent poker bot that combines Monte Carlo simulation with opponent modeling to create a formidable player that adapts its strategy based on its opponents' tendencies. By dynamically balancing aggression, calculating precise hand strengths, and making mathematically sound decisions, Reimu leverages both the science of poker probability and the art of reading opponents.

## Methodology

### Hand Strength Evaluation
At the core of Reimu's decision-making process is a robust Monte Carlo simulation engine that evaluates hand strength in the unique 3-card hold'em format. For each decision point:

1. The bot simulates hundreds of possible game outcomes by:
   - Creating fresh deck instances for each simulation
   - Removing known cards (hole cards and visible community cards)
   - Randomly dealing opponent cards and remaining community cards
   - Evaluating and comparing hand rankings

This approach allows Reimu to calculate a precise winning probability against random opponent holdings, accounting for the specific game state at each decision point.

### Opponent Modeling
Reimu tracks opponent actions across different streets (preflop, flop, turn) and categorizes them into:
- Raises (aggressive actions)
- Calls (passive actions)
- Folds (defensive actions)

This data builds a behavioral profile that influences Reimu's strategy. For example, against opponents who fold frequently, Reimu increases its bluffing frequency and aggression factor.

### Adaptive Aggression
Rather than employing a static strategy, Reimu features a dynamic aggression adjustment system:
```
aggression_factor = base_aggression + (0.5 - fold_probability) * 2
```

This formula allows Reimu to become more aggressive against tight players and more cautious against loose players, creating a constantly evolving meta-game.

### Decision Framework
Reimu's action selection balances several key factors:
- **Hand strength**: Determined through Monte Carlo simulation
- **Pot odds**: Ensuring mathematically profitable calls
- **Bluffing opportunities**: Strategically incorporated based on opponent tendencies
- **Raise sizing**: Proportional to hand strength and tailored to maximize expected value

### Training Methodology
The parameter optimization process for Reimu involves:
1. Grid search across key hyperparameters (aggression factor, bluff threshold, pot multiplier)
2. Simulating hundreds of hands for each parameter configuration
3. Tracking performance metrics across different opponent types
4. Selecting optimal parameters that maximize expected value

This evolutionary approach allows Reimu to continuously refine its strategy and adapt to changing poker environments.

## Conclusion
Reimu represents a sophisticated approach to poker AI, blending rigorous mathematical analysis with adaptive psychological modeling. Its ability to calculate precise equity while adapting to opponent tendencies makes it both theoretically sound and practically effective across various poker scenarios.
