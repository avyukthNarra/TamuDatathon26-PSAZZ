# Fixture: future_leaky

Deliberately invalid. It reads history ticks at or after each query's cutoff (the future) and
shifts its forecast toward them. On the released test data no such ticks exist, so output is uniform
and it passes format checks. Appending future ticks to history changes the output, which the
perturbation checks must catch. Standard library only.
