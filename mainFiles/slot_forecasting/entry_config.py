"""Switches for the packaged entry. The ablation runner overrides these with flags.

The size candidate is enabled, but a train-only gate chooses it only when its
episode-cross-fitted conditional-size Brier loss beats the PairMixer's own size
head under the fixed criterion. The gate is recomputed on the supplied kit's
training data; its decision is not frozen into the archive.
"""

USE_SIZE = True
USE_SIZE_GATE = True
USE_FLIP = True
USE_ROUND2 = True
