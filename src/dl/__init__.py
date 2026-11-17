"""Phase 2 — the neural acoustic model.

Phase 1 synthesised speech from 70 averaged phoneme templates. Phase 2 replaces
that lookup table with a trained generator, keeping the inputs (phonemes and
durations) and the frozen HiFi-GAN vocoder identical so the results stay
directly comparable to A1 and A2.

See docs/PHASE2.md for the plan and the division of work.
"""
