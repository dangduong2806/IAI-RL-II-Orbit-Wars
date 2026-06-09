# Hybrid Phantom Heuristic

Mini project for the Orbit Wars V1 hybrid phantom heuristic.

This folder keeps only the source code and lightweight runtime requirements:

- `source/`: source snapshot for the heuristic agent.
- `package/package_hybrid_phantom/`: compact package entrypoint used for submission-style runs.
- `requirements.txt`: Python dependencies.

The variant is heuristic-only. It uses phantom enemy-move prediction together with geometry-aware tie-breaking to choose safer tactical actions.
