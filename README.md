# QAST Music AVQA

This repository archives the QAST-EHR experiment reported at 77.38% test accuracy (seed 713, 7064/9129), together with seeds 123 and 456, component and auxiliary-loss ablations, code snapshots, logs, summary tables, and the three best-validation checkpoints.

- Protocol: random initialization, 15 training epochs, best checkpoint selected on validation, official train/validation/test split.
- Three-seed overall test accuracy: 77.27 ± 0.18% (sample standard deviation).
- Three-seed AV-Temporal accuracy: 70.92 ± 0.49%.
- The VGGish, CLIP-L/14, and ToMe feature archives are **not** included. The original configurations contain local paths that must be adapted to your environment.
- Checkpoints are stored with Git LFS. Install Git LFS before cloning if you need the `.pt` files.

See [README_归档说明.md](README_归档说明.md) for the inventory and [MANIFEST.json](MANIFEST.json) for checkpoint provenance and SHA-256 hashes.
