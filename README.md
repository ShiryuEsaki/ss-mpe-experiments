# SS-MPE experiments

Training and frame-level multi-pitch estimation evaluation code for the
proposed method.

This repository contains code only. Datasets, checkpoints, audio files, and
experimental results are not included.

## Installation

This implementation extends the public
[ss-nt-mpe-rc](https://github.com/DeReKPIgg/ss-nt-mpe-rc) codebase at commit
`abc0351cbd9bc999bfc52f5a8ff603f59d77473e`.

```bash
bash scripts/install_upstream.sh
python -m pip install -r requirements.txt
```

The code was tested with Python 3.10. Install a PyTorch build compatible with
your CUDA environment separately if necessary.

## Method configuration

The released configuration uses:

- harmonic orders H1–4;
- harmonic power exponent q = 0;
- harmonic aggregation order r = 1;
- strict-positive loss weight 40; and
- pair-relation loss disabled.

Training runs for 30,000 steps. The checkpoint with the minimum audio-only
validation loss is retained as `models/best.pt`.

## Usage

Training:

```bash
python train.py \
  --dataset-root /path/to/URMP/Dataset \
  --output-dir runs/example \
  --seed 4200 \
  --split-seed 20260921 \
  --gpu 0
```

Evaluation:

```bash
python test.py \
  --run-dir runs/example \
  --target urmp \
  --gpu 0
```

Use `python train.py --help`, `python test.py --help`, and
`python evaluate.py --help` for the complete options.

## Evaluation protocol

The detection threshold is selected by maximizing macro F1 on the validation
subset. The selected threshold is then fixed for a single test-set evaluation.
Test data are not used for checkpoint or threshold selection.

Precision, recall, and F1 are macro-averaged independently.

## License

No license is assigned to this repository.
