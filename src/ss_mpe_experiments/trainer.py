# STRICT_UPPER_POSITIVE_TRAINING_INTEGRATION_V1
"""
MPE-only URMP training script with seed/run-directory overrides.

Generated from the official Timbre-Drill train.py.
Main changes:
- Phase1=True
- Phase2=False
- Phase3=False
- URMP-only dataset construction
- annotation-free train/validation manifests supplied by ``train.py``
- test_splits are excluded from train_splits
- NSynth/MAPS/MusicNet dataset objects are not constructed
"""

from timbre_drill.datasets.MixedMultiPitch import URMP as URMP_Mixtures, Bach10 as Bach10_Mixtures, Su, TRIOS, MAPS, MusicNet
from timbre_drill.datasets.SoloMultiPitch import GuitarSet, URMP as URMP_Stems
from timbre_drill.datasets import ComboDataset

from timbre_drill.datasets.SoloMultiPitch import NSynth

from timbre_drill.framework import *
from timbre_drill.framework.objectives import *
from timbre_drill.utils import *
from evaluate import evaluate
from evaluate_note import evaluate_note
from evaluate_onset import evaluate_onset

from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from sacred import Experiment
from tqdm import tqdm

import numpy as np
import warnings
import librosa
import torch

from .harmonic_weighting import (
    get_effective_power_weights,
    make_harmonic_amplitude_weights,
)
from .diagnostics import compute_harmonic_diagnostics
from .pair_relation_loss import (
    build_pair_negative_confidence,
    pair_negative_loss,
    summarize_pair_negative,
)
from .strict_positive_loss import (
    build_strict_upper_positive,
    strict_positive_loss,
    summarize_strict_upper_positive,
)
import torch.nn.functional as F
import atexit
import math
import numbers
import os
import json

import matplotlib.pyplot as plt
from .harmonic_aggregation import install_from_environment as install_phase_k_aggregation
from .validation_loss import FixedAudioClips, RandomAudioClips

class NamedAudioOnlyDataset:
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]
    def name(self):
        return self.dataset_name

PHASE_K_AGGREGATION_STATUS = install_phase_k_aggregation()
import mir_eval.multipitch
mir_eval.multipitch.MAX_FREQ = 8000.0


WANDB_TRUE_VALUES = {"1", "true", "yes", "on"}
WANDB_ALLOWED_MODES = {"online", "offline", "disabled"}
WANDB_DEFAULT_PROJECT = "renew-ssmpe-harmonic"
WANDB_APPROVED_ENTITY = "harmonic-ssmpe"


def _wandb_scalar(value):
    """Return a plain numeric scalar, or None for unsupported values."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool):
        return value
    return None


class OptionalWandbTracker:
    """Small failure-tolerant wrapper around wandb.log()/wandb.finish()."""

    def __init__(self, wandb_module=None, run=None, log_interval=10):
        self._wandb = wandb_module
        self._run = run
        self._log_interval = max(1, int(log_interval))
        self._logging_enabled = run is not None
        self._finished = False

        if self._logging_enabled:
            # Explicit finish() is called on normal completion. This fallback also
            # closes the run when training exits through an uncaught exception.
            atexit.register(self.finish)

    @property
    def enabled(self):
        return self._logging_enabled and not self._finished

    def should_log(self, step, force=False):
        return self.enabled and (force or step % self._log_interval == 0)

    def log(self, metrics, *, step, force=False):
        if not self.should_log(step, force=force):
            return

        payload = {}
        for key, value in metrics.items():
            scalar = _wandb_scalar(value)
            if scalar is not None:
                payload[str(key)] = scalar

        if not payload:
            return

        # W&B's internal history step advances on every log call. A separate
        # global_step axis lets training and validation share checkpoint steps.
        payload.setdefault("global_step", float(step))

        try:
            self._wandb.log(payload)
        except Exception as exc:
            # W&B must never stop the training loop.
            print(
                "WARNING: wandb.log() failed; disabling further W&B logging: "
                f"{type(exc).__name__}: {exc}"
            )
            self._logging_enabled = False

    def log_validation(self, validation_results, *, step, epoch, learning_rates):
        payload = {
            "epoch": epoch,
            "global_step": step,
        }

        for name, value in learning_rates.items():
            payload[f"learning_rate/{name}"] = value

        for dataset_name, dataset_results in validation_results.items():
            if not isinstance(dataset_results, dict):
                continue
            for metric_name, value in dataset_results.items():
                payload[
                    f"validation/{dataset_name}/{metric_name}"
                ] = value

        self.log(payload, step=step, force=True)

    def set_summary(self, values):
        if self._run is None or self._finished:
            return
        try:
            for key, value in values.items():
                scalar = _wandb_scalar(value)
                if scalar is not None:
                    self._run.summary[str(key)] = scalar
        except Exception as exc:
            print(
                "WARNING: failed to update W&B summary; training will continue: "
                f"{type(exc).__name__}: {exc}"
            )

    def finish(self, exit_code=None):
        if self._wandb is None or self._finished:
            return
        self._finished = True
        try:
            if exit_code is None:
                self._wandb.finish()
            else:
                self._wandb.finish(exit_code=exit_code)
        except Exception as exc:
            print(
                "WARNING: wandb.finish() failed; ignoring the W&B error: "
                f"{type(exc).__name__}: {exc}"
            )


def initialize_wandb_tracking(*, config, run_name, run_dir):
    """
    Initialize W&B only when explicitly enabled.

    WANDB_ENTITY is mandatory. If it is absent, wandb.init() is not called, so
    a run can never be sent to an implicit Team or Organization.
    """
    enabled = os.environ.get("WANDB_ENABLED", "0").strip().lower()
    if enabled not in WANDB_TRUE_VALUES:
        print("W&B tracking: disabled (set WANDB_ENABLED=1 for a future run)")
        return OptionalWandbTracker()

    project = os.environ.get(
        "WANDB_PROJECT",
        WANDB_DEFAULT_PROJECT,
    ).strip()
    entity = os.environ.get(
        "WANDB_ENTITY",
        WANDB_APPROVED_ENTITY,
    ).strip()
    mode = os.environ.get("WANDB_MODE", "online").strip().lower()

    if not project:
        print("WARNING: WANDB_PROJECT is empty; W&B tracking is disabled.")
        return OptionalWandbTracker()
    if not entity:
        print(
            "WARNING: WANDB_ENTITY is not set; wandb.init() will not be called. "
            "Training will continue without W&B."
        )
        return OptionalWandbTracker()
    if entity != WANDB_APPROVED_ENTITY:
        print(
            "WARNING: refusing to initialize W&B for an unapproved entity: "
            f"{entity!r}. Expected approved Team entity "
            f"{WANDB_APPROVED_ENTITY!r}. Training will continue without W&B."
        )
        return OptionalWandbTracker()
    if mode not in WANDB_ALLOWED_MODES:
        print(
            f"WARNING: unsupported WANDB_MODE={mode!r}; W&B tracking is disabled."
        )
        return OptionalWandbTracker()

    try:
        log_interval = max(
            1,
            int(os.environ.get("WANDB_LOG_INTERVAL", "10")),
        )
    except ValueError:
        print("WARNING: invalid WANDB_LOG_INTERVAL; using 10 steps.")
        log_interval = 10

    # Do not copy terminal output into W&B. Metrics are sent only by wandb.log().
    os.environ.setdefault("WANDB_CONSOLE", "off")

    try:
        import wandb
    except Exception as exc:
        print(
            "WARNING: W&B is enabled but the wandb package is unavailable; "
            "training will continue without W&B: "
            f"{type(exc).__name__}: {exc}"
        )
        return OptionalWandbTracker()

    wandb_dir = os.environ.get(
        "WANDB_DIR",
        os.path.join(run_dir, "wandb"),
    )
    wandb_config = dict(config)
    wandb_config["wandb_log_interval_steps"] = log_interval

    try:
        os.makedirs(wandb_dir, exist_ok=True)
        run = wandb.init(
            project=project,
            entity=entity,
            config=wandb_config,
            name=run_name,
            dir=wandb_dir,
            mode=mode,
            save_code=False,
        )
    except Exception as exc:
        print(
            "WARNING: wandb.init() failed; training will continue without W&B: "
            f"{type(exc).__name__}: {exc}"
        )
        try:
            wandb.finish(exit_code=1)
        except Exception:
            pass
        return OptionalWandbTracker()

    try:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
    except Exception as exc:
        print(
            "WARNING: failed to configure the W&B global_step axis; "
            "metric logging will still continue: "
            f"{type(exc).__name__}: {exc}"
        )

    print(
        "W&B tracking: enabled "
        f"(project={project}, entity={entity}, mode={mode}, "
        f"log_interval={log_interval})"
    )
    return OptionalWandbTracker(
        wandb_module=wandb,
        run=run,
        log_interval=log_interval,
    )

######################
## TRAINING SETTING ##
######################

DEBUG = 0 # (0 - off | 1 - on)
CONFIG = 0 # (0 - MCT server | 1 - PC)
EX_NAME = '_'.join(['renew_ssmpe_mpe_urmp_repro'])

ex = Experiment('Train a model to perform NT with self-supervised objectives only')

@ex.config
def config():
    TRAIN_FROM_SCRATCH = True
    ##############################
    ## TRAINING HYPERPARAMETERS ##
    ##############################

    Phase1 = True
    Phase2 = False
    Phase3 = False

    pitch_set = 'URMP' # Nsynth
    onset_set = 'URMP' # fixed for MPE-only URMP reproduction

    set_dict = {'Nsynth': NSynth.name(), 'MAPS': MAPS.name(), 'URMP': URMP_Mixtures.name(), 'MusicNet': MusicNet.name()}

    # Specify a checkpoint from which to resume training (None to disable)

    checkpoint_path = None
    # checkpoint_path = None

    # Maximum number of training epochs to conduct
    max_epochs = 1 if CONFIG else 100

    # Stop Phase1 training after this many optimizer steps.
    # The paper reports 30,000 training steps.
    max_steps = 30000

    # Number of iterations between checkpoints
    checkpoint_interval = 1 if CONFIG else 300

    # Number of samples to gather for a batch
    batch_size = 1 if DEBUG else 20 # 20

    if ~DEBUG and Phase3:
        batch_size = 16

    # Number of seconds of audio per sample
    n_secs = 4

    # Initial learning rate for encoder
    learning_rate_encoder = 1e-4

    # Initial learning rate for decoder
    learning_rate_decoder = learning_rate_encoder

    # Initial learning rate for note & onset
    learning_rate_onset_encoder = learning_rate_encoder * 5
    learning_rate_onset_decoder = learning_rate_encoder * 5

    # Group together both learning rates
    learning_rates = [learning_rate_encoder, learning_rate_decoder, learning_rate_onset_encoder, learning_rate_onset_decoder]

    # Scaling factors for each loss term
    multipliers = {
        'support_p' : 1.2,
        'harmonic_p' : 1.5,
        'sparsity_p' : 1.5,
        'timbre_p' : 1,
        'geometric_p' : 1,
        'time_sim': 0,

        'sparsity_n': 5,
        'note_harmonic': 0.1,
        'note_support': 1,
        'note_compress': 5,
        'frequency_distance': 0,

        'bce_o' : 5,
        'sparsity_t_o' : 1,
        'sparsity_f_o' : 0,
        'timbre_o' : 1,
        'geometric_o' : 1,

        'reconstruction' : 1,
        'reconstruction_o' : 1,

        'supervised' : 0
    }

    # Number of epochs spanning warmup phase (0 to disable)
    n_epochs_warmup = 0

    # Set validation dataset to compare for learning rate decay and early stopping

    validation_criteria_set = set_dict[pitch_set]

    # Set validation metric to compare for learning rate decay and early stopping
    validation_criteria_metric = 'loss/total'

    # Select whether the validation criteria should be maximized or minimized
    validation_criteria_maximize = False # (False - minimize | True - maximize)

    # Late starting point (0 to disable)
    n_epochs_late_start = 0

    # Number of epochs without improvement before reducing learning rate (0 to disable)
    n_epochs_decay = 2

    # Number of epochs before starting epoch counter for learning rate decay
    n_epochs_cooldown = 0

    # Number of epochs without improvement before early stopping (None to disable)
    n_epochs_early_stop = None

    # IDs of the GPUs to use, if available
    gpu_ids = [0] if DEBUG else [0]

    # Random seed for this experiment
    seed = 4200

    ########################
    ## FEATURE EXTRACTION ##
    ########################

    '''sliCQ'''
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 256

    # # First center frequency (MIDI) of geometric progression
    fmin = librosa.note_to_midi('A0')

    # Number of bins in a single octave
    bins_per_octave = 60 # 5 bins per semitone

    # Number of octaves the CQT should span
    n_octaves = 8

    '''hcqt'''
    harmonics = [0.5, 1, 2, 3, 4, 5]

    '''onset'''
    onset_bins_per_semitone = 5

    '''CFP_HMLC'''
    CFP_HMLC_win = 7412
    CFP_HMLC_fr = 1.0
    CFP_HMLC_g = np.array([0.2, 0.6, 0.9, 0.9, 1.0])
    CFP_HMLC_bov = 2
    CFP_HMLC_Har = 1
    CFP_mode = False

    ############
    ## OTHERS ##
    ############

    # Number of threads to use for data loading
    n_workers = 0 if DEBUG else 16 * len(gpu_ids)

    # Top-level directory under which to save all experiment files
    root_dir = os.path.join('generated', 'experiments', EX_NAME)

    # Keep Sacred outputs isolated for simultaneous experiments.
    env_run_dir_for_observer = os.environ.get("BASELINE_RUN_DIR")
    if env_run_dir_for_observer:
        root_dir = env_run_dir_for_observer

    # Create the root directory
    os.makedirs(root_dir, exist_ok=True)

    if DEBUG:
        # Print a warning message indicating debug mode is active
        warnings.warn('Running in DEBUG mode...', RuntimeWarning)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def train_model(TRAIN_FROM_SCRATCH, Phase1, Phase2, Phase3, pitch_set, onset_set, set_dict,
                checkpoint_path, max_epochs, max_steps, checkpoint_interval, batch_size, n_secs, learning_rates,
                multipliers, n_epochs_warmup, validation_criteria_set, validation_criteria_metric,
                validation_criteria_maximize, n_epochs_late_start, n_epochs_decay, n_epochs_cooldown,
                n_epochs_early_stop, gpu_ids, seed, sample_rate, hop_length, fmin, bins_per_octave,
                n_octaves, harmonics, onset_bins_per_semitone,
                CFP_mode,
                CFP_HMLC_win, CFP_HMLC_fr, CFP_HMLC_g, CFP_HMLC_bov, CFP_HMLC_Har,
                n_workers, root_dir):
    # Discard read-only types
    learning_rates = list(learning_rates)
    multipliers = dict(multipliers)
    harmonics = list(harmonics)
    gpu_ids = list(gpu_ids)

    if not TRAIN_FROM_SCRATCH or not Phase1 or Phase2 or Phase3:
        raise RuntimeError(
            "Final runs require train-from-scratch MPE-only Phase1"
        )
    if checkpoint_path is not None:
        raise RuntimeError("Final runs may not initialize from a checkpoint")

    # Baseline seed-control experiment overrides.
    env_run_dir = os.environ.get("BASELINE_RUN_DIR")
    if env_run_dir:
        root_dir = env_run_dir
        os.makedirs(root_dir, exist_ok=True)

    env_max_steps = os.environ.get("BASELINE_MAX_STEPS")
    if env_max_steps is not None:
        max_steps = int(env_max_steps)

    env_checkpoint_interval = os.environ.get("BASELINE_CHECKPOINT_INTERVAL")
    if env_checkpoint_interval is not None:
        checkpoint_interval = int(env_checkpoint_interval)

    smoke_test = os.environ.get("SS_MPE_EXPERIMENTS_SMOKE_TEST", "0") == "1"
    if smoke_test:
        # The smoke manifest intentionally contains one training recording.
        # Keep one complete batch and avoid worker-startup noise; full runs retain
        # the registered batch size of 20 and the normal worker configuration.
        batch_size = 1
        n_workers = 0

    env_seed = os.environ.get("BASELINE_SEED")
    if env_seed is not None:
        seed = int(env_seed)

    # The final experiment registry resolves every loss coefficient explicitly.
    # This is also how Phase R disables exactly one term at a time.
    loss_weight_env = {
        "support_p": "LOSS_WEIGHT_SUPPORT",
        "harmonic_p": "LOSS_WEIGHT_HARMONIC",
        "sparsity_p": "LOSS_WEIGHT_SPARSITY",
        "timbre_p": "LOSS_WEIGHT_TIMBRE",
        "geometric_p": "LOSS_WEIGHT_GEOMETRIC",
        "reconstruction": "LOSS_WEIGHT_RECONSTRUCTION",
        "time_sim": "LOSS_WEIGHT_TIME_SIMILARITY",
    }
    for loss_key, variable in loss_weight_env.items():
        if variable not in os.environ:
            raise RuntimeError(f"missing required final loss setting: {variable}")
        multipliers[loss_key] = float(os.environ[variable])

    print("============================================================")
    print("Baseline seed-control experiment settings")
    print("============================================================")
    print("root_dir:", root_dir)
    print("max_steps:", max_steps)
    print("seed:", seed)
    print("============================================================")

    # Seed everything with the same seed
    seed_everything(seed)

    # Initialize the primary PyTorch device
    device = torch.device(f'cuda:{gpu_ids[0]}'
                          if torch.cuda.is_available() else 'cpu')

    ########################
    ## FEATURE EXTRACTION ##
    ########################

    n_bins = n_octaves * bins_per_octave

    # Harmonic weighting for the positive pseudo-label.
    #
    # SS_NT.get_all_features() computes:
    #     sum((features_am * harmonic_weights) ** 2)
    #
    # Therefore, HARMONIC_POWER_EXPONENT specifies the exponent
    # applied to the effective power weights, not directly to the
    # amplitude weights.
    harmonic_max_order = int(
        os.environ.get("HARMONIC_MAX_ORDER", "5")
    )
    harmonic_power_exponent = float(
        os.environ.get("HARMONIC_POWER_EXPONENT", "4.0")
    )
    harmonic_weight_normalization = os.environ.get(
        "HARMONIC_WEIGHT_NORMALIZATION",
        "legacy_amp_l1",
    )

    harmonic_weights = make_harmonic_amplitude_weights(
        harmonics,
        max_order=harmonic_max_order,
        power_exponent=harmonic_power_exponent,
        normalization=harmonic_weight_normalization,
        device=device,
    )

    effective_power_weights = get_effective_power_weights(
        harmonic_weights
    )

    print("===== Harmonic pseudo-label configuration =====")
    expected_aggregation_order = float(
        os.environ.get("HARMONIC_AGGREGATION_ORDER", "2.0")
    )
    expected_aggregation_status = (
        "native_rms_unchanged"
        if abs(expected_aggregation_order - 2.0) < 1.0e-12
        else f"generalized_r={expected_aggregation_order}"
    )
    if PHASE_K_AGGREGATION_STATUS != expected_aggregation_status:
        raise RuntimeError(
            "harmonic aggregation installation mismatch: "
            f"{PHASE_K_AGGREGATION_STATUS} != {expected_aggregation_status}"
        )
    print(f"harmonic_aggregation_order={expected_aggregation_order}")
    print(f"harmonic_aggregation_status={PHASE_K_AGGREGATION_STATUS}")
    print(
        f"harmonic_max_order="
        f"{harmonic_max_order}"
    )
    print(
        f"harmonic_power_exponent="
        f"{harmonic_power_exponent}"
    )
    print(
        f"harmonic_weight_normalization="
        f"{harmonic_weight_normalization}"
    )
    print(
        "harmonic_amplitude_weights="
        f"{harmonic_weights.squeeze(-1).squeeze(-1).detach().cpu().tolist()}"
    )
    print(
        "harmonic_effective_power_weights="
        f"{effective_power_weights.detach().cpu().tolist()}"
    )
    print(
        "harmonic_effective_power_sum="
        f"{effective_power_weights.sum().item()}"
    )

    hcqt_params = {'sample_rate': sample_rate,
                   'hop_length': hop_length,
                   'fmin': fmin,
                   'bins_per_octave': bins_per_octave,
                   'n_bins': n_bins,
                   'gamma': None,
                   'harmonics': harmonics,
                   'weights' : harmonic_weights}

    fmin = librosa.midi_to_hz(fmin)

    CFP_HMLC_params = { 'fs': sample_rate,
                        'win': CFP_HMLC_win,
                        'hop': hop_length,
                        'fc': fmin,
                        'tc': 1/(fmin*(2**n_octaves)),
                        'NumPerOctave': bins_per_octave,
                        'fr': CFP_HMLC_fr,
                        'g': CFP_HMLC_g,
                        'bov': CFP_HMLC_bov,
                        'Har': CFP_HMLC_Har
    }

    # Determine maximum supported MIDI frequency
    fmax = fmin + n_bins / (bins_per_octave / 12)

    if checkpoint_path is None:
        # Initialize autoencoder model
        model = Timbre_Drill(cqt_params=hcqt_params,
                        CFP_HMLC_params=CFP_HMLC_params,
                        latent_size=128,
                        model_complexity=2,
                        model_complexity_onset=1,
                        skip_connections=True,
                        onset_bins_per_semitone=onset_bins_per_semitone,
                        CFP_mode=CFP_mode)
    else:
        # Load weights of the specified model checkpoint
        model = SS_NT.load(checkpoint_path, device=device)

    if len(gpu_ids) > 1:
        # Wrap model for multi-GPU usage
        model = DataParallel(model, device_ids=gpu_ids)

    model = model.to(device)

    # All three source datasets use the same annotation-free manifest interface.
    ss_mpe_experiments_audio_manifest_for_log = os.environ.get(
        "SS_MPE_EXPERIMENTS_TRAIN_AUDIO_MANIFEST"
    )
    if not ss_mpe_experiments_audio_manifest_for_log:
        raise RuntimeError("SS_MPE_EXPERIMENTS_TRAIN_AUDIO_MANIFEST is required")

    split_config_path = "SS_MPE_EXPERIMENTS_AUDIO_ONLY"
    urmp_base_dir = "SS_MPE_EXPERIMENTS_AUDIO_ONLY"
    urmp_train_splits = ["audio-only"]
    urmp_val_splits = []

    print("ss_mpe_experiments_audio_manifest:", ss_mpe_experiments_audio_manifest_for_log)
    print("train_splits:", urmp_train_splits)
    print("val/test_splits:", urmp_val_splits)
    print("n_train_splits:", len(urmp_train_splits))
    print("n_val_test_splits:", len(urmp_val_splits))
    print("overlap: audio-only validation-loss selection")
    print("============================================================")

    pair_true_values = {
        "1", "true", "yes", "on", "enabled"
    }
    enable_pair_negative = (
        os.environ.get(
            "ENABLE_PAIR_NEGATIVE", "0"
        ).strip().lower()
        in pair_true_values
    )
    pair_activity_source = os.environ.get(
        "PAIR_ACTIVITY_SOURCE", "positive_label"
    ).strip()
    pair_bins_per_octave = float(
        os.environ.get(
            "PAIR_BINS_PER_OCTAVE",
            str(bins_per_octave),
        )
    )
    pair_candidate_threshold = float(
        os.environ.get(
            "PAIR_CANDIDATE_THRESHOLD", "0.1"
        )
    )
    pair_min_valid_pairs = int(
        os.environ.get(
            "PAIR_MIN_VALID_PAIRS", "2"
        )
    )
    pair_eps = float(
        os.environ.get("PAIR_EPS", "1e-8")
    )
    alpha_pair_neg = float(
        os.environ.get("ALPHA_PAIR_NEG", "0.05")
    )
    pair_time_smoothing_frames = int(
        os.environ.get(
            "PAIR_TIME_SMOOTHING_FRAMES", "1"
        )
    )
    pair_loss_start_step = int(
        os.environ.get(
            "PAIR_LOSS_START_STEP", "0"
        )
    )
    pair_loss_ramp_steps = int(
        os.environ.get(
            "PAIR_LOSS_RAMP_STEPS", "0"
        )
    )
    pair_validate_activity_range = (
        os.environ.get(
            "PAIR_VALIDATE_ACTIVITY_RANGE", "0"
        ).strip().lower()
        in pair_true_values
    )

    if pair_activity_source != "positive_label":
        raise NotImplementedError(
            "The first implementation supports only "
            "PAIR_ACTIVITY_SOURCE=positive_label"
        )
    if pair_time_smoothing_frames != 1:
        raise NotImplementedError(
            "The first implementation requires "
            "PAIR_TIME_SMOOTHING_FRAMES=1"
        )
    if pair_min_valid_pairs != 2:
        raise ValueError(
            "PAIR_MIN_VALID_PAIRS must be 2"
        )
    if not 0.0 <= pair_candidate_threshold <= 1.0:
        raise ValueError(
            "PAIR_CANDIDATE_THRESHOLD must be in [0,1]"
        )
    if alpha_pair_neg < 0.0:
        raise ValueError("ALPHA_PAIR_NEG must be non-negative")
    if pair_loss_start_step < 0:
        raise ValueError(
            "PAIR_LOSS_START_STEP must be non-negative"
        )
    if pair_loss_ramp_steps < 0:
        raise ValueError(
            "PAIR_LOSS_RAMP_STEPS must be non-negative"
        )

    # Strict upper-harmonic positive-anchor configuration.
    strict_positive_true_values = {"1", "true", "yes", "on"}
    enable_strict_positive = (
        os.environ.get(
            "ENABLE_STRICT_POSITIVE", "0"
        ).strip().lower()
        in strict_positive_true_values
    )
    strict_positive_mode = os.environ.get(
        "STRICT_POSITIVE_MODE",
        "strict_upper_relative_energy",
    ).strip()
    lambda_strict_positive = float(
        os.environ.get("LAMBDA_STRICT_POSITIVE", "0.0")
    )
    strict_candidate_threshold = float(
        os.environ.get(
            "STRICT_CANDIDATE_THRESHOLD",
            str(pair_candidate_threshold),
        )
    )
    threshold_base_snr_db = float(
        os.environ.get("STRICT_THRESHOLD_BASE_SNR_DB", "10.0")
    )
    threshold_db_2x = float(
        os.environ.get("STRICT_THRESHOLD_DB_2X", "-10.0")
    )
    threshold_db_3x = float(
        os.environ.get("STRICT_THRESHOLD_DB_3X", "-12.0")
    )
    threshold_db_1p5x = float(
        os.environ.get("STRICT_THRESHOLD_DB_1P5X", "-15.0")
    )
    strict_harmonic_band_radius_bins = int(
        os.environ.get(
            "STRICT_HARMONIC_BAND_RADIUS_BINS", "0"
        )
    )
    strict_tolerance_text = os.environ.get(
        "STRICT_HARMONIC_TOLERANCE_CENTS", ""
    ).strip()
    strict_harmonic_tolerance_cents = (
        None
        if strict_tolerance_text.lower() in {"", "none"}
        else float(strict_tolerance_text)
    )
    strict_harmonic_energy_reduction = os.environ.get(
        "STRICT_HARMONIC_ENERGY_REDUCTION", "sum"
    ).strip()
    strict_peak_radius_bins = int(
        os.environ.get("STRICT_PEAK_RADIUS_BINS", "2")
    )
    strict_peak_prominence_threshold_db = float(
        os.environ.get(
            "STRICT_PEAK_PROMINENCE_THRESHOLD_DB", "6.0"
        )
    )
    strict_relative_energy_eps = float(
        os.environ.get("STRICT_RELATIVE_ENERGY_EPS", "1e-8")
    )

    if lambda_strict_positive < 0.0:
        raise ValueError(
            "LAMBDA_STRICT_POSITIVE must be non-negative"
        )
    if not 0.0 <= strict_candidate_threshold <= 1.0:
        raise ValueError(
            "STRICT_CANDIDATE_THRESHOLD must be in [0, 1]"
        )
    if strict_harmonic_band_radius_bins < 0:
        raise ValueError(
            "STRICT_HARMONIC_BAND_RADIUS_BINS must be non-negative"
        )
    if strict_peak_radius_bins < 1:
        raise ValueError(
            "STRICT_PEAK_RADIUS_BINS must be at least 1"
        )
    if strict_relative_energy_eps <= 0.0:
        raise ValueError(
            "STRICT_RELATIVE_ENERGY_EPS must be positive"
        )

    print("===== Strict upper-positive configuration =====")
    print("enable_strict_positive:", enable_strict_positive)
    print("strict_positive_mode:", strict_positive_mode)
    print("lambda_strict_positive:", lambda_strict_positive)
    print(
        "strict_candidate_threshold:",
        strict_candidate_threshold,
    )
    print(
        "threshold_base_snr_db:",
        threshold_base_snr_db,
        "(PROVISIONAL)",
    )
    print(
        "threshold_db_2x:",
        threshold_db_2x,
        "(PROVISIONAL)",
    )
    print(
        "threshold_db_3x:",
        threshold_db_3x,
        "(PROVISIONAL)",
    )
    print(
        "threshold_db_1p5x:",
        threshold_db_1p5x,
        "(PROVISIONAL)",
    )
    print(
        "strict_harmonic_band_radius_bins:",
        strict_harmonic_band_radius_bins,
    )
    print(
        "strict_harmonic_tolerance_cents:",
        strict_harmonic_tolerance_cents,
    )
    print(
        "strict_harmonic_energy_reduction:",
        strict_harmonic_energy_reduction,
    )
    print(
        "strict_peak_radius_bins:",
        strict_peak_radius_bins,
    )
    print(
        "strict_peak_prominence_threshold_db:",
        strict_peak_prominence_threshold_db,
    )
    print(
        "strict_relative_energy_eps:",
        strict_relative_energy_eps,
    )

    print("===== Pair-relation negative configuration =====")
    print("enable_pair_negative:", enable_pair_negative)
    print("pair_activity_source:", pair_activity_source)
    print("pair_bins_per_octave:", pair_bins_per_octave)
    print(
        "pair_candidate_threshold:",
        pair_candidate_threshold,
    )
    print("pair_min_valid_pairs:", pair_min_valid_pairs)
    print("pair_eps:", pair_eps)
    print("alpha_pair_neg:", alpha_pair_neg)
    print(
        "pair_time_smoothing_frames:",
        pair_time_smoothing_frames,
    )
    print("pair_loss_start_step:", pair_loss_start_step)
    print("pair_loss_ramp_steps:", pair_loss_ramp_steps)
    print(
        "pair_validate_activity_range:",
        pair_validate_activity_range,
    )

    wandb_run_name = os.path.basename(os.path.normpath(root_dir))
    wandb_tracker = initialize_wandb_tracking(
        run_name=wandb_run_name,
        run_dir=root_dir,
        config={
            "learning_rate": learning_rates[0],
            "learning_rate_encoder": learning_rates[0],
            "learning_rate_decoder": learning_rates[1],
            "learning_rate_onset_encoder": learning_rates[2],
            "learning_rate_onset_decoder": learning_rates[3],
            "epochs": max_epochs,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "checkpoint_interval_steps": checkpoint_interval,
            "optimizer": "AdamW",
            "seed": seed,
            "phase1": Phase1,
            "phase2": Phase2,
            "phase3": Phase3,
            "pitch_set": pitch_set,
            "onset_set": onset_set,
            "n_secs": n_secs,
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "fmin_midi": float(librosa.hz_to_midi(fmin)),
            "bins_per_octave": bins_per_octave,
            "n_octaves": n_octaves,
            "harmonics": harmonics,
            "harmonic_max_order": harmonic_max_order,
            "harmonic_power_exponent": harmonic_power_exponent,
            "harmonic_aggregation_order": expected_aggregation_order,
            "harmonic_aggregation_status": PHASE_K_AGGREGATION_STATUS,
            "harmonic_weight_normalization": harmonic_weight_normalization,
            "harmonic_amplitude_weights": (
                harmonic_weights.squeeze(-1)
                .squeeze(-1)
                .detach()
                .cpu()
                .tolist()
            ),
            "harmonic_effective_power_weights": (
                effective_power_weights.detach().cpu().tolist()
            ),
            "enable_pair_negative": enable_pair_negative,
            "pair_activity_source": pair_activity_source,
            "pair_bins_per_octave": pair_bins_per_octave,
            "pair_candidate_threshold": pair_candidate_threshold,
            "pair_min_valid_pairs": pair_min_valid_pairs,
            "pair_eps": pair_eps,
            "alpha_pair_neg": alpha_pair_neg,
            "pair_time_smoothing_frames": pair_time_smoothing_frames,
            "pair_loss_start_step": pair_loss_start_step,
            "pair_loss_ramp_steps": pair_loss_ramp_steps,
            "enable_strict_positive": enable_strict_positive,
            "strict_positive_mode": strict_positive_mode,
            "lambda_strict_positive": lambda_strict_positive,
            "strict_candidate_threshold": strict_candidate_threshold,
            "strict_threshold_base_snr_db": threshold_base_snr_db,
            "strict_threshold_db_2x": threshold_db_2x,
            "strict_threshold_db_3x": threshold_db_3x,
            "strict_threshold_db_1p5x": threshold_db_1p5x,
            "strict_harmonic_band_radius_bins": strict_harmonic_band_radius_bins,
            "strict_harmonic_tolerance_cents": strict_harmonic_tolerance_cents,
            "strict_harmonic_energy_reduction": strict_harmonic_energy_reduction,
            "strict_peak_radius_bins": strict_peak_radius_bins,
            "strict_peak_prominence_threshold_db": strict_peak_prominence_threshold_db,
            "strict_relative_energy_eps": strict_relative_energy_eps,
            "loss_multipliers": multipliers,
            "validation_threshold": 0.5,
            "validation_criteria_set": validation_criteria_set,
            "validation_criteria_metric": validation_criteria_metric,
            "split_config": split_config_path,
            "train_splits": urmp_train_splits,
            "validation_splits": urmp_val_splits,
            "device": str(device),
            "gpu_ids": gpu_ids,
        },
    )

    # Training set: URMP train splits only.
    # Phase N: audio-only Dataset construction.
    # No annotation-bearing Dataset is instantiated here.
    audio_manifest = os.environ.get("SS_MPE_EXPERIMENTS_TRAIN_AUDIO_MANIFEST")
    validation_manifest = os.environ.get("SS_MPE_EXPERIMENTS_VALIDATION_AUDIO_MANIFEST")
    if not audio_manifest or not validation_manifest:
        raise RuntimeError(
            "SS_MPE_EXPERIMENTS_TRAIN_AUDIO_MANIFEST and "
            "SS_MPE_EXPERIMENTS_VALIDATION_AUDIO_MANIFEST are required"
        )

    phase_n_train = RandomAudioClips.from_manifest(
        audio_manifest,
        sample_rate=sample_rate,
        n_secs=n_secs,
    )
    phase_n_val = NamedAudioOnlyDataset(
        FixedAudioClips.from_manifest(
            validation_manifest,
            sample_rate=sample_rate,
            n_secs=n_secs,
        ),
        "ss_mpe_experiments_val_audio_only",
    )
    validation_criteria_set = "ss_mpe_experiments_val_audio_only"
    validation_criteria_metric = "loss/total"
    validation_criteria_maximize = False
    onset_train = ComboDataset([phase_n_train])

    if len(onset_train):
        to_onset_loader = onset_train
        origin_step_per_loader = len(onset_train) // batch_size

        if origin_step_per_loader < checkpoint_interval:
            augment_step = (
                checkpoint_interval
                // max(origin_step_per_loader, 1)
                + 1
            )
            for _ in range(augment_step):
                to_onset_loader = ConcatDataset(
                    [to_onset_loader, onset_train]
                )

        onset_loader = DataLoader(
            dataset=to_onset_loader,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise RuntimeError("Phase N audio-only training set is empty.")

    # Held-out audio only; no ground-truth labels are constructed.
    validation_sets_onset = []
    validation_sets = [phase_n_val]
    evaluation_sets = []

    #################
    ## PREPARATION ##
    #################

    # Initialize an optimizer for the model parameters with differential learning rates
    optimizer = torch.optim.AdamW([{'params' : model.encoder_parameters(), 'lr' : learning_rates[0]},
                                   {'params' : model.decoder_parameters(), 'lr' : learning_rates[1]}])

    optimizer_onset = torch.optim.AdamW([
                                {'params' : model.encoder_onset_parameters(), 'lr' : learning_rates[2]},
                                {'params' : model.decoder_onset_parameters(), 'lr' : learning_rates[3]}])

    # Determine amount of batches in one epoch
    # onset train is shorter
    epoch_steps = len(onset_train)

    # Compute number of validation checkpoints corresponding to learning rate decay cooldown and window
    n_checkpoints_cooldown = math.ceil(n_epochs_cooldown * epoch_steps / checkpoint_interval)
    n_checkpoints_decay = math.ceil(n_epochs_decay * epoch_steps / checkpoint_interval)

    if n_epochs_early_stop is not None:
        # Compute number of validation checkpoints corresponding to early stopping window
        n_checkpoints_early_stop = math.ceil(n_epochs_early_stop * epoch_steps / checkpoint_interval)
    else:
        # Early stopping is disabled
        n_checkpoints_early_stop = None

    # Warmup global learning rate over a fixed number of steps according to a cosine function
    warmup_scheduler = CosineWarmup(optimizer, n_steps=n_epochs_warmup * checkpoint_interval)
    #warmup_scheduler_note = CosineWarmup(optimizer_note, n_steps=n_epochs_warmup * checkpoint_interval)
    warmup_scheduler_onset = CosineWarmup(optimizer_onset, n_steps=n_epochs_warmup * checkpoint_interval)

    # Decay global learning rate by a factor of 1/2 after validation performance has plateaued
    decay_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                 mode='max' if validation_criteria_maximize else 'min',
                                                                 factor=0.5,
                                                                 patience=n_checkpoints_decay,
                                                                 threshold=2E-3,
                                                                 cooldown=n_checkpoints_cooldown)

    # decay_scheduler_note = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_note,
    #                                                              mode='max' if validation_criteria_maximize else 'min',
    #                                                              factor=0.5,
    #                                                              patience=n_checkpoints_decay,
    #                                                              threshold=2E-3,
    #                                                              cooldown=n_checkpoints_cooldown)

    decay_scheduler_onset = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_onset,
                                                                 mode='max' if validation_criteria_maximize else 'min',
                                                                 factor=0.5,
                                                                 patience=n_checkpoints_decay,
                                                                 threshold=2E-3,
                                                                 cooldown=n_checkpoints_cooldown)

    # Enable anomaly detection to debug any NaNs (can increase overhead)
    # torch.autograd.set_detect_anomaly(True)

    # Enable cuDNN auto-tuner to optimize CUDA kernel (might improve
    # performance, but adds initial overhead to find the best kernel)
    cudnn_benchmarking = False

    if cudnn_benchmarking:
        # Enable benchmarking prior to training
        torch.backends.cudnn.benchmark = True

    # Construct the path to the directory for saving models
    log_dir = os.path.join(root_dir, 'models')
    os.makedirs(log_dir, exist_ok=True)

    # Initialize a writer to log results
    writer = SummaryWriter(log_dir)

    # Number of batches that have been processed
    batch_count = 0

    # Keep track of the model with the best validation results
    best_model_checkpoint = None

    # Keep track of the best model's results for comparison
    best_results = None

    # Counter for number of checkpoints since previous best results
    n_checkpoints_elapsed = 0

    # Flag to indicate early stopping criteria has been met
    early_stop_criteria = False

    #################
    ## TIMBRE LOSS ##
    #################

    # Maximum amplitude for Gaussian equalization
    max_A = 0.375

    # Maximum standard deviation for Gaussian equalization
    max_std_dev = 2 * bins_per_octave

    # Whether to sample fixed rather than varied shapes
    fixed_shape = False

    # Set keyword arguments for Gaussian equalization
    gaussian_kwargs = {
        'max_A' : max_A,
        'max_std_dev' : max_std_dev,
        'fixed_shape' : fixed_shape,
        'CFP_mode' : CFP_mode
    }

    # Set equalization type and corresponding parameter values
    eq_fn, eq_kwargs = sample_gaussian_equalization, gaussian_kwargs

    ####################
    ## GEOMETRIC LOSS ##
    ####################

    # Determine training sequence length in frames
    n_frames = int(n_secs * sample_rate / hop_length)

    # Define maximum time and frequency shift
    max_shift_v = 2 * bins_per_octave
    max_shift_h = n_frames // 4

    # Maximum rate by which audio can be sped up or slowed down
    max_stretch_factor = 2

    # Set keyword arguments for geometric transformations
    gm_kwargs = {
        'max_shift_v' : max_shift_v,
        'max_shift_h' : max_shift_h,
        'max_stretch_factor' : max_stretch_factor
    }

    ##############################
    ## TRAINING/VALIDATION LOOP ##
    ##############################

    # Loop through epochs
    for i in range(max_epochs):

        if max_steps is not None and batch_count >= max_steps:
            print(f'Reached max_steps={max_steps} at batch_count={batch_count}. Stopping training loop.')
            break

        if Phase1:

            #Phase1_loader = loader

            Phase1_loader = onset_loader

            loader_multiplier = 1

            # Loop through batches of audio
            for data in tqdm(Phase1_loader, desc=f'Epoch {i + 1} pitch'):
                # Increment the batch counter
                batch_count += 1

                if warmup_scheduler.is_active():
                    # Step the learning rate warmup scheduler
                    warmup_scheduler.step()

                # Extract audio and add to appropriate device
                audio = data[constants.KEY_AUDIO].to(device)

                # Log the current learning rates for this batch
                writer.add_scalar('train/loss/learning_rate/encoder', optimizer.param_groups[0]['lr'], batch_count)
                writer.add_scalar('train/loss/learning_rate/decoder', optimizer.param_groups[1]['lr'], batch_count)

                # Compute full set of spectral features
                features = model.get_all_features(
                    audio,
                    CFP_mode=CFP_mode,
                    return_hcqt_amplitude=enable_strict_positive,
                )

                # Extract relevant feature sets
                input = features['hcqt'] # (B, 2, F(per), T)
                weak_label_neg = features['pitch_negative_label'] # (B, F(per), T)
                weak_label_pos = features['pitch_positive_label'] # (B, F(per), T)
                strict_hcqt_amplitude = (
                    features.get("hcqt_amplitude")
                )

                # tfrL0 = features['tfrL0']
                # tfrLF = features['tfrLF']
                # tfrLQ = features['tfrLQ']

                with torch.autocast(device_type=f'cuda'):

                    #############
                    ## Phase 1 ##
                    #############
                    '''------layer setting------'''
                    model.set_model_trainable()
                    model.set_layer_freeze(model.encoder_onset)
                    model.set_layer_freeze(model.decoder_onset)
                    '''------layer setting------'''

                    output = model(input)

                    pitch_logits = output['pitch_logits']
                    pitch_const = output['pitch_const']
                    pitch_salience = output['pitch_salience']

                    # RECONSTRUCTION
                    reconstruction_loss = compute_reconstruction_loss(weak_label_neg, pitch_const)
                    writer.add_scalar('train/loss/reconstruction', reconstruction_loss.item(), batch_count)

                    # PITCH SPARSITY
                    pitch_sparsity_loss = compute_sparsity_loss(pitch_salience)
                    writer.add_scalar('train/loss/pitch_sparsity', pitch_sparsity_loss.item(), batch_count)

                    # PITCH BINARY CROSS ENTROPY
                    positive_support_mask_alpha = float(
                        os.environ.get(
                            "POSITIVE_SUPPORT_MASK_ALPHA", "0.0"
                        )
                    )
                    if not 0.0 <= positive_support_mask_alpha <= 1.0:
                        raise ValueError(
                            "POSITIVE_SUPPORT_MASK_ALPHA must be in [0, 1]"
                        )
                    support_negative_label = (
                        1.0
                        - (1.0 - weak_label_neg)
                        * (
                            1.0
                            - positive_support_mask_alpha
                            * weak_label_pos.detach()
                        )
                    )
                    pitch_support_loss = compute_support_loss(
                        pitch_logits, support_negative_label
                    )
                    writer.add_scalar('train/loss/pitch_support', pitch_support_loss.item(), batch_count)
                    pitch_harmonic_loss = compute_harmonic_loss(pitch_logits, weak_label_pos)
                    writer.add_scalar('train/loss/pitch_harmonic', pitch_harmonic_loss.item(), batch_count)

                    pair_neg_loss_value = None
                    pair_neg_metrics = {}
                    pair_alpha_effective = 0.0

                    # The complete relation branch is skipped when OFF.
                    # This preserves the original objective, outputs,
                    # gradients, and random-number consumption.
                    if (
                        enable_pair_negative
                        and batch_count >= pair_loss_start_step
                    ):
                        if pair_loss_ramp_steps > 0:
                            pair_alpha_effective = (
                                alpha_pair_neg
                                * min(
                                    1.0,
                                    max(
                                        0.0,
                                        (
                                            batch_count
                                            - pair_loss_start_step
                                            + 1
                                        )
                                        / pair_loss_ramp_steps,
                                    ),
                                )
                            )
                        else:
                            pair_alpha_effective = alpha_pair_neg

                        pair_maps = (
                            build_pair_negative_confidence(
                                weak_label_pos,
                                bins_per_octave=(
                                    pair_bins_per_octave
                                ),
                                freq_dim=-2,
                                candidate_threshold=(
                                    pair_candidate_threshold
                                ),
                                min_valid_pairs=(
                                    pair_min_valid_pairs
                                ),
                                eps=pair_eps,
                                validate_range=(
                                    pair_validate_activity_range
                                ),
                            )
                        )
                        pair_neg_loss_value = pair_negative_loss(
                            pitch_logits,
                            pair_maps["w_minus"],
                            pair_maps["base_mask"],
                        )

                        writer.add_scalar(
                            "pair_neg/loss",
                            pair_neg_loss_value.item(),
                            batch_count,
                        )
                        writer.add_scalar(
                            "pair_neg/weighted_loss",
                            (
                                pair_alpha_effective
                                * pair_neg_loss_value.detach()
                            ).item(),
                            batch_count,
                        )

                        if wandb_tracker.should_log(batch_count):
                            pair_neg_metrics = (
                                summarize_pair_negative(
                                    pair_maps,
                                    pair_neg_loss_value,
                                )
                            )
                            pair_neg_metrics[
                                "pair_neg/alpha_effective"
                            ] = pitch_logits.new_tensor(
                                pair_alpha_effective
                            )
                            pair_neg_metrics[
                                "pair_neg/weighted_loss"
                            ] = (
                                pair_alpha_effective
                                * pair_neg_loss_value.detach()
                            )

                            for (
                                pair_metric_name,
                                pair_metric_value,
                            ) in pair_neg_metrics.items():
                                writer.add_scalar(
                                    pair_metric_name,
                                    pair_metric_value.item(),
                                    batch_count,
                                )

                        del pair_maps

                    strict_positive_loss_value = None
                    strict_positive_metrics = {}

                    if enable_strict_positive:
                        if strict_hcqt_amplitude is None:
                            raise RuntimeError(
                                "hcqt_amplitude was not returned"
                            )

                        strict_maps = build_strict_upper_positive(
                            strict_hcqt_amplitude,
                            weak_label_pos,
                            harmonics=harmonics,
                            bins_per_octave=bins_per_octave,
                            candidate_threshold=(
                                strict_candidate_threshold
                            ),
                            threshold_base_snr_db=(
                                threshold_base_snr_db
                            ),
                            threshold_db_2x=threshold_db_2x,
                            threshold_db_3x=threshold_db_3x,
                            threshold_db_1p5x=(
                                threshold_db_1p5x
                            ),
                            harmonic_band_radius_bins=(
                                strict_harmonic_band_radius_bins
                            ),
                            harmonic_tolerance_cents=(
                                strict_harmonic_tolerance_cents
                            ),
                            harmonic_energy_reduction=(
                                strict_harmonic_energy_reduction
                            ),
                            peak_radius_bins=(
                                strict_peak_radius_bins
                            ),
                            peak_prominence_threshold_db=(
                                strict_peak_prominence_threshold_db
                            ),
                            relative_energy_eps=(
                                strict_relative_energy_eps
                            ),
                            positive_mode=strict_positive_mode,
                        )

                        strict_positive_loss_value = (
                            strict_positive_loss(
                                pitch_logits,
                                strict_maps["strict_positive"],
                            )
                        )

                        writer.add_scalar(
                            "strict_positive/loss",
                            strict_positive_loss_value.item(),
                            batch_count,
                        )
                        writer.add_scalar(
                            "strict_positive/weighted_loss",
                            (
                                lambda_strict_positive
                                * strict_positive_loss_value.detach()
                            ).item(),
                            batch_count,
                        )

                        if wandb_tracker.should_log(batch_count):
                            strict_positive_metrics = (
                                summarize_strict_upper_positive(
                                    strict_maps,
                                    strict_positive_loss_value,
                                )
                            )

                            strict_positive_metrics[
                                "strict_positive/lambda"
                            ] = pitch_logits.new_tensor(
                                lambda_strict_positive
                            )
                            strict_positive_metrics[
                                "strict_positive/weighted_loss"
                            ] = (
                                lambda_strict_positive
                                * strict_positive_loss_value.detach()
                            )

                            for (
                                strict_metric_name,
                                strict_metric_value,
                            ) in strict_positive_metrics.items():
                                writer.add_scalar(
                                    strict_metric_name,
                                    strict_metric_value.item(),
                                    batch_count,
                                )

                            print(
                                "STRICT_POSITIVE_DIAGNOSTIC "
                                + json.dumps(
                                    {
                                        name: float(
                                            value.detach()
                                            .float()
                                            .cpu()
                                            .item()
                                        )
                                        for name, value
                                        in strict_positive_metrics.items()
                                    },
                                    sort_keys=True,
                                )
                            )

                        del strict_maps

                    diagnostic_metrics = {}
                    if wandb_tracker.should_log(batch_count):
                        diagnostic_metrics = compute_harmonic_diagnostics(
                            pitch_logits=pitch_logits,
                            positive_label=weak_label_pos,
                            negative_label=support_negative_label,
                            support_loss=pitch_support_loss,
                            harmonic_loss=pitch_harmonic_loss,
                            sparsity_loss=pitch_sparsity_loss,
                            multipliers=multipliers,
                        )

                    # PITCH TIMBRE
                    pitch_timbre_loss = compute_pitch_timbre_loss(model, input, pitch_logits, eq_fn, **eq_kwargs)
                    writer.add_scalar('train/loss/pitch_timbre', pitch_timbre_loss.item(), batch_count)
                    # PITCH GEOMETRIC
                    pitch_geometric_loss = compute_pitch_geometric_loss(model, input, pitch_logits, **gm_kwargs)
                    writer.add_scalar('train/loss/pitch_geometric', pitch_geometric_loss.item(), batch_count)

                    # TIME SIMILARITY
                    pitch_time_sim_loss = compute_time_sim_loss(pitch_salience)
                    writer.add_scalar('train/loss/pitch_time_similarity', pitch_time_sim_loss.item(), batch_count)

                    Phase1_loss = multipliers['support_p'] * pitch_support_loss + \
                                multipliers['harmonic_p'] * pitch_harmonic_loss + \
                                multipliers['sparsity_p'] * pitch_sparsity_loss + \
                                multipliers['timbre_p'] * pitch_timbre_loss + \
                                multipliers['geometric_p'] * pitch_geometric_loss + \
                                multipliers['reconstruction'] * reconstruction_loss + \
                                multipliers['time_sim'] * pitch_time_sim_loss

                    if pair_neg_loss_value is not None:
                        Phase1_loss = (
                            Phase1_loss
                            + pair_alpha_effective
                            * pair_neg_loss_value
                        )

                    if strict_positive_loss_value is not None:
                        Phase1_loss = (
                            Phase1_loss
                            + lambda_strict_positive
                            * strict_positive_loss_value
                        )

                    Phase1_loss *= loader_multiplier

                    optimizer.zero_grad()
                    Phase1_loss.backward()

                    # Compute the average gradient norm across the encoder
                    avg_norm_encoder = average_gradient_norms(model.encoder)
                    # Log the average gradient norm of the encoder for this batch
                    writer.add_scalar('Phase1/avg_norm/encoder', avg_norm_encoder, batch_count)
                    # Determine the maximum gradient norm across encoder
                    max_norm_encoder = get_max_gradient_norm(model.encoder)
                    # Log the maximum gradient norm of the encoder for this batch
                    writer.add_scalar('Phase1/max_norm/encoder', max_norm_encoder, batch_count)

                    # Compute the average gradient norm across the decoder
                    avg_norm_decoder = average_gradient_norms(model.decoder)
                    # Log the average gradient norm of the decoder for this batch
                    writer.add_scalar('Phase1/avg_norm/decoder', avg_norm_decoder, batch_count)
                    # Determine the maximum gradient norm across decoder
                    max_norm_decoder = get_max_gradient_norm(model.decoder)
                    # Log the maximum gradient norm of the decoder for this batch
                    writer.add_scalar('Phase1/max_norm/decoder', max_norm_decoder, batch_count)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                    optimizer.step()

                    wandb_tracker.log(
                        {
                            "epoch": i + 1,
                            "global_step": batch_count,
                            "train/loss/total": Phase1_loss,
                            "train/loss/reconstruction": reconstruction_loss,
                            "train/loss/pitch_sparsity": pitch_sparsity_loss,
                            "train/loss/pitch_support": pitch_support_loss,
                            "train/loss/pitch_harmonic": pitch_harmonic_loss,
                            "train/loss/pitch_timbre": pitch_timbre_loss,
                            "train/loss/pitch_geometric": pitch_geometric_loss,
                            "train/loss/pitch_time_similarity": pitch_time_sim_loss,
                            "learning_rate/encoder": optimizer.param_groups[0]["lr"],
                            "learning_rate/decoder": optimizer.param_groups[1]["lr"],
                            "gradient_norm/encoder_average": avg_norm_encoder,
                            "gradient_norm/encoder_max": max_norm_encoder,
                            "gradient_norm/decoder_average": avg_norm_decoder,
                            "gradient_norm/decoder_max": max_norm_decoder,
                            **pair_neg_metrics,
                            **strict_positive_metrics,
                            **diagnostic_metrics,
                        },
                        step=batch_count,
                    )

                    if batch_count % checkpoint_interval == 0:
                        # Validation-loss selection: best.pt only.
                        best_model_path = os.path.join(log_dir, "best.pt")
                        # Initialize dictionary to hold all validation results
        # Held-out audio-only validation for the final experiment.
                        # It uses the training objective itself and never constructs or reads
                        # annotation / ground-truth tensors.
                        if enable_pair_negative:
                            raise RuntimeError(
                                "Final validation requires pair-negative OFF."
                            )

                        validation_loader = DataLoader(
                            dataset=phase_n_val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False,
                        )
                        validation_total = 0.0
                        validation_samples = 0
                        validation_component_totals = {
                            "support": 0.0,
                            "harmonic": 0.0,
                            "sparsity": 0.0,
                            "timbre": 0.0,
                            "geometric": 0.0,
                            "reconstruction": 0.0,
                            "time_similarity": 0.0,
                            "strict_positive": 0.0,
                        }
                        rng_devices = (
                            [device.index]
                            if device.type == "cuda" and device.index is not None
                            else []
                        )

                        model.eval()
                        with torch.random.fork_rng(devices=rng_devices):
                            # Fixed validation crops/augmentations without perturbing training RNG.
                            # This seed is deliberately independent of model seed and checkpoint step.
                            validation_rng_seed = int(
                                os.environ.get("SS_MPE_EXPERIMENTS_VALIDATION_RNG_SEED", "9173")
                            )
                            torch.manual_seed(validation_rng_seed)
                            if device.type == "cuda":
                                torch.cuda.manual_seed_all(validation_rng_seed)

                            with torch.inference_mode():
                                for validation_data in validation_loader:
                                    validation_audio = validation_data[
                                        constants.KEY_AUDIO
                                    ].to(device)

                                    validation_features = model.get_all_features(
                                        validation_audio,
                                        CFP_mode=CFP_mode,
                                        return_hcqt_amplitude=enable_strict_positive,
                                    )
                                    validation_input = validation_features["hcqt"]
                                    validation_negative = validation_features[
                                        "pitch_negative_label"
                                    ]
                                    validation_positive = validation_features[
                                        "pitch_positive_label"
                                    ]
                                    validation_hcqt_amplitude = validation_features.get(
                                        "hcqt_amplitude"
                                    )

                                    with torch.autocast(device_type="cuda"):
                                        validation_output = model(validation_input)
                                        validation_logits = validation_output["pitch_logits"]
                                        validation_const = validation_output["pitch_const"]
                                        validation_salience = validation_output[
                                            "pitch_salience"
                                        ]

                                        validation_reconstruction = (
                                            compute_reconstruction_loss(
                                                validation_negative,
                                                validation_const,
                                            )
                                        )
                                        validation_sparsity = compute_sparsity_loss(
                                            validation_salience
                                        )

                                        validation_alpha = float(
                                            os.environ.get(
                                                "POSITIVE_SUPPORT_MASK_ALPHA",
                                                "0.0",
                                            )
                                        )
                                        validation_support_target = (
                                            1.0
                                            - (1.0 - validation_negative)
                                            * (
                                                1.0
                                                - validation_alpha
                                                * validation_positive.detach()
                                            )
                                        )
                                        validation_support = compute_support_loss(
                                            validation_logits,
                                            validation_support_target,
                                        )
                                        validation_harmonic = compute_harmonic_loss(
                                            validation_logits,
                                            validation_positive,
                                        )
                                        validation_timbre = compute_pitch_timbre_loss(
                                            model,
                                            validation_input,
                                            validation_logits,
                                            eq_fn,
                                            **eq_kwargs,
                                        )
                                        validation_geometric = (
                                            compute_pitch_geometric_loss(
                                                model,
                                                validation_input,
                                                validation_logits,
                                                **gm_kwargs,
                                            )
                                        )
                                        validation_time_similarity = compute_time_sim_loss(
                                            validation_salience
                                        )

                                        validation_strict = None
                                        if enable_strict_positive:
                                            if validation_hcqt_amplitude is None:
                                                raise RuntimeError(
                                                    "validation hcqt_amplitude was not returned"
                                                )

                                            validation_strict_maps = (
                                                build_strict_upper_positive(
                                                    validation_hcqt_amplitude,
                                                    validation_positive,
                                                    harmonics=harmonics,
                                                    bins_per_octave=bins_per_octave,
                                                    candidate_threshold=(
                                                        strict_candidate_threshold
                                                    ),
                                                    threshold_base_snr_db=(
                                                        threshold_base_snr_db
                                                    ),
                                                    threshold_db_2x=threshold_db_2x,
                                                    threshold_db_3x=threshold_db_3x,
                                                    threshold_db_1p5x=threshold_db_1p5x,
                                                    harmonic_band_radius_bins=(
                                                        strict_harmonic_band_radius_bins
                                                    ),
                                                    harmonic_tolerance_cents=(
                                                        strict_harmonic_tolerance_cents
                                                    ),
                                                    harmonic_energy_reduction=(
                                                        strict_harmonic_energy_reduction
                                                    ),
                                                    peak_radius_bins=strict_peak_radius_bins,
                                                    peak_prominence_threshold_db=(
                                                        strict_peak_prominence_threshold_db
                                                    ),
                                                    relative_energy_eps=(
                                                        strict_relative_energy_eps
                                                    ),
                                                    positive_mode=strict_positive_mode,
                                                )
                                            )
                                            validation_strict = strict_positive_loss(
                                                validation_logits,
                                                validation_strict_maps["strict_positive"],
                                            )

                                        validation_loss = (
                                            multipliers["support_p"] * validation_support
                                            + multipliers["harmonic_p"]
                                            * validation_harmonic
                                            + multipliers["sparsity_p"]
                                            * validation_sparsity
                                            + multipliers["timbre_p"] * validation_timbre
                                            + multipliers["geometric_p"]
                                            * validation_geometric
                                            + multipliers["reconstruction"]
                                            * validation_reconstruction
                                            + multipliers["time_sim"]
                                            * validation_time_similarity
                                        )
                                        if validation_strict is not None:
                                            validation_loss = (
                                                validation_loss
                                                + lambda_strict_positive
                                                * validation_strict
                                            )

                                    validation_batch_size = int(validation_audio.size(0))
                                    validation_total += validation_batch_size * float(
                                        validation_loss.detach().float().cpu()
                                    )
                                    validation_raw_components = {
                                        "support": validation_support,
                                        "harmonic": validation_harmonic,
                                        "sparsity": validation_sparsity,
                                        "timbre": validation_timbre,
                                        "geometric": validation_geometric,
                                        "reconstruction": validation_reconstruction,
                                        "time_similarity": validation_time_similarity,
                                        "strict_positive": validation_strict,
                                    }
                                    for component_name, component_value in (
                                        validation_raw_components.items()
                                    ):
                                        if component_value is not None:
                                            validation_component_totals[component_name] += (
                                                validation_batch_size
                                                * float(
                                                    component_value.detach().float().cpu()
                                                )
                                            )
                                    validation_samples += validation_batch_size

                        if validation_samples == 0:
                            raise RuntimeError("audio-only validation loader is empty")

                        validation_loss_total = validation_total / validation_samples
                        validation_results = {
                            validation_criteria_set: {
                                validation_criteria_metric: validation_loss_total
                            }
                        }
                        writer.add_scalar(
                            "validation/loss/total",
                            validation_loss_total,
                            batch_count,
                        )
                        print(
                            "AUDIO_ONLY_VALIDATION: "
                            f"step={batch_count} "
                            f"loss={validation_loss_total:.9f} "
                            f"samples={validation_samples}",
                            flush=True,
                        )

                        if not math.isfinite(validation_loss_total):
                            raise RuntimeError(
                                f"non-finite validation loss at step {batch_count}: "
                                f"{validation_loss_total}"
                            )
                        history_record = {
                            "step": int(batch_count),
                            "loss_total": float(validation_loss_total),
                            "samples": int(validation_samples),
                            "validation_rng_seed": int(validation_rng_seed),
                            "raw_component_means": {
                                name: total / validation_samples
                                for name, total in validation_component_totals.items()
                            },
                        }
                        history_path = os.path.join(log_dir, "validation_history.jsonl")
                        with open(history_path, "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(history_record, sort_keys=True) + "\n")

                        # Resume optimization mode after validation.
                        model = model.to(device)
                        model.train()

                        wandb_tracker.log_validation(
                            validation_results,
                            step=batch_count,
                            epoch=i + 1,
                            learning_rates={
                                "encoder": optimizer.param_groups[0]["lr"],
                                "decoder": optimizer.param_groups[1]["lr"],
                            },
                        )

                        if cudnn_benchmarking:
                            # Re-enable benchmarking after validation
                            torch.backends.cudnn.benchmark = True

                        if decay_scheduler.patience and not warmup_scheduler.is_active() and i >= n_epochs_late_start:
                            # Step the learning rate decay scheduler by logging the validation metric for the checkpoint
                            decay_scheduler.step(validation_results[validation_criteria_set][validation_criteria_metric])

                        # Extract the result on the specified metric from the validation results for comparison
                        current_score = validation_results[validation_criteria_set][validation_criteria_metric]

                        if best_results is not None:
                            # Extract the currently tracked best result on the specified metric for comparison
                            best_score = best_results[validation_criteria_set][validation_criteria_metric]

                        if best_results is None or \
                                (validation_criteria_maximize and current_score > best_score) or \
                                (not validation_criteria_maximize and current_score < best_score):
                            print(f'New best at {batch_count} iterations...')

                            # Phase K keeps score tracking for scheduler/diagnostics,
                            # but does not retain intermediate checkpoint files.
                            # Set current checkpoint as best
                            best_model_checkpoint = batch_count
                            # Update best results
                            best_results = validation_results
                            if not smoke_test:
                                temporary_model_path = best_model_path + ".candidate"
                                try:
                                    model.save(temporary_model_path)
                                    os.replace(temporary_model_path, best_model_path)
                                finally:
                                    if os.path.exists(temporary_model_path):
                                        os.unlink(temporary_model_path)
                            selection_path = os.path.join(log_dir, "validation_selection.json")
                            with open(selection_path, "w", encoding="utf-8") as handle:
                                json.dump({
                                    "rule": "argmin validation loss/total",
                                    "selected_step": int(batch_count),
                                    "validation_loss_total": float(current_score),
                                }, handle, indent=2)
                            print(f"VALIDATION_BEST: step={batch_count} loss={current_score}")
                            # Reset number of checkpoints
                            n_checkpoints_elapsed = 0
                        else:
                            # Increment number of checkpoints
                            n_checkpoints_elapsed += 1

                        if n_checkpoints_early_stop is not None and n_checkpoints_elapsed >= n_checkpoints_early_stop:
                            # Early stop criteria has been reached
                            early_stop_criteria = True

                            break

                        if early_stop_criteria:
                            # Stop training
                            break

                    # Check the fixed training budget only after checkpointing.
                    # When max_steps is a checkpoint boundary (for example
                    # 18000 with a 300-step interval), this preserves and
                    # validates model-18000.pt before leaving the batch loop.
                    if max_steps is not None and batch_count >= max_steps:
                        early_stop_criteria = True
                        break
        else:
            print('No Phase1 training...')

        # if Phase2:
        #     for data in tqdm(onset_loader, desc=f'Epoch {i + 1} note'):
        #         # Increment the batch counter
        #         batch_count += 1

        #         if warmup_scheduler_note.is_active():
        #             # Step the learning rate warmup scheduler
        #             warmup_scheduler_note.step()

        #         audio = data[constants.KEY_AUDIO].to(device)
        #         features = model.get_all_features(audio, CFP_mode=CFP_mode)

        #         # Extract relevant feature sets
        #         input = features['hcqt'] # (B, 2, F(per), T)
        #         #onset_select = features['onset_selection']
        #         weak_label_neg = features['pitch_negative_label'] # (B, F(per), T)
        #         weak_label_pos = features['pitch_positive_label'] # (B, F(per), T)

        #         # Log the current learning rates for this batch
        #         writer.add_scalar('train/LR/learning_rate/pitch2note_E', optimizer_note.param_groups[0]['lr'], batch_count)
        #         writer.add_scalar('train/LR/learning_rate/pitch2note_D', optimizer_note.param_groups[1]['lr'], batch_count)

        #         with torch.autocast(device_type=f'cuda'):
        #             #############
        #             ## Phase 2 ##
        #             #############
        #             model.set_model_trainable()
        #             model.set_layer_freeze(model.encoder)
        #             model.set_layer_freeze(model.decoder)
        #             model.set_layer_freeze(model.note_aug)
        #             model.set_layer_freeze(model.input2note)
        #             model.set_layer_freeze(model.note2onset)

        #             _, _, latents, _, _, note_logits, note_salience, _, _, losses = model(input, contour_compress=True)

        #             # NOTE HARMONIC
        #             note_harmonic_loss = compute_harmonic_loss(note_logits, weak_label_pos)
        #             writer.add_scalar('train/loss/note_harmonic', note_harmonic_loss.item(), batch_count)

        #             # NOTE SUPPORT
        #             note_support_loss = compute_support_loss(note_logits, weak_label_neg)
        #             writer.add_scalar('train/loss/note_support', note_support_loss.item(), batch_count)

        #             # NOTE SPARSITY
        #             note_sparsity_loss = compute_sparsity_loss(note_salience)
        #             writer.add_scalar('train/loss/note_freq_sparsity', note_sparsity_loss.item(), batch_count)

        #             # NOTE COMPRESS
        #             note_time_sim_loss = compute_time_sim_loss(note_salience)
        #             writer.add_scalar('train/loss/time_compress', note_time_sim_loss.item(), batch_count)

        #             # NOTE DISTANCE
        #             note_frequency_dis_loss = compute_frequency_dis_loss(note_salience)
        #             writer.add_scalar('train/loss/frequency_distance', note_frequency_dis_loss.item(), batch_count)

        #             Phase2_loss = multipliers['note_compress'] * note_time_sim_loss + \
        #                             multipliers['note_harmonic'] * note_harmonic_loss + \
        #                             multipliers['note_support'] * note_support_loss + \
        #                             multipliers['frequency_distance'] * note_frequency_dis_loss + \
        #                             multipliers['sparsity_n'] * note_sparsity_loss

        #             optimizer_note.zero_grad()
        #             Phase2_loss.backward()

        #             # Compute the average gradient norm across the pitch2note encoder
        #             avg_norm_decoder = average_gradient_norms(model.pitch2note_E)
        #             # Log the average gradient norm of the decoder for this batch
        #             writer.add_scalar('Phase2/avg_norm/pitch2note_E', avg_norm_decoder, batch_count)
        #             # Determine the maximum gradient norm across pitch2note encoder
        #             max_norm_decoder = get_max_gradient_norm(model.pitch2note_E)
        #             # Log the maximum gradient norm of the decoder for this batch
        #             writer.add_scalar('Phase2/max_norm/pitch2note_E', max_norm_decoder, batch_count)

        #             # Compute the average gradient norm across the pitch2note decoder
        #             avg_norm_decoder = average_gradient_norms(model.pitch2note_D)
        #             # Log the average gradient norm of the decoder for this batch
        #             writer.add_scalar('Phase2/avg_norm/pitch2note_D', avg_norm_decoder, batch_count)
        #             # Determine the maximum gradient norm across pitch2note decoder
        #             max_norm_decoder = get_max_gradient_norm(model.pitch2note_D)
        #             # Log the maximum gradient norm of the decoder for this batch
        #             writer.add_scalar('Phase2/max_norm/pitch2note_D', max_norm_decoder, batch_count)

        #             # Apply gradient clipping for training stability
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        #             # Perform an optimization step
        #             optimizer_note.step()

        #         if batch_count % checkpoint_interval == 0:
        #              # Construct a path to save the model checkpoint
        #             model_path = os.path.join(log_dir, f'model-{batch_count}.pt')
        #             # Save model checkpoint
        #             model.save(model_path)

        #             if cudnn_benchmarking:
        #                 # Disable benchmarking prior to validation
        #                 torch.backends.cudnn.benchmark = False

        #             # Initialize dictionary to hold all validation results
        #             validation_results = dict()

        #             for val_set in validation_sets:
        #                 # Validate the model checkpoint on each validation dataset
        #                 validation_results[val_set.name()] = evaluate_note(model=model,
        #                                                             eval_set=val_set,
        #                                                             multipliers=multipliers,
        #                                                             THRESHOLD=0.5,
        #                                                             writer=writer,
        #                                                             i=batch_count,
        #                                                             device=device,
        #                                                             eq_fn=eq_fn,
        #                                                             eq_kwargs=eq_kwargs,
        #                                                             gm_kwargs=gm_kwargs)

        #             # Make sure model is on correct device and switch to training mode
        #             model = model.to(device)
        #             model.train()

        #             if cudnn_benchmarking:
        #                 # Re-enable benchmarking after validation
        #                 torch.backends.cudnn.benchmark = True

        #                 if decay_scheduler_note.patience and not warmup_scheduler_note.is_active() and i >= n_epochs_late_start:
        #                     # Step the learning rate decay scheduler by logging the validation metric for the checkpoint
        #                     decay_scheduler_note.step(validation_results[validation_criteria_set][validation_criteria_metric])

        #                 # Extract the result on the specified metric from the validation results for comparison
        #                 current_score = validation_results[validation_criteria_set][validation_criteria_metric]

        #                 if best_results is not None:
        #                     # Extract the currently tracked best result on the specified metric for comparison
        #                     best_score = best_results[validation_criteria_set][validation_criteria_metric]

        #                 if best_results is None or \
        #                         (validation_criteria_maximize and current_score > best_score) or \
        #                         (not validation_criteria_maximize and current_score < best_score):
        #                     print(f'New best at {batch_count} iterations...')

        #                     # Set current checkpoint as best
        #                     best_model_checkpoint = batch_count
        #                     # Update best results
        #                     best_results = validation_results
        #                     # Reset number of checkpoints
        #                     n_checkpoints_elapsed = 0
        #                 else:
        #                     # Increment number of checkpoints
        #                     n_checkpoints_elapsed += 1

        #                 if n_checkpoints_early_stop is not None and n_checkpoints_elapsed >= n_checkpoints_early_stop:
        #                     # Early stop criteria has been reached
        #                     early_stop_criteria = True

        #                     break

        #                 if early_stop_criteria:
        #                     # Stop training
        #                     break
        # else:
        #     print('No Phase2 training...')

        if Phase3:
            for data in tqdm(onset_loader, desc=f'Epoch {i + 1} onset'):
                # Increment the batch counter
                batch_count += 1

                if warmup_scheduler_onset.is_active():
                    # Step the learning rate warmup scheduler
                    warmup_scheduler_onset.step()

                audio = data[constants.KEY_AUDIO].to(device)
                features = model.get_all_features(audio, onset_mode=True, CFP_mode=CFP_mode)

                # Extract relevant feature sets
                input = features['hcqt'] # (B, 2, F(per), T)

                spectral_flux = features['spectral_flux']
                onset_select = features['onset_selection']

                weak_label_neg = features['pitch_negative_label'] # (B, F(per), T)
                #weak_label_pos = features['pitch_positive_label'] # (B, F(per), T)

                # Log the current learning rates for this batch
                writer.add_scalar('train/LR/learning_rate/encoder_onset', optimizer_onset.param_groups[0]['lr'], batch_count)
                writer.add_scalar('train/LR/learning_rate/decoder_onset', optimizer_onset.param_groups[1]['lr'], batch_count)


                with torch.autocast(device_type=f'cuda'):
                    #############
                    ## Phase 3 ##
                    #############
                    model.set_model_trainable()
                    model.set_layer_freeze(model.encoder)
                    model.set_layer_freeze(model.decoder)

                    # output = {
                    #     'pitch_logits': pitch_logits,
                    #     'pitch_const': pitch_const,
                    #     'latents': latents,
                    #     'pitch_salience': pitch_salience,

                    #     'onset_logits': onset_logits,
                    #     'pitch_logits_const': pitch_logits_const,
                    #     'latents_trans': latents_trans,
                    #     'onset_salience': onset_salience,

                    #     'HCQT_logits': HCQT_logits,
                    #     'pitch_const_const': pitch_const_const,
                    #     'latents_const': latents_const
                    # }

                    output = model(input, transcribe=True)

                    pitch_logits = output['pitch_logits']
                    pitch_const = output['pitch_const']
                    pitch_salience = output['pitch_salience']

                    # print(pitch_logits)
                    # print(pitch_const)
                    # print(pitch_salience)

                    onset_logits = output['onset_logits']
                    pitch_logits_const = output['pitch_logits_const']
                    onset_salience = output['onset_salience']

                    # print(onset_logits)
                    # print(pitch_logits_const)
                    # print(onset_salience)

                    '''pitch trans / onset const & pitch const / onset const'''
                    onset_reconstruction_loss = compute_reconstruction_loss(spectral_flux, pitch_logits_const)
                    writer.add_scalar('train/loss/onset_reconstruction', onset_reconstruction_loss.item(), batch_count)

                    '''pitch trans / onset trans'''

                    # ONSET SPARSITY
                    onset_frequency_sparsity_loss = compute_sparsity_loss(onset_salience)
                    writer.add_scalar('train/loss/onset_freq_sparsity', onset_frequency_sparsity_loss.item(), batch_count)

                    onset_time_sparsity_loss = compute_time_sparsity_loss(ProbLike(onset_logits))
                    writer.add_scalar('train/loss/onset_time_sparsity', onset_time_sparsity_loss.item(), batch_count)

                    # ONSET BINARY CROSS ENTROPY
                    onset_bce_loss = compute_onset_bce_loss(onset_logits, onset_select)
                    writer.add_scalar('train/loss/onset_bce', onset_bce_loss.item(), batch_count)
                    # ONSET TIMBRE
                    onset_timbre_loss = compute_onset_timbre_loss(model, input, onset_salience, eq_fn, **eq_kwargs)
                    writer.add_scalar('train/loss/onset_timbre', onset_timbre_loss.item(), batch_count)
                    # ONSET GEOMETRIC
                    onset_geometric_loss = compute_onset_geometric_loss(model, input, onset_salience, **gm_kwargs)
                    writer.add_scalar('train/loss/onset_geometric', onset_geometric_loss.item(), batch_count)

                    Phase3_loss = multipliers['bce_o'] * onset_bce_loss + \
                                    multipliers['sparsity_f_o'] * onset_frequency_sparsity_loss + \
                                    multipliers['sparsity_t_o'] * onset_time_sparsity_loss + \
                                    multipliers['timbre_o'] * onset_timbre_loss + \
                                    multipliers['geometric_o'] * onset_geometric_loss + \
                                    multipliers['reconstruction_o'] * onset_reconstruction_loss

                    optimizer_onset.zero_grad()
                    Phase3_loss.backward()

                    # Compute the average gradient norm across the encoder_onset
                    avg_norm_encoder = average_gradient_norms(model.encoder_onset)
                    # Log the average gradient norm of the encoder for this batch
                    writer.add_scalar('Phase3/avg_norm/encoder_onset', avg_norm_encoder, batch_count)
                    # Determine the maximum gradient norm across encoder_onset
                    max_norm_encoder = get_max_gradient_norm(model.encoder_onset)
                    # Log the maximum gradient norm of the encoder for this batch
                    writer.add_scalar('Phase3/max_norm/encoder_onset', max_norm_encoder, batch_count)

                    # Compute the average gradient norm across the decoder_onset
                    avg_norm_decoder = average_gradient_norms(model.decoder_onset)
                    # Log the average gradient norm of the decoder for this batch
                    writer.add_scalar('Phase3/avg_norm/decoder_onset', avg_norm_decoder, batch_count)
                    # Determine the maximum gradient norm across decoder_onset
                    max_norm_decoder = get_max_gradient_norm(model.decoder_onset)
                    # Log the maximum gradient norm of the decoder for this batch
                    writer.add_scalar('Phase3/max_norm/decoder_onset', max_norm_decoder, batch_count)

                    # Apply gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    # Perform an optimization step
                    optimizer_onset.step()

                if batch_count % checkpoint_interval == 0:
                     # Construct a path to save the model checkpoint
                    model_path = os.path.join(log_dir, f'{onset_set}_{batch_count}.pt')


                    if cudnn_benchmarking:
                        # Disable benchmarking prior to validation
                        torch.backends.cudnn.benchmark = False

                    # Initialize dictionary to hold all validation results
                    validation_results = dict()

                    for val_set in validation_sets_onset:
                        # Validate the model checkpoint on each validation dataset
                        validation_results[val_set.name()] = evaluate_onset(model=model,
                                                                    eval_set=val_set,
                                                                    multipliers=multipliers,
                                                                    THRESHOLD=0.5,
                                                                    writer=writer,
                                                                    i=batch_count,
                                                                    device=device,
                                                                    eq_fn=eq_fn,
                                                                    eq_kwargs=eq_kwargs,
                                                                    gm_kwargs=gm_kwargs)

                    # Make sure model is on correct device and switch to training mode
                    model = model.to(device)
                    model.train()

                    if cudnn_benchmarking:
                        # Re-enable benchmarking after validation
                        torch.backends.cudnn.benchmark = True

                    if decay_scheduler_onset.patience and not warmup_scheduler_onset.is_active() and i >= n_epochs_late_start:
                        # Step the learning rate decay scheduler by logging the validation metric for the checkpoint
                        decay_scheduler_onset.step(validation_results[validation_criteria_set][validation_criteria_metric])

                    # Extract the result on the specified metric from the validation results for comparison
                    current_score = validation_results[validation_criteria_set][validation_criteria_metric]

                    if best_results is not None:
                        # Extract the currently tracked best result on the specified metric for comparison
                        best_score = best_results[validation_criteria_set][validation_criteria_metric]


                    if best_results is None or \
                            (validation_criteria_maximize and current_score > best_score) or \
                            (not validation_criteria_maximize and current_score < best_score):
                        print(f'New best at {batch_count} iterations...')
                        # Save model checkpoint
                        model.save(model_path)
                        # Set current checkpoint as best
                        best_model_checkpoint = batch_count
                        # Update best results
                        best_results = validation_results
                        # Reset number of checkpoints
                        n_checkpoints_elapsed = 0
                    else:
                        # Increment number of checkpoints
                        n_checkpoints_elapsed += 1

                    if n_checkpoints_early_stop is not None and n_checkpoints_elapsed >= n_checkpoints_early_stop:
                        # Early stop criteria has been reached
                        early_stop_criteria = True

                        break

                    if early_stop_criteria:
                        # Stop training
                        break
        else:
            print('No Phase3 training...')

    print(f'Achieved best results at {best_model_checkpoint} iterations...')
    best_path = os.path.join(log_dir, "best.pt")
    selection_path = os.path.join(log_dir, "validation_selection.json")
    if not os.path.isfile(selection_path):
        raise RuntimeError("No validation selection was produced")
    if not smoke_test and not os.path.isfile(best_path):
        raise RuntimeError("No validation-selected best.pt was produced")
    if smoke_test and os.path.isfile(best_path):
        raise RuntimeError("Smoke test must not save a model")
    with open(os.path.join(log_dir, "TRAINING_COMPLETE.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "best_checkpoint": None if smoke_test else best_path,
            "smoke_test": smoke_test,
        }, handle)
    wandb_tracker.set_summary({
        "best/checkpoint_step": best_model_checkpoint,
        "final/global_step": batch_count,
    })
    wandb_tracker.finish(exit_code=0)
