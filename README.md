# Adversarial Privacy for Multimodal Sensor Embeddings

Gabriel Pereira — NPSC2001 Advanced Science research project

## Overview

Wearable and camera-based activity recognition systems increasingly run on edge devices
that send learned embeddings, rather than raw sensor data, to downstream services. These
embeddings can still reveal who produced them. This project builds and evaluates a
modality-agnostic framework that trains an encoder to keep the information needed for
activity recognition while suppressing information about the subject's identity.

**Research question:** can adversarial training produce sensor embeddings that preserve
activity recognition while suppressing subject identity, and how robust is that
suppression when measured by independent attackers of increasing strength?

## Method

### Data

Experiments use [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)
(8 subjects × 27 activities × 4 trials) with two structurally different modalities:
wearable inertial data (accelerometer and gyroscope, 6 channels) and RGB video. The
dataset is not included in this repository.

### Architecture

- **IMU encoder:** a 1D convolutional network over a fixed-length inertial sequence, with
  mean and max pooling over time.
- **RGB encoder:** an r3d_18 3D CNN pretrained on Kinetics-400, with its final block
  fine-tuned on 32-frame clips.
- **Projectors:** map each modality into a shared 512-dimensional embedding space.
- **Task head:** classifies the activity from the embedding.
- **Identity probe:** an adversary that tries to identify the subject from the embedding.

### Adversarial training

A gradient reversal layer (Ganin & Lempitsky, 2015) connects the embedding to the identity
probe. The probe learns to identify the subject, while the reversed gradient pushes the
encoder to make that harder; the task head simultaneously keeps activity recognition
accurate. The adversarial strength λ is held at zero during a warm-up period and then
ramped up on a sigmoid schedule. An undefended baseline uses the identical configuration
with λ = 0.

### Evaluation

- **Leave-one-subject-out cross-validation** over all 8 subjects.
- **Utility:** activity accuracy on the held-out subject.
- **Privacy:** measured by independent attackers trained after the fact on the frozen
  embeddings, not by the adversary used during training. Attackers form a ladder of
  increasing strength:
  - a multilayer perceptron trained from scratch;
  - frozen large language models (Qwen2.5-0.5B and SmolLM2-1.7B) that read each embedding
    through a small trainable adapter as soft tokens in a text prompt;
  - a pairwise verification attack that decides whether two recordings come from the same
    subject.

## Findings

- Activity recognition accuracy is unaffected by adversarial training.
- On inertial data, adversarial training substantially reduces subject re-identification,
  but does not eliminate it. The effect varies considerably between held-out subjects.
- Stronger attackers recover more identity than weaker ones, so privacy claims depend on
  the attacker used to measure them.
- On RGB video, suppression is modest, and it only occurs when part of the pretrained
  backbone is trainable.
- The defence protects against identifying a subject from a single embedding, but not
  against deciding whether two recordings come from the same person.
- The accuracy of the adversary used during training is not a reliable measure of privacy;
  independent attackers are required.

## Repository structure

| path | purpose |
|---|---|
| `train.py` | Trains the encoders, projectors, task head and identity probe with adversarial training for one cross-validation fold |
| `tune.py` | Multi-objective hyperparameter search with Optuna |
| `attack.py` | Independent MLP attacker trained on the frozen embeddings of a checkpoint |
| `export_embeddings.py` | Exports a checkpoint's embeddings and metadata for offline analysis |
| `llm_attack.py` | Frozen LLM attacker with a trainable adapter, targeting subject or activity |
| `llm_verify.py` | Pairwise verification attack using a frozen LLM and adapter |
| `summarise_folds.py` | Aggregates per-fold results into cross-validation summary tables |
| `ppf/data/dataloader.py` | UTD-MHAD loading, inertial and RGB preprocessing, leave-one-subject-out splits |
| `ppf/models/` | Encoders, projectors, task head, identity probe, gradient reversal layer and attacker model |
| `ppf/training/multimodal_trainer.py` | The adversarial training loop |
| `ppf/training/schedule.py` | λ and learning-rate schedules |
| `requirements.txt` | Python dependencies |
