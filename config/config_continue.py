"""Low-learning-rate continuation config for the blended PG19 run."""

import importlib.util
import os


_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
_spec = importlib.util.spec_from_file_location("_base_train_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

for _key, _value in vars(_base).items():
    if not _key.startswith("_"):
        globals()[_key] = _value


# Continue from the first 10,000-step run with a low-LR refinement phase.
max_steps = 20000
steps_this_run = 10000
lr_schedule_start_step = 10000
lr_schedule_steps = 10000
warmup_steps = 0

max_lr = 3e-5
min_lr = 1e-5
muon_lr = 0.002
