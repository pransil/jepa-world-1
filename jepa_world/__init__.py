"""JEPA world model on rolling-ball MuJoCo episodes. See SPEC.md."""

import os

# SPEC §2: MPS ops that are not implemented fall back to CPU instead of raising. This must be set
# before torch is imported, so it lives here: any `import jepa_world...` that comes before
# `import torch` gets it.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
