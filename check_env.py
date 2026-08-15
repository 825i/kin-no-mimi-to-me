#!/usr/bin/env python3
"""Environment self-check — run this first when something does not work.

    python check_env.py          (or .venv\\Scripts\\python.exe check_env.py)

Reports Python, PyTorch/CUDA, FFmpeg, the condensed-audio backend and whether the model is
already cached. Exit code 0 = ready to go, 1 = something is wrong.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
problems = []
warnings_ = []


def line(status, label, detail=""):
    print(f"[{status}] {label:<22}{detail}")


def fail(label, detail, fix=None):
    line(BAD, label, detail)
    problems.append((label, fix))


def warn(label, detail, fix=None):
    line(WARN, label, detail)
    warnings_.append((label, fix))


print("=" * 78)
print("  Japanese immersion subtitles + condensed audio - environment check")
print("=" * 78)

# ---- Python ----------------------------------------------------------------
v = sys.version_info
detail = f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})"
if (v.major, v.minor) >= (3, 9):
    line(OK, "Python", detail)
else:
    fail("Python", detail, "Need Python 3.9+. winget install Python.Python.3.12")

if sys.platform == "win32" and not os.environ.get("PYTHONUTF8"):
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf-8" not in enc:
        warn("stdout encoding", f"{enc or 'unknown'} (Japanese may fail when redirected)",
             "Use jpsubs.cmd, or set PYTHONUTF8=1")
    else:
        line(OK, "stdout encoding", enc)
else:
    line(OK, "stdout encoding", (getattr(sys.stdout, "encoding", "") or "?"))

# ---- Python packages -------------------------------------------------------
try:
    import numpy
    line(OK, "numpy", numpy.__version__)
except ImportError as e:
    fail("numpy", str(e), "pip install -r requirements.txt")

try:
    import transformers
    line(OK, "transformers", transformers.__version__)
except ImportError as e:
    fail("transformers", str(e), "pip install -r requirements.txt")

torch = None
try:
    import torch
    line(OK, "torch", torch.__version__)
except ImportError as e:
    fail("torch", str(e),
         "Windows+NVIDIA: pip install torch --index-url "
         "https://download.pytorch.org/whl/cu130")

# ---- accelerator -----------------------------------------------------------
if torch is not None:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        arch = "sm_%d%d" % cap
        arches = torch.cuda.get_arch_list()
        line(OK, "GPU", f"{torch.cuda.get_device_name(0)}  ({arch})")
        line(OK, "CUDA runtime", f"{torch.version.cuda}   kernels: {', '.join(arches)}")
        if arch not in arches:
            fail("GPU kernels", f"this torch build has no {arch} kernels",
                 "Reinstall torch from a newer CUDA index, e.g. "
                 "https://download.pytorch.org/whl/cu130")
        else:
            try:
                a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
                _ = a @ a
                torch.cuda.synchronize()
                line(OK, "GPU fp16 matmul", "works")
            except Exception as e:
                fail("GPU fp16 matmul", repr(e)[:120],
                     "Driver/toolkit mismatch - update the NVIDIA driver")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        line(OK, "GPU", "Apple MPS")
    else:
        warn("GPU", "none detected - transcription will run on the CPU (slow)",
             "Fine for testing; expect roughly 10-20x slower than a GPU")

# ---- FFmpeg ----------------------------------------------------------------
for tool in ("ffmpeg", "ffprobe"):
    p = shutil.which(tool)
    if not p:
        fix = "winget install Gyan.FFmpeg  then open a NEW terminal"
        if sys.platform != "win32":
            fix = "Install ffmpeg - https://ffmpeg.org/download.html"
        fail(tool, "not on PATH", fix)
    else:
        try:
            cp = subprocess.run([p, "-version"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
            ver = (cp.stdout or "").splitlines()[0] if cp.stdout else "?"
            line(OK, tool, ver[:60])
        except OSError as e:
            fail(tool, repr(e)[:100], None)

# libopus is what the condensed audio is encoded with.
ff = shutil.which("ffmpeg")
if ff:
    try:
        cp = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
        if "libopus" in (cp.stdout or ""):
            line(OK, "ffmpeg libopus", "present (needed for condensed audio)")
        else:
            fail("ffmpeg libopus", "missing - condensed audio cannot be encoded",
                 "Use a full FFmpeg build, e.g. winget install Gyan.FFmpeg")
    except OSError:
        pass

# ---- condensed-audio backend ----------------------------------------------
try:
    import subgen
    backend = subgen.which_condenser()
    if backend == "native":
        why = "impd cannot run on Windows" if sys.platform == "win32" else \
              "no Bash 5 found, or vendor/impd missing"
        line(OK, "condenser", f"native ({why})")
    else:
        line(OK, "condenser", f"impd via {subgen.find_bash5()}")
except Exception as e:
    fail("condenser", repr(e)[:120], None)

# ---- model cache -----------------------------------------------------------
try:
    import subgen
    model = subgen.MODEL_ID
    if subgen.model_is_available(model):
        line(OK, "model", f"{model} (cached)")
    else:
        line(OK, "model", f"{model} (not cached - ~3 GB downloads on first run)")
except Exception:
    pass

# ---- summary ---------------------------------------------------------------
print("-" * 78)
if problems:
    print(f"{len(problems)} problem(s) to fix:\n")
    for label, fix in problems:
        print(f"  * {label}")
        if fix:
            print(f"      -> {fix}")
    if warnings_:
        print()
if warnings_:
    print(f"{len(warnings_)} warning(s):\n")
    for label, fix in warnings_:
        print(f"  * {label}")
        if fix:
            print(f"      -> {fix}")
if not problems:
    print("Ready." if not warnings_ else "Ready, with warnings above.")
    print('  .\\jpsubs.cmd --dry-run --batch "D:\\path\\to\\Library"'
          if sys.platform == "win32" else
          '  ./subgen --dry-run --batch "/path/to/Library"')
sys.exit(1 if problems else 0)
