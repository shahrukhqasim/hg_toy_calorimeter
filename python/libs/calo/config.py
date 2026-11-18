"""Environment helpers for running the Geant4 module (minicalo).

Geant4 11 finds its data sets from its own installation (or from the G4*DATA variables that geant4.sh sets), so
nothing is hard-coded here any more; the 10.7 data paths that set_env() used to force are gone.
"""
import os
import shutil
import subprocess
import sys

is_laptop = False       # kept for old scripts; nothing in the pipeline depends on the machine any more

# Geant4's shared libraries need more static TLS than glibc reserves for dlopen'ed modules (Geant4 11.x): without it
# importing minicalo fails ("cannot allocate memory in static TLS block"). The variable is read at interpreter start,
# so ensure_tls() re-executes the script with it. 3 MB is what fcc-ai-design uses (g4sim/python/g4env.py): 2 MB is the
# minimum for Geant4 11.4, 16 MB overflows thread stacks.
TLS_TUNABLE = "glibc.rtld.optional_static_tls=3145728"


def ensure_tls():
    if sys.platform.startswith("linux") and TLS_TUNABLE not in os.environ.get("GLIBC_TUNABLES", ""):
        old = os.environ.get("GLIBC_TUNABLES")
        os.environ["GLIBC_TUNABLES"] = TLS_TUNABLE + (":" + old if old else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def set_env():
    """Kept for compatibility; Geant4 11 needs no data paths from here."""
    pass


def pythia_data(conf=None):
    """Pythia's xmldoc directory: the config key pythia8_data, else $PYTHIA8DATA, else pythia8-config --xmldoc."""
    if conf is not None and 'pythia8_data' in conf:
        return str(conf['pythia8_data'])
    if os.environ.get('PYTHIA8DATA'):
        return os.environ['PYTHIA8DATA']
    if shutil.which('pythia8-config'):
        return subprocess.check_output(['pythia8-config', '--xmldoc'], text=True).strip()
    raise RuntimeError("Pythia data not found: set PYTHIA8DATA, put pythia8-config on PATH or set pythia8_data")
