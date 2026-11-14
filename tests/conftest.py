"""Test-session setup: never touch a real database."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="crm-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test_predictions.db"
