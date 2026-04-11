import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.paths import DEFAULT_LINKERHAND_SDK_PATH, PROJECT_ROOT


def test_default_linkerhand_sdk_path_points_inside_repo():
    assert DEFAULT_LINKERHAND_SDK_PATH == PROJECT_ROOT / "third_party" / "linkerhand-python-sdk"
