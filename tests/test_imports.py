import capability_capsule
import capability_capsule.cli
import capability_capsule.eval
import capability_capsule.packager
import capability_capsule.planner
import capability_capsule.rag
import capability_capsule.runtime
import capability_capsule.scanner
import capability_capsule.telemetry


def test_package_version() -> None:
    assert capability_capsule.__version__ == "0.1.0"
    assert capability_capsule.cli.app.info.name == "capsule"
