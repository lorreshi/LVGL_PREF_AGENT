"""骨架 smoke 测试：验证包可导入、pytest 能正常收集运行。"""

import embedded_device_agent


def test_package_importable() -> None:
    assert embedded_device_agent.__version__ == "0.1.0"


def test_subpackages_importable() -> None:
    import embedded_device_agent.api  # noqa: F401
    import embedded_device_agent.capabilities  # noqa: F401
    import embedded_device_agent.core  # noqa: F401
