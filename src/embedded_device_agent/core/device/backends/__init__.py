"""具体 DeviceIO 后端子包。

SerialDeviceIO（全自动，任务 4.2）/ HalfAutoDeviceIO（半自动 + 人在环，任务 4.3）
及测试用 FakeDeviceIO 由本子包实现，并经 ``@DeviceIOFactory.register(...)`` 注册。
导入本子包应触发各后端注册。
"""
