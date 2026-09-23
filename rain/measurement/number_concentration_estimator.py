"""NumberConcentrationEstimator 实现：名义数量浓度 C = N / V。

V 来自配置的名义观测体积（590.4 cm³ / 0.0005904 m³），这是**双路共同观测
体积**——两台相机正交拍摄同一块空间，理论上只有双路都确认存在的目标才
真正处于这块体积里。这个类本身只是"C=N/V"这一步公式计算，不关心 N 是
从哪里数出来的：调用方是单路调试统计(single_camera_proxy)还是双路共同
体积统计(stereo_common_volume)，由调用方自己负责传对 N 并标注清楚场景，
不是这个类的职责。

volume_status 从构造参数读取（默认 "nominal_unvalidated"，因为 590.4 cm³
尚未经过实验标定），不是模块里写死的字符串常量——真正完成体积标定后，
只需要在 config/pipeline.yaml 里把这个值改成 "validated"，不用改代码。
"""

from __future__ import annotations

from ..registry import register_concentration_estimator
from .types import ConcentrationResult


@register_concentration_estimator("nominal_volume_concentration")
class NumberConcentrationEstimator:
    """C = N / V。

    参数（来自 config/pipeline.yaml 的 statistics.concentration_estimator.params）：
        volume_cm3     : 观测体积，立方厘米（写进结果方便人工核对换算）
        volume_m3      : 观测体积，立方米（真正参与计算的数值）
        volume_status  : 体积可信状态的文本标记，默认 "nominal_unvalidated"。
                         从配置读取，不是代码里的永久常量。

    volume_cm3 和 volume_m3 必须互相一致（volume_cm3 * 1e-6 ≈ volume_m3），
    构造时校验，避免配置文件手误导致两个字段对不上却没人发现。
    """

    def __init__(
        self,
        volume_cm3: float = 590.4,
        volume_m3: float = 0.0005904,
        volume_status: str = "nominal_unvalidated",
    ):
        self.volume_cm3 = float(volume_cm3)
        self.volume_m3 = float(volume_m3)
        self.volume_status = str(volume_status)

        expected_m3 = self.volume_cm3 * 1e-6
        if abs(expected_m3 - self.volume_m3) > 1e-9:
            raise ValueError(
                f"volume_cm3={self.volume_cm3} 换算成立方米应为 {expected_m3}，"
                f"但配置里 volume_m3={self.volume_m3}，两者不一致，检查 config.yaml"
            )
        if self.volume_m3 <= 0:
            raise ValueError(f"volume_m3 必须大于0，收到 {self.volume_m3}")

    def estimate(self, n_valid_targets: int) -> ConcentrationResult:
        if n_valid_targets < 0:
            raise ValueError(f"n_valid_targets 不能为负数: {n_valid_targets}")

        concentration = float(n_valid_targets) / self.volume_m3
        return ConcentrationResult(
            n_valid_targets=n_valid_targets,
            volume_m3=self.volume_m3,
            concentration_m3=concentration,
            volume_status=self.volume_status,
            unit="m^-3",
        )
