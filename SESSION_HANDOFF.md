# SESSION_HANDOFF.md

> 双路正交相机雨滴测量系统 —— 会话交接文档。
> 用途：供压缩后的新会话（或任何接手者）快速恢复上下文，不需要重新翻遍全部历史对话。
> 最后更新：2026-09-23，阶段1～阶段7B全部完成。

---

## 1. 已完成阶段和验收结果

| 阶段 | 内容 | 验收结果 |
|---|---|---|
| 阶段1 | 数据读取（元数据/时间戳/双路配对/时间质量/图像完整性） | **PASS**（180/180帧对配对成功，0个时间异常，A/B两路capture_metadata.json与capture_timing.csv字节级(sha256)相同） |
| 阶段2 | 检测拆分（六步模块化：亮度归一化→背景差分→自适应阈值→形态学→连通域→轮廓提取） | **PASS**（模块化重构，与原Demo逐帧100%一致）+ **LIMITED**（检测参数未针对新数据标定：A检出24个/B检出14个，180帧，参数沿用旧数据集cama/camb经验值） |
| 阶段3 | 跨帧跟踪（KalmanPredictor + TrackAssociator(门控+NN/Hungarian) + TrackManager） | 功能完成。A：18条轨迹（仅1条达7点，17条为1点短轨迹）；B：14条轨迹（全部仅1点）。轨迹稀疏直接继承自阶段2检测稀疏，非跟踪算法缺陷 |
| 阶段4 | 双路匹配（EpipolarGate→MatchGate→CostMatrixBuilder→NN/Hungarian→StereoMatcher） | 模块**PASS**（29个单元测试覆盖唯一匹配/极线拒绝/代价超阈值/多目标冲突/无标定安全处理等7类场景）。真实数据**0个已验证匹配**（因无标定文件，`calibration_unavailable`，这是诚实结果不是bug，明确禁止用此0值代表"精度"） |
| 阶段5 | 测速（LinearVelocityEstimator，z(t)=z0+v·t） | **PASS**（18个单元测试）+ **LIMITED**：真实数据仅1条A轨迹达到≥3点拟合条件，且因残差过大(47.8px)被正确降级为低置信度；B路0条轨迹满足测速点数要求 |
| 阶段6 | 统计（NumberConcentrationEstimator/DropSizeDistributionEstimator/WindowStatistics/StatisticsQualityController/StereoCommonVolumeStatistics） | **PASS**（34个单元测试）。已修正体积语义：`single_camera_proxy`（单路非正式代理统计，`usable_for_final_concentration`恒False）与`stereo_common_volume`（590.4cm³真正对应的统计，因无标定`concentration_status`恒`unavailable`）严格分离，不互相替代 |
| 阶段7 | 性能测试（完整流水线端到端） | 完成。基线吞吐**14.4对/秒**，瓶颈`read`阶段占66.8%（已用IFD标签解析确认为LZW压缩，非猜测） |
| 阶段7B | 性能优化（逐因素测试：解码库替换/多线程读取/生产者-消费者架构） | 完成。最优方案(v3_producer_consumer)达**约39-40对/秒**，与基线相比结果**100%一致**（检测/跟踪/匹配/测速/统计逐字段比对0差异）。**仍未达到312.5/406.25对/秒目标** |

**全局测试状态**：`python -m pytest tests/ -q` → **224 collected**（阶段6/7B完成时为222 passed + 2 skipped；后续如有新增以`pytest --co`实测为准）。

**未执行**：阶段6/7B之后未进入"上位机开发"，按用户要求在阶段7B完成后停止。

---

## 2. 当前代码结构和新增文件

### 2.1 新增的阶段化模块包（`rain/` 下，全部纯新增，不改动原Demo文件）

```
rain/data/            阶段1：MetadataReader, TimingReader, TimeQualityChecker, FramePairer,
                       ImageIntegrityChecker, CalibrationLoader无关（标定在stereo包）,
                       file_hash.py, output_integrity.py, types.py
rain/detection/       阶段2：BrightnessNormalizer, BackgroundDifferencer,
                       AdaptiveThresholdSegmenter, MorphologyProcessor,
                       ConnectedComponentAnalyzer, ContourExtractor, interfaces.py
rain/detectors/modular_pipeline.py   阶段2：ModularPipelineDetector（组合六步，注册名"modular_pipeline"）
rain/tracking/        阶段3：KalmanPredictor(constant_velocity_kalman), HungarianSolver(kuhn_munkres),
                       TrackAssociator(gated_nearest_neighbor), TrackManager(gated_kalman_track_manager),
                       types.py, interfaces.py
rain/stereo/          阶段4：CalibrationLoader, EpipolarGate(fundamental_matrix_gate),
                       MatchGate(size_shape_gate), CostMatrixBuilder(weighted_normalized_cost),
                       NearestNeighborSolver(greedy_nearest_neighbor),
                       StereoMatcher(gated_cost_stereo_matcher)，复用阶段3的HungarianSolver
rain/measurement/     阶段5+6：LinearVelocityEstimator, StatisticsQualityController,
                       NumberConcentrationEstimator, DropSizeDistributionEstimator,
                       WindowStatistics(单路代理统计), StereoCommonVolumeStatistics(双路共同体积统计)
rain/perf/            阶段7+7B：StageProfiler, ResourceMonitor, TiffInspector,
                       BoundedPipelineRunner（有界队列生产者-消费者）
rain/readers/fast_tif.py   阶段7B新增：PIL解码的FrameReader，注册名"fast_tif"
                       （与原lzw_tif.py并存，像素级验证完全一致，不改动lzw_tif.py）
```

### 2.2 Pipeline 入口脚本（`pipeline/`，每个独立可运行）

```
pipeline/stage1_data_audit.py         data_audit.json / frame_pairing.csv / timing_quality.csv / image_integrity.csv
pipeline/stage2_detection.py          detections.csv（A+B合并，cam列区分）/ stage2_report.json
pipeline/stage3_tracking.py           tracks.csv / a_tracks_overlay.png / b_tracks_overlay.png / stage3_report.json
pipeline/stage4_stereo_matching.py    stereo_matches.csv / overlay/ / stage4_report.json
pipeline/stage5_velocity.py           velocities.csv / plots/ / stage5_report.json
pipeline/stage6_statistics.py         window_statistics.csv(single_camera_proxy+stereo_common_volume) /
                                       size_distribution.csv / spectrum/ / stage6_report.json
pipeline/stage7_performance.py        performance_report.csv / stage7_report.json（基线测试）
pipeline/stage7b_optimization.py      performance_comparison.csv / consistency_report.json /
                                       performance_optimization_report.json（优化对比）
```

全部运行方式：`cd D:\Codex\毕设相关\rain && python pipeline\stageN_xxx.py`

### 2.3 配置与测试与文档

```
config/pipeline.yaml       新流水线的唯一配置源，按阶段分段（data/time_quality/reader/background/
                            detection/tracking/stereo/measurement/statistics/performance/
                            performance_optimization）。与根目录旧config.yaml（服务于tools/demo.py）完全独立
docs/PYTHON_LEARNING_GUIDE.md   面向Python初学者的语法说明+每阶段可独立运行的示例
tests/test_data_stage1.py / test_detection_stage2.py / test_tracking_stage3.py /
tests/test_stereo_stage4.py / test_measurement_stage5.py / test_statistics_stage6.py /
tests/test_perf_stage7.py / test_perf_optimization_stage7b.py
                            每阶段独立测试文件，全部不依赖真实数据（合成数据/临时文件），可在任何机器跑
```

### 2.4 对既有文件的"纯增量"修改（已用`git diff`逐一核实无删除、无行为变化，只加行）

- `rain/__init__.py`：新增对`rain.data/detection/tracking/stereo/measurement`等包的import（触发注册）
- `rain/registry.py`：新增约20张子模块注册表 + register_xxx/build_xxx函数（阶段2-6各模块）
- `rain/detectors/__init__.py`：新增`from . import modular_pipeline`
- `rain/readers/__init__.py`：新增`from . import fast_tif`
- `pyproject.toml`：新增依赖 `matplotlib>=3.5`（阶段5/6画图）、`psutil>=5.9`（阶段7/7B资源监控）

---

## 3. 当前性能基线

| 指标 | 数值 | 备注 |
|---|---|---|
| 阶段7基线吞吐 | **14.4 对/秒**（单线程顺序，cv2/lzw_tif解码） | read阶段占66.8%总耗时 |
| 阶段7B优化后最优(v3_producer_consumer) | **约39～40 对/秒** | PIL解码 + 8线程读取(有界队列maxsize=8) + 4线程检测 + 单写出线程；3轮重复压力测试稳定在39.5/40.2/40.1对/秒 |
| 输入基准 | 312.5 对/秒 | **未达到** |
| 建议目标（+30%余量） | 406.25 对/秒 | **未达到** |
| 提升倍数 | 2.7×～18.4×（因基线本身波动大） | 见下方"测量环境噪声"说明 |

**已确认的瓶颈演变**：
1. 原始瓶颈：`read`（LZW解码），已用`rain/perf/tiff_inspector.py`直接解析TIFF IFD确认 Compression=LZW(5)、Predictor=2、RowsPerStrip=3，非猜测。
2. 优化后剩余瓶颈：`background`（背景建模）在最优方案v3里占**46%**的总耗时（A/B两路各1.07秒，全程单线程，未并行化——因为改动BackgroundModel算法不在阶段7B授权范围内）。
3. 次要因素：A、B两路相机的整条流水线（读取+背景建模+检测）目前是**完全顺序执行**（A做完才做B），两路数据独立、直到双路匹配才有关联，这是明确的、未执行的下一步优化方向（不涉及算法改动）。

**测量环境噪声（重要，务必知悉）**：阶段7B两次完整对比运行中，v0/v1（单线程、耗时长）的绝对数字波动极大（v0: 2.1～14.4对/秒；v1一度比v0还慢，180秒/1.0对/秒，同时CPU仅7.7%、磁盘读仅10MB——不是CPU或磁盘瓶颈，怀疑是Windows Defender实时扫描等系统层面干扰）。而v3（producer_consumer）在两次完整对比+3轮重复压力测试中**高度稳定**，全部落在39.3～40.2对/秒区间。**结论**：v3本身的~40对/秒是可信、可复现的数字；v0/v1的绝对数字仅供参考，不应作为精确基线使用。

---

## 4. 当前未解决问题

1. **吞吐未达标**：即使最优方案也只有约40对/秒，距312.5目标差约8倍。下一步建议（按预期收益排序，均未执行）：
   a. 并行处理A、B两路相机的整条流水线（不涉及算法改动，纯架构调整）；
   b. 背景建模算法本身提速（涉及算法改动，需额外单独授权，不在阶段7B范围内）；
   c. 用多进程替代多线程以绕开GIL，可能提升detect阶段并行效率；
   d. 180对数据样本量小，线程启动/收尾开销占比不小，更大样本下实际吞吐可能更接近理论值，需验证。
2. **长期稳定性完全未测试**：仅180对数据（约0.58秒原始时长）做过短批测试和3轮同批重复压力测试，不构成长时间连续运行（数小时/数天级）的稳定性证明。`stage7_report.json`/`performance_optimization_report.json`均明确标注`tested: false`。
3. **检测/跟踪参数未针对新数据集标定**：`config/pipeline.yaml`里的阈值（thresh=20、min_area=10等）和跟踪门控参数（gate_dx_max/gate_dy_max等）沿用老数据集(cama/camb)的经验值，未用A/B-Capture_085820这批新数据或人工标注验证集重新标定，导致检出偏稀疏（A:24个/B:14个，180帧）。
4. **双相机标定文件不存在**：`config/pipeline.yaml`的`stereo.calibration.path`（`config/stereo_calibration.json`）指向一个**故意不存在**的文件——这是有意为之，不是遗漏，因为项目没有真实的双相机内外参/基础矩阵标定数据，禁止虚构。一旦有真实标定文件，只需要把它放到该路径，代码不用改。
5. **由此连锁未解决**：阶段4双路匹配恒0个已验证匹配；阶段6`stereo_common_volume`统计恒`concentration_status=unavailable`；阶段5测速仅1条低置信度轨迹；均属正常的"诚实降级"，不是bug。
6. **像素-毫米标定不存在**：所有粒径/速度只能是像素单位（px, px/s），任何mm/s、m/s、正式雨滴谱、雨强字段都**不存在**（不是留空，是代码结构上就没有这些字段）。

---

## 5. 必须遵守的时间戳/标定/观测体积约束（硬性红线，任何后续修改都不能违反）

### 时间戳
- **A路轨迹/测速一律用`a_device_relative_us`；B路一律用`b_device_relative_us`**（单位微秒，换算成秒需除以1e6）。
- **禁止使用**：文件名里的毫秒（如`frame_000_20260922_085820_120.tif`里的`120`）、`host_receive`相关时间字段、`pair_wall_datetime_ms`、帧号除以固定帧率。
- `capture_metadata.json`确认：中位帧周期3200μs（对应312.5帧/秒），曝光175μs——这两个是**实测值**，可以在文档/注释里引用，但不能替代`a/b_device_relative_us`参与任何计算。
- A、B两路目录下的`capture_metadata.json`与`capture_timing.csv`内容**字节级(sha256)相同**（已在阶段1验证），代码约定统一以`dir_a`（A路目录）下的文件为主记录来源，避免重复读取解析B路副本。

### 标定
- **没有双相机内外参/基础矩阵标定文件**，任何时候都**不得虚构**这类参数去"让代码跑通"。`EpipolarGate`在`calibration=None`时必须让所有候选判定不通过，原因固定标记`calibration_unavailable`，不得计算出一个"看似正常"的极线距离。
- **固定曝光延迟未标定**：A、B设备时钟零点独立。即使未来标定文件就位，只要固定曝光延迟这一项仍未标定，所有双路匹配结果依然要保持低置信度标记（`fixed_exposure_delay_calibrated`配置项当前为`false`，且不应该被随意改成`true`除非真的完成了曝光延迟标定）。
- **像素-毫米标定不存在**：`spatial_calibration_status`在所有相关输出（`VelocityFitResult`、`DropSizeDistributionResult`、`SingleCameraProxyStat`、`StereoCommonVolumeStat`）中默认恒为`"spatial_calibration_unavailable"`，该字段是从配置读取的（不是硬编码常量），但当前配置值就是"未标定"，不应该无凭据地改成"validated"。

### 观测体积
- **590.4 cm³（=0.0005904 m³）是双路共同观测体积的名义值，不是单路视场体积，且尚未经过实验标定**。
- `volume_status`默认恒为`"nominal_unvalidated"`（同样是从配置读取，非硬编码，但当前就是未标定状态）。
- **两条统计产出线必须严格分离，不得互相替代**：
  - `single_camera_proxy`（`rain/measurement/window_statistics.py`）：单路（A或B单独）的非正式调试统计，`is_single_camera_proxy=True`、`usable_for_final_concentration=False`恒定。
  - `stereo_common_volume`（`rain/measurement/stereo_common_volume_statistics.py`）：590.4cm³真正对应的统计，N**只能**来自阶段4双路匹配成功(`status="matched"`)的目标数，**禁止**用A路或B路单独目标数（甚至两者的和/最小值）替代。当前因无标定，此统计线的`concentration_status`恒为`"unavailable"`（不是"浓度为0"）。
- **不输出雨强(rain rate)**：需要同时具备可靠物理速度（阶段5当前仅1条低置信度轨迹）、有效观测体积（未标定）、毫米粒径（未标定）三者，目前一个都不满足，代码里**没有**任何`rain_rate`/`rainfall_intensity`相关字段（已用结构性测试`test_no_module_outputs_mm_or_rain_rate_fields`验证）。

---

## 6. 不得修改的原Demo和原始数据路径

### 原始数据（只读，禁止移动/删除/修改/重命名）
```
D:\毕设\数据\A-Capture_085820         阶段1-7B使用的主数据集（A路，180帧TIF+3个csv/json）
D:\毕设\数据\B-Capture_085820         阶段1-7B使用的主数据集（B路，180帧TIF+3个csv/json）
D:\毕设\数据\cama, camb               老Demo(tools/demo.py)使用的历史数据集，本项目阶段1-7B未使用，但同样禁止改动
D:\毕设\数据\bja, bjb                 老Demo的背景参考数据集，同样禁止改动
```
每个阶段完成后均用文件mtime、size、文件数三项核对确认原始数据未被触碰（约定俗成的验证方式，之后的会话应继续沿用）。

### 原Demo文件（禁止修改一个字节，已用`git diff`全程核实为空）
```
rain/interfaces.py
rain/types.py
rain/sink.py
rain/io_index.py
rain/imageio.py
rain/registry.py 的"原有五张表"部分（READERS/BACKGROUNDS/DETECTORS/TRACKERS/STEREO及其register/build函数——
                  新增的~20张子模块表是纯增量，不影响这五张老表）
rain/detectors/simple_threshold.py
rain/trackers/gravity_gate.py
rain/trackers/nn_gate.py
rain/background/batch_quantile.py
rain/background/external_ref.py
rain/readers/lzw_tif.py
rain/readers/raw_bin.py
config.yaml                          根目录，服务于tools/demo.py和runners/run_batch.py
tools/demo.py
tools/test_smoke.py
runners/run_batch.py
```

### 唯一允许的例外：新增import行（已逐一用git diff核实只有插入、没有删除或修改已有行）
```
rain/__init__.py, rain/registry.py（追加新表）, rain/detectors/__init__.py,
rain/readers/__init__.py, pyproject.toml（追加依赖）
```

---

## 快速恢复命令

```bash
cd "D:\Codex\毕设相关\rain"

# 确认原Demo和原始数据完好
git diff --stat rain/interfaces.py rain/types.py rain/sink.py rain/io_index.py rain/imageio.py \
  rain/detectors/simple_threshold.py rain/trackers/gravity_gate.py rain/trackers/nn_gate.py \
  rain/background/batch_quantile.py rain/background/external_ref.py rain/readers/lzw_tif.py \
  rain/readers/raw_bin.py config.yaml tools/demo.py tools/test_smoke.py runners/run_batch.py

# 跑全部测试
python -m pytest tests/ -q

# 依次跑各阶段（结果落在 results/stageN/ 下）
python pipeline/stage1_data_audit.py
python pipeline/stage2_detection.py
python pipeline/stage3_tracking.py
python pipeline/stage4_stereo_matching.py
python pipeline/stage5_velocity.py
python pipeline/stage6_statistics.py
python pipeline/stage7_performance.py
python pipeline/stage7b_optimization.py
```
