# Rain

双路图像降雨检测与轨迹分析项目。项目按“读取 → 背景建模 → 目标检测 → 轨迹跟踪 → 结果写出”组织，算法实现通过注册表按配置切换。

## 目录结构

```text
rain/
├── rain/                 # 核心 Python 包
│   ├── readers/          # 图像读取器（LZW TIF、RAW）
│   ├── background/       # 背景建模
│   ├── detectors/        # 雨滴检测
│   ├── trackers/         # 轨迹跟踪
│   ├── types.py          # Detection、Track 等数据契约
│   ├── interfaces.py     # 模块接口协议
│   ├── registry.py       # 注册表与配置工厂
│   ├── io_index.py       # 批次扫描和双路配对
│   ├── imageio.py        # Unicode 路径图像读写
│   └── sink.py           # CSV 和叠加图输出
├── runners/              # 批处理流程
├── tools/                # 演示和冒烟测试脚本
├── tests/                # 自动化测试
├── config.yaml           # 数据路径、算法和并发配置
├── demo_output/          # 演示运行产物
└── test_output/          # 测试运行产物
```

## 安装

建议使用 Python 3.10 或更高版本：

```bash
python -m pip install -e .
```

开发依赖（包括测试工具）：

```bash
python -m pip install -e ".[dev]"
```

## 常用命令

运行契约测试：

```bash
python -m pytest tests/ -v
```

运行算法演示：

```bash
python tools/demo.py
```

运行冒烟测试：

```bash
python tools/test_smoke.py
```

运行前请先检查 `config.yaml` 中的双路数据目录和输出目录。当前输出单位是像素；项目不会在时间基准和标定未确认前推导物理速度、毫米粒径或雨强。

## 扩展算法

在对应子包中实现类，并使用 `rain.registry` 提供的注册装饰器登记名称。例如检测器使用 `@register_detector("name")`。然后在 `config.yaml` 的对应段落中填写该名称和参数，无需修改批处理主流程。
