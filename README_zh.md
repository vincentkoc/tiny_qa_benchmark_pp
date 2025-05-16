\
> [!NOTE]
> 注意：本文档是机器翻译版本，可能存在不准确之处。欢迎您贡献并改进翻译！

<!-- SPDX-License-Identifier: Apache-2.0 OR CC BY 4.0 OR other -->
<div align="center"><b><a href="README.md">English</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a></b></div>

<h1 align="center" style="border: none">
    <div style="border: none">
        <!-- 如果您有徽标，可以在此处添加。例如：
        <a href="YOUR_PROJECT_LINK"><picture>
            <source media="(prefers-color-scheme: dark)" srcset="PATH_TO_DARK_LOGO.svg">
            <source media="(prefers-color-scheme: light)" srcset="PATH_TO_LIGHT_LOGO.svg">
            <img alt="项目徽标" src="PATH_TO_LIGHT_LOGO.svg" width="200" />
        </picture></a>
        <br>
        -->
        Tiny QA Benchmark++ (TQB++)
    </div>
</h1>

<p align="center">
一个超轻量级的评估数据集和合成生成器 <br>可在几秒钟内暴露关键的 LLM 故障，非常适合 CI/CD 和 LLMOps。
</p>

<div align="center">
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><img alt="PyPI 版本" src="https://img.shields.io/pypi/v/tinyqabenchmarkpp"></a>
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp/blob/main/LICENSE"><img alt="许可证" src="https://img.shields.io/badge/Apache-2.0-green"></a>
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><img alt="Hugging Face 数据集" src="https://img.shields.io/badge/🤗%20Dataset-Tiny%20QA%20Benchmark%2B%2B-blue"></a>
    <!-- 如果您配置了 CI，请考虑添加 GitHub Actions 工作流徽章 -->
    <!-- 例如：<a href="YOUR_WORKFLOW_LINK"><img alt="构建状态" src="YOUR_WORKFLOW_BADGE_SVG_LINK"></a> -->
</div>

<p align="center">
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp"><b>GitHub</b></a> •
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><b>Hugging Face 数据集</b></a> •
    <!-- 论文可用后链接 -->
    <!-- <a href="#"><b>论文 (链接即将推出)</b></a> • -->
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><b>PyPI</b></a>
</p>

<hr>
<!-- 可选：如果您有项目缩略图，可以在此处添加 -->
<!-- <p align="center"><img alt="TQB++ 缩略图" src="path/to/your/thumbnail.png" width="700"></p> -->

**Tiny QA Benchmark++ (TQB++)** 是一个超轻量级的评估套件和 Python 包，旨在在几秒钟内暴露大型语言模型 (LLM) 系统中的关键故障。它充当 LLM 软件单元测试，非常适合 CI/CD 快速检查、提示工程以及现代 LLMOps 中的持续质量保证，可与现有的 LLM 评估工具（如 [Opik](https://github.com/comet-ml/opik/)）一起使用。

本存储库包含论文 *Tiny QA Benchmark++: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests* 的官方实现和合成数据集。

**论文：** (详情和预印本链接将在发布后在此处提供)

- **Hugging Face Hub：** [datasets/vincentkoc/tiny_qa_benchmark_pp](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp)
- **GitHub 存储库：** [vincentkoc/tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp)

## 主要功能

*   **不可变的黄金标准核心：** 一个包含 52 个人工制作的英语问答 (QA) 数据集 (`core_en`)，用于确定性回归测试，源自早期的 [datasets/vincentkoc/tiny_qa_benchmark](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark)。
*   **综合定制工具包：** 一个 Python 脚本 (`tools/generator`)，使用 LiteLLM 按需为任何语言、主题或难度生成定制的微基准。
*   **标准化元数据：** 以 Croissant JSON-LD 格式 (`metadata/`) 打包的工件，以便工具和搜索引擎发现和自动加载。
*   **开放科学：** 所有代码（生成器、评估脚本）和核心英语数据集均在 Apache-2.0 许可下发布。综合生成的数据包具有自定义的仅评估许可。
*   **LLMOps 对齐：** 旨在轻松集成到 CI/CD 管道、提示工程工作流、跨语言漂移检测和可观察性仪表板中。
*   **多语言包：** 为多种语言（包括英语、法语、西班牙语、葡萄牙语、德语、中文、日语、土耳其语、阿拉伯语和俄语）预构建的包。

## 使用 `tinyqabenchmarkpp` Python 包

TQB++ 的核心综合生成功能以 Python 包 `tinyqabenchmarkpp` 的形式提供，可从 PyPI 安装。

### 安装

```bash
pip install tinyqabenchmarkpp
```

(注意：确保您已安装 Python 3.8+ 和 pip。PyPI 上的确切包名称可能会有所不同；如果此命令不起作用，请检查官方 [PyPI 项目页面](https://pypi.org/project/tinyqabenchmarkpp/) 获取正确的包名称。)

### 通过 CLI 生成合成数据集

安装后，您可以使用 `tinyqabenchmarkpp` 命令 (或 `python -m tinyqabenchmarkpp.generate`) 创建自定义 QA 数据集。

**示例：**
```bash
tinyqabenchmarkpp --num 10 --languages "en,es" --categories "science" --output-file "./science_pack.jsonl"
```

这将生成一个包含 10 个英语和西班牙语科学问题的小型数据包。

有关所有可用参数 (如 `--model`、`--context`、`--difficulty` 等) 的详细说明、高级用法以及不同 LLM 提供程序 (OpenAI、OpenRouter、Ollama) 的示例，请参阅 **[生成器工具包 README](tools/generator/README.md)** 或运行 `tinyqabenchmarkpp --help`。

虽然 `tinyqabenchmarkpp` 包专注于数据集*生成*，但 TQB++ 项目还提供预生成的数据集和评估工具，如下所述。

## 使用 Hugging Face `datasets` 加载数据集

TQB++ 数据集可在 Hugging Face Hub 上找到，并可以使用 `datasets` 库轻松加载。这是访问数据的推荐方法。

```python
from datasets import load_dataset, get_dataset_config_names

# 发现可用的数据集配置 (例如 core_en, pack_fr_40 等)
configs = get_dataset_config_names("vincentkoc/tiny_qa_benchmark_pp")
print(f"可用配置: {configs}")

# 加载核心英语数据集 (假设 \'core_en\' 是一个配置)
if "core_en" in configs:
    core_dataset = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="core_en", split="train")
    print(f"\\n从 core_en 加载了 {len(core_dataset)} 个示例:")
    # print(core_dataset[0]) # 打印第一个示例
else:
    print("\\n未找到 \'core_en\' 配置。")

# 加载特定的合成包 (例如法语包)
# 将 \'pack_fr_40\' 替换为 `configs` 列表中的实际配置名称
example_pack_name = "pack_fr_40" # 或其他有效的配置名称
if example_pack_name in configs:
    synthetic_pack = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name=example_pack_name, split="train")
    print(f"\\n从 {example_pack_name} 加载了 {len(synthetic_pack)} 个示例:")
    # print(synthetic_pack[0]) # 打印第一个示例
else:
    print(f"\\n未找到 \'{example_pack_name}\' 配置。请从可用配置中选择。")

```

有关数据集的更多详细信息，包括其结构和特定许可证，请参阅 `data/` 目录中的 README 文件 (即 `data/README.md`, `data/core_en/README.md`, 和 `data/packs/README.md`)。

## 存储库结构

*   `data/`: 包含 QA 数据集。
    *   `core_en/`: 原始的 52 个人工制作的英语核心数据集。
    *   `packs/`: 综合生成的多语言和主题数据集包。
*   `tools/`: 包含用于数据集生成和评估的脚本。
    *   `generator/`: 综合 QA 数据集生成器。
    *   `eval/`: 用于根据 TQB++ 数据集评估模型的脚本和实用程序。
*   `paper/`: 研究论文的 LaTeX 源代码和相关文件。
*   `metadata/`: 数据集的 Croissant JSON-LD 元数据文件。
*   `LICENSE`: 代码库的主要许可证 (Apache-2.0)。
*   `LICENCE.data_packs.md`: 综合生成的数据包的自定义许可证。
*   `LICENCE.paper.md`: 论文内容的许可证。

## 使用场景

TQB++ 专为各种 LLMOps 和评估工作流而设计：

*   **CI/CD 管道测试：** 与 LLM 测试工具一起用作 LLM 服务的单元测试以捕获回归。
*   **提示工程和代理开发：** 在迭代提示或代理设计时获得快速反馈。
*   **评估工具集成：** 编码为 OpenAI Evals YAML 或 Opik 数据集以进行仪表板跟踪。
*   **跨语言漂移检测：** 使用多语言 TQB++ 包监控本地化回归。
*   **自适应测试：** 动态合成针对特定功能或数据漂移定制的微基准。
*   **监控微调动态：** 跟踪微调过程中的知识侵蚀或意外能力变化。

## 引文

如果您在研究或工作中使用 TQB++，请引用原始 TQB 和 TQB++ 论文：

```bibtex
% 此合成数据集和生成器
@misc{koctinyqabenchmarkpp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark++ (TQB++) 数据集和工具包},
  year         = {2025},
  publisher    = {Hugging Face & GitHub},
  doi          = {10.57967/hf/5531},
  howpublished = {\\url{https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp}},
  note         = {另请参阅: \\url{https://github.com/vincentkoc/tiny_qa_benchmark_pp}}
}

% 原始 core_en.json (52 个英语条目)
@misc{koctinyqabenchmark_original,
  author       = {Vincent Koc},
  title        = {tiny_qa_benchmark},
  year         = {2025},
  publisher    = {Hugging Face},
  journal      = {Hugging Face Hub},
  doi          = {10.57967/hf/5417},
  url          = {https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark}
}
```

<!-- % TQB++ JMLR 论文引文占位符 - 可用时更新
@article{koc2025tqb_pp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark$^{++}$: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests},
  journal      = {Journal of Machine Learning Research (待定)},
  year         = {2025},
  volume       = {XX},
  number       = {X},
  pages        = {X-XX},
  url          = {http://jmlr.org/papers/vXX/koc25a.html} % 示例 URL
} -->

## 许可证
本存储库中的代码（包括生成器和评估脚本）以及 `data/core_en` 数据集和任何其他未提及许可证的内容均在 Apache License 2.0 下获得许可。有关详细信息，请参阅 `LICENSE` 文件。

`data/packs/` 中的综合生成的数据集包在自定义的"仅评估、非商业、禁止衍生"许可证下分发。有关详细信息，请参阅 `LICENCE.data_packs.md`。

`metadata/` 中的 Croissant JSON-LD 元数据文件可在 CC0-1.0 下获得。

`paper/` 中的论文内容受其自己的许可条款约束，详见 `LICENCE.paper.md`。 