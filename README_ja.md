> [!NOTE]
> 注意：このドキュメントは機械翻訳されたものであり、不正確な点が含まれている可能性があります。翻訳の改善にご協力ください！

<!-- SPDX-License-Identifier: Apache-2.0 OR CC BY 4.0 OR other -->
<div align="center"><b><a href="README.md">English</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a></b></div>

<h1 align="center" style="border: none">
    <div style="border: none">
        <!-- ロゴをお持ちの場合は、ここに追加できます。例：
        <a href="YOUR_PROJECT_LINK"><picture>
            <source media="(prefers-color-scheme: dark)" srcset="PATH_TO_DARK_LOGO.svg">
            <source media="(prefers-color-scheme: light)" srcset="PATH_TO_LIGHT_LOGO.svg">
            <img alt="プロジェクトロゴ" src="PATH_TO_LIGHT_LOGO.svg" width="200" />
        </picture></a>
        <br>
        -->
        Tiny QA Benchmark++ (TQB++)
    </div>
</h1>

<p align="center">
超軽量の評価データセットと合成ジェネレータ<br>CI/CD および LLMOps に最適な、LLM の重大な障害を数秒で検出します。
</p>

<div align="center">
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><img alt="PyPI バージョン" src="https://img.shields.io/pypi/v/tinyqabenchmarkpp"></a>
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp/blob/main/LICENSE"><img alt="ライセンス" src="https://img.shields.io/badge/Apache-2.0-green"></a>
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><img alt="Hugging Face データセット" src="https://img.shields.io/badge/🤗%20Dataset-Tiny%20QA%20Benchmark%2B%2B-blue"></a>
    <a href="https://arxiv.org/abs/2505.12058"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.12058-b31b1b.svg"></a>
    <!-- CI を設定している場合は、GitHub Actions ワークフローバッジの追加を検討してください -->
    <!-- 例：<a href="YOUR_WORKFLOW_LINK"><img alt="ビルドステータス" src="YOUR_WORKFLOW_BADGE_SVG_LINK"></a> -->
</div>

<p align="center">
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp"><b>GitHub</b></a> •
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><b>Hugging Face データセット</b></a> •
    <a href="https://arxiv.org/abs/2505.12058"><b>論文 (arXiv:2505.12058)</b></a> •
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><b>PyPI</b></a>
</p>

<hr>
<!-- オプション：プロジェクトのサムネイル画像をお持ちの場合は、ここに追加できます -->
<!-- <p align="center"><img alt="TQB++ サムネイル" src="path/to/your/thumbnail.png" width="700"></p> -->

**Tiny QA Benchmark++ (TQB++)** は、大規模言語モデル (LLM) システムの重大な障害を数秒で検出するように設計された超軽量の評価スイートおよび Python パッケージです。これは LLM ソフトウェアの単体テストとして機能し、CI/CD の迅速なチェック、プロンプトエンジニアリング、および最新の LLMOps における継続的な品質保証に最適であり、[Opik](https://github.com/comet-ml/opik/) などの既存の LLM 評価ツールと並行して使用できます。

このリポジトリには、論文 *Tiny QA Benchmark++: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests* の公式実装と合成データセットが含まれています。

**論文：** (詳細はプレプリントのリンクとともに公開され次第ここに記載されます)

- **Hugging Face Hub：** [datasets/vincentkoc/tiny_qa_benchmark_pp](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp)
- **GitHub リポジトリ：** [vincentkoc/tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp)

## 主な機能

*   **不変のゴールドスタンダードコア：** 初期の [datasets/vincentkoc/tiny_qa_benchmark](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark) からの、決定論的リグレッションテスト用の 52 項目の手作り英語質疑応答 (QA) データセット (`core_en`)。
*   **合成カスタマイズツールキット：** LiteLLM を使用して、あらゆる言語、トピック、または難易度に合わせてオンデマンドで特注のマイクロベンチマークを生成する Python スクリプト (`tools/generator`)。
*   **標準化されたメタデータ：** ツールや検索エンジンによる検出と自動読み込みのために、Croissant JSON-LD 形式 (`metadata/`) でパッケージ化されたアーティファクト。
*   **オープンサイエンス：** すべてのコード (ジェネレータ、評価スクリプト) とコア英語データセットは Apache-2.0 ライセンスの下でリリースされています。合成的に生成されたデータパックには、カスタムの評価専用ライセンスがあります。
*   **LLMOps との連携：** CI/CD パイプライン、プロンプトエンジニアリングワークフロー、多言語ドリフト検出、および可観測性ダッシュボードへの容易な統合のために設計されています。
*   **多言語パック：** 英語、フランス語、スペイン語、ポルトガル語、ドイツ語、中国語、日本語、トルコ語、アラビア語、ロシア語など、多数の言語用の事前構築済みパック。

## `tinyqabenchmarkpp` Python パッケージの使用

TQB++ のコア合成生成機能は、PyPI からインストールできる Python パッケージ `tinyqabenchmarkpp` として利用できます。

### インストール

```bash
pip install tinyqabenchmarkpp
```

(注意：Python 3.8 以降と pip がインストールされていることを確認してください。PyPI 上の正確なパッケージ名は異なる場合があります。このコマンドが機能しない場合は、公式の [PyPI プロジェクトページ](https://pypi.org/project/tinyqabenchmarkpp/) で正しいパッケージ名を確認してください。)

### CLI を介した合成データセットの生成

インストール後、`tinyqabenchmarkpp` コマンド (または `python -m tinyqabenchmarkpp.generate`) を使用してカスタム QA データセットを作成できます。

**例：**
```bash
tinyqabenchmarkpp --num 10 --languages "en,es" --categories "science" --output-file "./science_pack.jsonl"
```

これにより、10 個の英語とスペイン語の科学に関する質問の小さなパックが生成されます。

利用可能なすべてのパラメータ ( `--model`、`--context`、`--difficulty` など)、高度な使用法、およびさまざまな LLM プロバイダー (OpenAI、OpenRouter、Ollama) の例に関する詳細な手順については、**[ジェネレータツールキット README](tools/generator/README.md)** を参照するか、`tinyqabenchmarkpp --help` を実行してください。

`tinyqabenchmarkpp` パッケージはデータセットの*生成*に重点を置いていますが、TQB++ プロジェクトでは、以下で説明するように、事前に生成されたデータセットと評価ツールも提供しています。

## Hugging Face `datasets` を使用したデータセットの読み込み

TQB++ データセットは Hugging Face Hub で利用可能であり、`datasets` ライブラリを使用して簡単に読み込むことができます。これはデータにアクセスするための推奨される方法です。

```python
from datasets import load_dataset, get_dataset_config_names

# 利用可能なデータセット構成を検出 (例：core_en, pack_fr_40 など)
configs = get_dataset_config_names("vincentkoc/tiny_qa_benchmark_pp")
print(f"利用可能な構成: {configs}")

# コア英語データセットを読み込む (\'core_en\' が構成であると仮定)
if "core_en" in configs:
    core_dataset = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="core_en", split="train")
    print(f"\\ncore_en から {len(core_dataset)} 個のサンプルを読み込みました:")
    # print(core_dataset[0]) # 最初のサンプルを印刷
else:
    print("\\n\'core_en\' 構成が見つかりません。")

# 特定の合成パックを読み込む (例：フランス語パック)
# \'pack_fr_40\' を `configs` リストの実際の構成名に置き換えます
example_pack_name = "pack_fr_40" # または他の有効な構成名
if example_pack_name in configs:
    synthetic_pack = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name=example_pack_name, split="train")
    print(f"\\n{example_pack_name} から {len(synthetic_pack)} 個のサンプルを読み込みました:")
    # print(synthetic_pack[0]) # 最初のサンプルを印刷
else:
    print(f"\\n\'{example_pack_name}\' 構成が見つかりません。利用可能な構成から選択してください。")

```

データセットの構造や特定のライセンスなど、データセットに関する詳細については、`data/` ディレクトリ内の README ファイル (つまり `data/README.md`, `data/core_en/README.md`, および `data/packs/README.md`) を参照してください。

## リポジトリ構造

*   `data/`: QA データセットが含まれています。
    *   `core_en/`: 元の 52 項目の手作り英語コアデータセット。
    *   `packs/`: 合成的に生成された多言語およびトピックデータセットパック。
*   `tools/`: データセットの生成と評価のためのスクリプトが含まれています。
    *   `generator/`: 合成 QA データセットジェネレータ。
    *   `eval/`: TQB++ データセットに対してモデルを評価するためのスクリプトとユーティリティ。
*   `paper/`: 研究論文の LaTeX ソースと関連ファイル。
*   `metadata/`: データセットの Croissant JSON-LD メタデータファイル。
*   `LICENSE`: コードベースのメインライセンス (Apache-2.0)。
*   `LICENCE.data_packs.md`: 合成的に生成されたデータパックのカスタムライセンス。
*   `LICENCE.paper.md`: 論文コンテンツのライセンス。

## 使用シナリオ

TQB++ は、さまざまな LLMOps および評価ワークフロー向けに設計されています。

*   **CI/CD パイプラインテスト：** LLM テストツールとともに LLM サービスの単体テストとして使用して、リグレッションを検出します。
*   **プロンプトエンジニアリングとエージェント開発：** プロンプトまたはエージェントの設計を反復処理する際に、迅速なフィードバックを得ます。
*   **評価ハーネス統合：** 評価ハーネスとシームレスに使用できるように設計されています。OpenAI Evals YAML（`intergrations/openai-evals/README.md` を参照）またはOpikデータセットとしてエンコードし、ダッシュボードでの追跡と堅牢なLLM評価を実現します。`intergrations/`フォルダには、利用可能な標準サポートに関する詳細が記載されています。
*   **多言語ドリフト検出：** 多言語 TQB++ パックを使用して、ローカリゼーションのリグレッションを監視します。
*   **適応型テスト：** 特定の機能またはデータドリフトに合わせて、オンザフライでマイクロベンチマークを合成します。
*   **微調整ダイナミクスの監視：** 微調整中の知識の侵食または意図しない能力の変化を追跡します。

## 引用

研究または仕事で TQB++ を使用する場合は、元の TQB と TQB++ 論文を引用してください。

```bibtex
% この合成データセットとジェネレータ
@misc{koctinyqabenchmarkpp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark++ (TQB++) データセットとツールキット},
  year         = {2025},
  publisher    = {Hugging Face & GitHub},
  doi          = {10.57967/hf/5531},
  howpublished = {\\url{https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp}},
  note         = {参照: \\url{https://github.com/vincentkoc/tiny_qa_benchmark_pp}}
}

% TQB++ 論文
@misc{koc2025tinyqabenchmarkultralightweight,
      title={Tiny QA Benchmark++: Ultra-Lightweight, Synthetic Multilingual Dataset Generation & Smoke-Tests for Continuous LLM Evaluation}, 
      author={Vincent Koc},
      year={2025},
      eprint={2505.12058},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.12058}
}

% 元の core_en.json (英語で 52 項目)
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

## ライセンス
このリポジトリのコード (ジェネレータと評価スクリプトを含む)、`data/core_en` データセット、およびライセンスが記載されていないその他のものは、Apache License 2.0 の下でライセンスされています。詳細については、`LICENSE` ファイルを参照してください。

`data/packs/` の合成的に生成されたデータセットパックは、カスタムの「評価専用、非商用、派生物禁止」ライセンスの下で配布されています。詳細については、`LICENCE.data_packs.md` を参照してください。

`metadata/` の Croissant JSON-LD メタデータファイルは、CC0-1.0 の下で利用可能です。

`paper/` の論文コンテンツは、独自のライセンス条項に従います。詳細は `LICENCE.paper.md` を参照してください。 