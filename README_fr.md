\
> [!NOTE]
> Remarque : Ce document est une version traduite automatiquement et peut contenir des inexactitudes. Vos contributions pour améliorer la traduction sont les bienvenues !

<!-- SPDX-License-Identifier: Apache-2.0 OR CC BY 4.0 OR other -->
<div align="center"><b><a href="README.md">English</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a></b></div>

<h1 align="center" style="border: none">
    <div style="border: none">
        <!-- Si vous avez un logo, vous pouvez l'ajouter ici. Exemple :
        <a href="YOUR_PROJECT_LINK"><picture>
            <source media="(prefers-color-scheme: dark)" srcset="PATH_TO_DARK_LOGO.svg">
            <source media="(prefers-color-scheme: light)" srcset="PATH_TO_LIGHT_LOGO.svg">
            <img alt="Logo du Projet" src="PATH_TO_LIGHT_LOGO.svg" width="200" />
        </picture></a>
        <br>
        -->
        Tiny QA Benchmark++ (TQB++)
    </div>
</h1>

<p align="center">
Un jeu de données d'évaluation ultra-léger et un générateur synthétique <br>pour exposer les défaillances critiques des LLM en quelques secondes, idéal pour CI/CD et LLMOps.
</p>

<div align="center">
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><img alt="Version PyPI" src="https://img.shields.io/pypi/v/tinyqabenchmarkpp"></a>
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp/blob/main/LICENSE"><img alt="Licence" src="https://img.shields.io/badge/Apache-2.0-green"></a>
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><img alt="Jeu de données Hugging Face" src="https://img.shields.io/badge/🤗%20Dataset-Tiny%20QA%20Benchmark%2B%2B-blue"></a>
    <a href="https://arxiv.org/abs/2505.12058"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.12058-b31b1b.svg"></a>
    <!-- Envisagez d'ajouter un badge de workflow GitHub Actions si vous avez configuré la CI -->
    <!-- ex: <a href="YOUR_WORKFLOW_LINK"><img alt="État de la construction" src="YOUR_WORKFLOW_BADGE_SVG_LINK"></a> -->
</div>

<p align="center">
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp"><b>GitHub</b></a> •
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><b>Jeu de données Hugging Face</b></a> •
    <a href="https://arxiv.org/abs/2505.12058"><b>Article (arXiv:2505.12058)</b></a> •
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><b>PyPI</b></a>
</p>

<hr>
<!-- Optionnel : Si vous avez une miniature de projet, vous pouvez l'ajouter ici -->
<!-- <p align="center"><img alt="Miniature TQB++" src="path/to/your/thumbnail.png" width="700"></p> -->

**Tiny QA Benchmark++ (TQB++)** est une suite d'évaluation ultra-légère et un package Python conçu pour exposer les défaillances critiques des systèmes de grands modèles de langage (LLM) en quelques secondes. Il sert de tests unitaires logiciels pour LLM, idéal pour les vérifications rapides CI/CD, l'ingénierie des prompts et l'assurance qualité continue dans les LLMOps modernes, à utiliser parallèlement aux outils d'évaluation LLM existants tels que [Opik](https://github.com/comet-ml/opik/).

Ce dépôt contient l'implémentation officielle et les jeux de données synthétiques pour l'article : *Tiny QA Benchmark++: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests*.

**Article :** (Les détails et le lien vers la prépublication seront fournis ici une fois publiés)

- **Hugging Face Hub :** [datasets/vincentkoc/tiny_qa_benchmark_pp](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp)
- **Dépôt GitHub :** [vincentkoc/tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp)

## Fonctionnalités Clés

*   **Noyau Standard Or Immuable :** Un jeu de données de questions-réponses (QA) en anglais de 52 éléments élaboré à la main (`core_en`) pour les tests de régression déterministes, issu de [datasets/vincentkoc/tiny_qa_benchmark](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark).
*   **Boîte à Outils de Personnalisation Synthétique :** Un script Python (`tools/generator`) utilisant LiteLLM pour générer des micro-benchmarks sur mesure à la demande pour n'importe quelle langue, sujet ou difficulté.
*   **Métadonnées Normalisées :** Artefacts empaquetés au format Croissant JSON-LD (`metadata/`) pour la découvrabilité et le chargement automatique par les outils et les moteurs de recherche.
*   **Science Ouverte :** Tout le code (générateur, scripts d'évaluation) et le jeu de données principal en anglais sont publiés sous la licence Apache-2.0. Les packs de données générés synthétiquement ont une licence personnalisée à des fins d'évaluation uniquement.
*   **Alignement LLMOps :** Conçu pour une intégration facile dans les pipelines CI/CD, les flux de travail d'ingénierie des prompts, la détection de dérive multilingue et les tableaux de bord d'observabilité.
*   **Packs Multilingues :** Packs préconstruits pour de nombreuses langues, dont l'anglais, le français, l'espagnol, le portugais, l'allemand, le chinois, le japonais, le turc, l'arabe et le russe.

## Utilisation du Package Python `tinyqabenchmarkpp`

Les capacités principales de génération synthétique de TQB++ sont disponibles sous forme de package Python, `tinyqabenchmarkpp`, installable depuis PyPI.

### Installation

```bash
pip install tinyqabenchmarkpp
```

(Remarque : Assurez-vous que Python 3.8+ et pip sont installés. Le nom exact du package sur PyPI peut varier ; veuillez consulter la [page officielle du projet PyPI](https://pypi.org/project/tinyqabenchmarkpp/) pour le nom correct du package si cette commande ne fonctionne pas.)

### Génération de Jeux de Données Synthétiques via CLI

Une fois installé, vous pouvez utiliser la commande `tinyqabenchmarkpp` (ou `python -m tinyqabenchmarkpp.generate`) pour créer des jeux de données QA personnalisés.

**Exemple :**
```bash
tinyqabenchmarkpp --num 10 --languages "en,es" --categories "science" --output-file "./science_pack.jsonl"
```

Cela générera un petit pack de 10 questions scientifiques en anglais et en espagnol.

Pour des instructions détaillées sur tous les paramètres disponibles (comme `--model`, `--context`, `--difficulty`, etc.), l'utilisation avancée et des exemples pour différents fournisseurs LLM (OpenAI, OpenRouter, Ollama), veuillez consulter le **[README de la Boîte à Outils du Générateur](tools/generator/README.md)** ou exécuter `tinyqabenchmarkpp --help`.

Bien que le package `tinyqabenchmarkpp` se concentre sur la *génération* de jeux de données, le projet TQB++ fournit également des jeux de données pré-générés et des outils d'évaluation, comme décrit ci-dessous.

## Chargement de Jeux de Données avec `datasets` de Hugging Face

Les jeux de données TQB++ sont disponibles sur le Hugging Face Hub et могут быть легко загружены с помощью библиотеки `datasets`. C'est la méthode recommandée pour accéder aux données.

```python
from datasets import load_dataset, get_dataset_config_names

# Découvrir les configurations de jeux de données disponibles (ex. core_en, pack_fr_40, etc.)
configs = get_dataset_config_names("vincentkoc/tiny_qa_benchmark_pp")
print(f"Configurations disponibles : {configs}")

# Charger le jeu de données principal en anglais (en supposant que \'core_en\' est une configuration)
if "core_en" in configs:
    core_dataset = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="core_en", split="train")
    print(f"\\nChargé {len(core_dataset)} exemples depuis core_en :")
    # print(core_dataset[0]) # Imprimer le premier exemple
else:
    print("\\nConfiguration \'core_en\' non trouvée.")

# Charger un pack synthétique spécifique (ex. un pack en français)
# Remplacez \'pack_fr_40\' par un nom de configuration réel de la liste `configs`
example_pack_name = "pack_fr_40" # ou un autre nom de configuration valide
if example_pack_name in configs:
    synthetic_pack = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name=example_pack_name, split="train")
    print(f"\\nChargé {len(synthetic_pack)} exemples depuis {example_pack_name} :")
    # print(synthetic_pack[0]) # Imprimer le premier exemple
else:
    print(f"\\nConfiguration \'{example_pack_name}\' non trouvée. Veuillez choisir parmi les configurations disponibles.")

```

Pour plus d'informations détaillées sur les jeux de données, y compris leur structure et leurs licences spécifiques, veuillez consulter les fichiers README dans le répertoire `data/` (c.-à-d. `data/README.md`, `data/core_en/README.md` et `data/packs/README.md`).

## Structure du Dépôt

*   `data/` : Contient les jeux de données QA.
    *   `core_en/` : Le jeu de données original de 52 éléments en anglais élaboré à la main.
    *   `packs/` : Packs de données multilingues et thématiques générés synthétiquement.
*   `tools/` : Contient des scripts pour la génération et l'évaluation de jeux de données.
    *   `generator/` : Le générateur de jeux de données QA synthétiques.
    *   `eval/` : Scripts et utilitaires pour évaluer les modèles par rapport aux jeux de données TQB++.
*   `paper/` : La source LaTeX et les fichiers associés pour l'article de recherche.
*   `metadata/` : Fichiers de métadonnées Croissant JSON-LD pour les jeux de données.
*   `LICENSE` : Licence principale pour la base de code (Apache-2.0).
*   `LICENCE.data_packs.md` : Licence personnalisée pour les packs de données générés synthétiquement.
*   `LICENCE.paper.md` : Licence pour le contenu de l'article.

## Scénarios d'Utilisation

TQB++ est conçu pour divers flux de travail LLMOps et d'évaluation :

*   **Tests de Pipeline CI/CD :** À utiliser comme test unitaire avec des outils de test LLM pour les services LLM afin de détecter les régressions.
*   **Ingénierie des Prompts et Développement d'Agents :** Obtenez un retour rapide lors de l'itération sur les prompts ou les conceptions d'agents.
*   **Intégration du Harnais d'Évaluation :** Conçu pour une utilisation transparente avec les harnais d'évaluation. Encodez-le en tant que YAML OpenAI Evals (voir `intergrations/openai-evals/README.md`) ou un jeu de données Opik pour le suivi du tableau de bord et une évaluation LLM robuste. Le dossier `intergrations/` fournit plus de détails sur le support prêt à l'emploi disponible.
*   **Détection de Dérive Multilingue :** Surveillez les régressions de localisation à l'aide des packs multilingues TQB++.
*   **Tests Adaptatifs :** Synthétisez de nouveaux micro-benchmarks à la volée adaptés à des fonctionnalités spécifiques ou à des dérives de données.
*   **Surveillance de la Dynamique de Fine-tuning :** Suivez l'érosion des connaissances ou les changements de capacité involontaires pendant le fine-tuning.

## Citation

Si vous utilisez TQB++ dans vos recherches ou votre travail, veuillez citer le TQB original et l'article TQB++ :

```bibtex
% Ce jeu de données synthétique et générateur
@misc{koctinyqabenchmarkpp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark++ (TQB++) Jeux de Données et Boîte à Outils},
  year         = {2025},
  publisher    = {Hugging Face & GitHub},
  doi          = {10.57967/hf/5531},
  howpublished = {\\url{https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp}},
  note         = {Voir aussi : \\url{https://github.com/vincentkoc/tiny_qa_benchmark_pp}}
}

% Article TQB++
@misc{koc2025tinyqabenchmarkultralightweight,
      title={Tiny QA Benchmark++: Ultra-Lightweight, Synthetic Multilingual Dataset Generation & Smoke-Tests for Continuous LLM Evaluation}, 
      author={Vincent Koc},
      year={2025},
      eprint={2505.12058},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.12058}
}

% core_en.json original (52 en en)
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

<!-- % Placeholder pour la citation de l'article TQB++ JMLR - à mettre à jour dès que disponible
@article{koc2025tqb_pp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark$^{++}$: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests},
  journal      = {Journal of Machine Learning Research (en attente)},
  year         = {2025},
  volume       = {XX},
  number       = {X},
  pages        = {X-XX},
  url          = {http://jmlr.org/papers/vXX/koc25a.html} % URL d'exemple
} -->

## Licence
Le code de ce dépôt (y compris le générateur et les scripts d'évaluation) ainsi que le jeu de données `data/core_en` et tout ce qui n'est pas mentionné avec une licence sont sous licence Apache License 2.0. Consultez le fichier `LICENSE` pour plus de détails.

Les packs de jeux de données générés synthétiquement dans `data/packs/` sont distribués sous une licence personnalisée "Évaluation Uniquement, Non Commercial, Sans Dérivés". Consultez `LICENCE.data_packs.md` pour plus de détails.

Les fichiers de métadonnées Croissant JSON-LD dans `metadata/` sont disponibles sous CC0-1.0.

Le contenu de l'article dans `paper/` est soumis à ses propres termes de licence, détaillés dans `LICENCE.paper.md`. 