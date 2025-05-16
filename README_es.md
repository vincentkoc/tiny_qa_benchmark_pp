\
> [!NOTE]
> Nota: Este documento es una versión traducida automáticamente y puede contener imprecisiones. ¡Agradecemos sus contribuciones para mejorar la traducción!

<!-- SPDX-License-Identifier: Apache-2.0 OR CC BY 4.0 OR other -->
<div align="center"><b><a href="README.md">English</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a></b></div>

<h1 align="center" style="border: none">
    <div style="border: none">
        <!-- Si tienes un logo, puedes añadirlo aquí. Ejemplo:
        <a href="YOUR_PROJECT_LINK"><picture>
            <source media="(prefers-color-scheme: dark)" srcset="PATH_TO_DARK_LOGO.svg">
            <source media="(prefers-color-scheme: light)" srcset="PATH_TO_LIGHT_LOGO.svg">
            <img alt="Logo del Proyecto" src="PATH_TO_LIGHT_LOGO.svg" width="200" />
        </picture></a>
        <br>
        -->
        Tiny QA Benchmark++ (TQB++)
    </div>
</h1>

<p align="center">
Un conjunto de datos de evaluación ultraligero y un generador sintético <br>para exponer fallos críticos de LLM en segundos, ideal para CI/CD y LLMOps.
</p>

<div align="center">
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><img alt="Versión de PyPI" src="https://img.shields.io/pypi/v/tinyqabenchmarkpp"></a>
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp/blob/main/LICENSE"><img alt="Licencia" src="https://img.shields.io/github/license/vincentkoc/tiny_qa_benchmark_pp"></a>
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><img alt="Conjunto de datos de Hugging Face" src="https://img.shields.io/badge/🤗%20Dataset-Tiny%20QA%20Benchmark%2B%2B-blue"></a>
    <!-- Considera añadir una insignia de flujo de trabajo de GitHub Actions si tienes CI configurada -->
    <!-- ej.: <a href="YOUR_WORKFLOW_LINK"><img alt="Estado de la compilación" src="YOUR_WORKFLOW_BADGE_SVG_LINK"></a> -->
</div>

<p align="center">
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp"><b>GitHub</b></a> •
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><b>Conjunto de datos de Hugging Face</b></a> •
    <!-- Enlace al artículo cuando esté disponible -->
    <!-- <a href="#"><b>Artículo (Enlace próximamente)</b></a> • -->
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><b>PyPI</b></a>
</p>

<hr>
<!-- Opcional: Si tienes una imagen en miniatura del proyecto, puedes añadirla aquí -->
<!-- <p align="center"><img alt="Miniatura de TQB++" src="path/to/your/thumbnail.png" width="700"></p> -->

**Tiny QA Benchmark++ (TQB++)** es un conjunto de evaluación ultraligero y un paquete de Python diseñado para exponer fallos críticos en sistemas de Modelos de Lenguaje Grandes (LLM) en cuestión de segundos. Sirve como pruebas unitarias de software para LLM, ideal para comprobaciones rápidas de CI/CD, ingeniería de prompts y garantía de calidad continua en LLMOps modernos, para ser utilizado junto con herramientas de evaluación de LLM existentes como [Opik](https://github.com/comet-ml/opik/).

Este repositorio contiene la implementación oficial y los conjuntos de datos sintéticos para el artículo: *Tiny QA Benchmark++: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests*.

**Artículo:** (Los detalles y el enlace a la preimpresión se proporcionarán aquí una vez publicados)

- **Hugging Face Hub:** [datasets/vincentkoc/tiny_qa_benchmark_pp](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp)
- **Repositorio de GitHub:** [vincentkoc/tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp)

## Características Principales

*   **Núcleo Inmutable Gold Standard:** Un conjunto de datos de Preguntas y Respuestas (QA) en inglés de 52 ítems elaborado a mano (`core_en`) para pruebas de regresión deterministas de [datasets/vincentkoc/tiny_qa_benchmark](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark) anteriores.
*   **Kit de Herramientas de Personalización Sintética:** Un script de Python (`tools/generator`) que utiliza LiteLLM para generar micro-benchmarks a medida bajo demanda para cualquier idioma, tema o dificultad.
*   **Metadatos Estandarizados:** Artefactos empaquetados en formato Croissant JSON-LD (`metadata/`) para su descubrimiento y carga automática por herramientas y motores de búsqueda.
*   **Ciencia Abierta:** Todo el código (generador, scripts de evaluación) y el conjunto de datos principal en inglés se publican bajo la licencia Apache-2.0. Los paquetes de datos generados sintéticamente tienen una licencia personalizada de solo evaluación.
*   **Alineación con LLMOps:** Diseñado para una fácil integración en pipelines de CI/CD, flujos de trabajo de ingeniería de prompts, detección de deriva multilingüe y paneles de observabilidad.
*   **Paquetes Multilingües:** Paquetes preconstruidos para numerosos idiomas, incluyendo inglés, francés, español, portugués, alemán, chino, japonés, turco, árabe y ruso.

## Instalación

El kit de herramientas TQB++ (incluyendo el generador de conjuntos de datos y las utilidades de evaluación) se puede instalar como un paquete de Python desde PyPI.

### Generación de Conjuntos de Datos Sintéticos (paquete de python)

```bash
pip install tinyqabenchmarkpp
```

(Nota: Asegúrate de tener Python 3.8+ y pip instalados. El nombre exacto del paquete en PyPI puede variar; por favor, comprueba los enlaces oficiales del proyecto si este comando no funciona.)

Una vez instalado, deberías poder usar los scripts del generador y de evaluación desde tu línea de comandos o importar funcionalidades en tus proyectos de Python.

## Carga de Conjuntos de Datos con `datasets` de Hugging Face

Los conjuntos de datos TQB++ están disponibles en Hugging Face Hub y se pueden cargar fácilmente usando la biblioteca `datasets`. Esta es la forma recomendada de acceder a los datos.

```python
from datasets import load_dataset, get_dataset_config_names

# Descubrir configuraciones de conjuntos de datos disponibles (ej. core_en, pack_fr_40, etc.)
configs = get_dataset_config_names("vincentkoc/tiny_qa_benchmark_pp")
print(f"Configuraciones disponibles: {configs}")

# Cargar el conjunto de datos principal en inglés (asumiendo que \'core_en\' es una configuración)
if "core_en" in configs:
    core_dataset = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="core_en", split="train")
    print(f"\\nCargados {len(core_dataset)} ejemplos de core_en:")
    # print(core_dataset[0]) # Imprimir el primer ejemplo
else:
    print("\\nNo se encontró la configuración \'core_en\'.")

# Cargar un paquete sintético específico (ej. un paquete en francés)
# Reemplaza \'pack_fr_40\' con un nombre de configuración real de la lista `configs`
example_pack_name = "pack_fr_40" # u otro nombre de configuración válido
if example_pack_name in configs:
    synthetic_pack = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name=example_pack_name, split="train")
    print(f"\\nCargados {len(synthetic_pack)} ejemplos de {example_pack_name}:")
    # print(synthetic_pack[0]) # Imprimir el primer ejemplo
else:
    print(f"\\nNo se encontró la configuración \'{example_pack_name}\'. Por favor, elige entre las configuraciones disponibles.")

```

Para obtener información más detallada sobre los conjuntos de datos, incluyendo su estructura y licencias específicas, por favor consulta los archivos README dentro del directorio `data/` (es decir, `data/README.md`, `data/core_en/README.md` y `data/packs/README.md`).

## Estructura del Repositorio

*   `data/`: Contiene los conjuntos de datos QA.
    *   `core_en/`: El conjunto de datos original de 52 ítems en inglés elaborado a mano.
    *   `packs/`: Paquetes de datos multilingües y temáticos generados sintéticamente.
*   `tools/`: Contiene scripts para la generación y evaluación de conjuntos de datos.
    *   `generator/`: El generador de conjuntos de datos QA sintéticos.
    *   `eval/`: Scripts y utilidades para evaluación de modelos contra los conjuntos de datos TQB++.
*   `paper/`: El código fuente LaTeX y archivos asociados para el artículo de investigación.
*   `metadata/`: Archivos de metadatos Croissant JSON-LD para los conjuntos de datos.
*   `LICENSE`: Licencia principal para la base de código (Apache-2.0).
*   `LICENCE.data_packs.md`: Licencia personalizada para los paquetes de datos generados sintéticamente.
*   `LICENCE.paper.md`: Licencia para el contenido del artículo.

## Escenarios de Uso

TQB++ está diseñado para diversos flujos de trabajo de LLMOps y evaluación:

*   **Pruebas de Pipeline CI/CD:** Úsalo como prueba unitaria con herramientas de prueba de LLM para servicios LLM para detectar regresiones.
*   **Ingeniería de Prompts y Desarrollo de Agentes:** Obtén retroalimentación rápida al iterar en prompts o diseños de agentes.
*   **Integración de Arnés de Evaluación:** Codifícalo como un YAML de OpenAI Evals o un conjunto de datos de Opik para el seguimiento en el panel de control.
*   **Detección de Deriva Multilingüe:** Monitoriza las regresiones de localización usando paquetes TQB++ multilingües.
*   **Pruebas Adaptativas:** Sintetiza nuevos micro-benchmarks sobre la marcha adaptados a características específicas o derivas de datos.
*   **Monitorización de Dinámicas de Ajuste Fino:** Rastrea la erosión del conocimiento o cambios de capacidad no deseados durante el ajuste fino.

## Citación

Si usas TQB++ en tu investigación o trabajo, por favor cita el TQB original y el artículo de TQB++:

```bibtex
% Este conjunto de datos sintético y generador
@misc{koctinyqabenchmarkpp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark++ (TQB++) Conjuntos de Datos y Kit de Herramientas},
  year         = {2025},
  publisher    = {Hugging Face & GitHub},
  doi          = {10.57967/hf/5531},
  howpublished = {\\url{https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp}},
  note         = {Ver también: \\url{https://github.com/vincentkoc/tiny_qa_benchmark_pp}}
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

<!-- % Marcador de posición para la cita del artículo de TQB++ JMLR - actualizar cuando esté disponible
@article{koc2025tqb_pp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark$^{++}$: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests},
  journal      = {Journal of Machine Learning Research (pendiente)},
  year         = {2025},
  volume       = {XX},
  number       = {X},
  pages        = {X-XX},
  url          = {http://jmlr.org/papers/vXX/koc25a.html} % URL de ejemplo
} -->

## Licencia
El código en este repositorio (incluyendo el generador y los scripts de evaluación) y el conjunto de datos `data/core_en` y cualquier otra cosa no mencionada con una licencia están licenciados bajo la Licencia Apache 2.0. Consulta el archivo `LICENSE` para más detalles.

Los paquetes de datos generados sintéticamente en `data/packs/` se distribuyen bajo una licencia personalizada "Solo Evaluación, No Comercial, Sin Derivadas". Consulta `LICENCE.data_packs.md` para más detalles.

Los archivos de metadatos Croissant JSON-LD en `metadata/` están disponibles bajo CC0-1.0.

El contenido del artículo en `paper/` está sujeto a sus propios términos de licencia, detallados en `LICENCE.paper.md`. 