# ComicAnalizer

Sistema modular para reconstruir el orden narrativo de paginas de comic a
partir de imagenes desordenadas.

El objetivo final no es ordenar por nombre de archivo ni por OCR simple. El
sistema busca aprender una probabilidad direccional:

```text
P(A -> B) = probabilidad de que la pagina B venga despues de la pagina A
```

## Estado Actual

El proyecto ya tiene una base funcional:

- Ingesta de imagenes desde carpetas.
- Validacion de archivos ilegibles.
- Extraccion de metadata con OpenCV.
- Extraccion de embeddings visuales globales con CLIP.
- Modelo heuristico reemplazable.
- Modelo entrenable direccional en PyTorch.
- Construccion de grafo narrativo con NetworkX.
- Ordenamiento greedy y beam search.
- Validacion basica de duplicados, outliers, gaps y clusters.
- Output JSON con `pages`, `ordered_pages`, `relations`, `clusters`,
  `anomalies`, `order` y `analysis`.
- Exportacion de una carpeta con paginas copiadas y renombradas en el orden
  predicho.
- Integracion experimental estable de Magi v3 para detectar paneles, textos,
  personajes, colas/asociaciones y OCR propio de Magi.
- Comparacion complementaria con PaddleOCR.
- Fusion OCR Magi/PaddleOCR para sugerir el mejor texto por region y marcar
  desacuerdos automaticamente.
- Visualizaciones separadas por comic para:
  - cajas Magi (`magi_boxes`)
  - cajas PaddleOCR individuales (`ocr_boxes`)
  - grupos OCR tipo frase/globo (`ocr_groups`)
- Exportacion de evidencia OCR para futuras correcciones y entrenamiento.
- Reporte HTML interactivo con seleccion de cajas, comparacion de alternativas
  OCR, correcciones locales y exportacion de correcciones normalizables.

La limitacion principal actual es que el modelo entrenable aprende sobre
embeddings CLIP de pagina completa. Esto sirve como base, pero todavia no
equivale a comprender narrativa de vinetas, globos de texto, personajes y
continuidad interna.

## Arquitectura

El pipeline esta separado en modulos reemplazables:

1. `core/ingestion.py`: descubre imagenes, valida lectura y calcula ids/hashes.
2. `features/`: extrae metadata, embeddings visuales, OCR/texto, layout y
   futuros features estructurales.
3. `models/`: modelos de transicion `score(A, B)`.
4. `graph/`: construccion de grafo, ordenamiento y validacion.
5. `analyzers/`: plugins read-only con interfaz `Analyzer.run(data)`.
6. `reports/`: serializa JSON y exporta paginas ordenadas.
7. `tools/`: herramientas experimentales de inspeccion.
8. `main.py`: CLI de orquestacion.

## Uso Basico

```bash
pip install -r requirements.txt
python main.py --input ./imagenes --output ./outputs/narrative_order.json
```

Para evitar guardar vectores completos en el JSON:

```bash
python main.py --input ./imagenes --output ./outputs/narrative_order.json --no-embeddings
```

Para generar el JSON y, en el mismo proceso, exportar una carpeta con las
paginas copiadas y renombradas en el orden predicho por el modelo:

```bash
python main.py ^
  --input "C:\ruta\comic_desordenado" ^
  --output outputs/comic_order.json ^
  --config config/learned.json ^
  --no-embeddings ^
  --export-ordered-dir outputs/ordered_comics/comic ^
  --overwrite-export ^
  --export-order-source model ^
  --export-name-mode numbered
```

`ordered_pages` es la vista clara del orden narrativo predicho. `pages` conserva
la lista de paginas cargadas desde la carpeta de entrada, que puede coincidir o
no con el orden final.

## Entrenamiento Direccional

El modelo entrenable vive en `models/`:

- `feature_extractor.py`: CLIP real con cache de embeddings; no usa fallback
  OpenCV.
- `dataset_pairs.py`: carga `metadata.json` y genera pares positivos/negativos.
- `pairwise_model.py`: MLP direccional sobre
  `[emb_A, emb_B, emb_B - emb_A, abs(emb_B - emb_A)]`.
- `train.py`: entrenamiento y evaluacion.
- `evaluate.py`: evaluacion ranking full-candidate.

Ejemplo desde un indice global:

```bash
python -m models.train ^
  --dataset "C:\Users\nico4\Downloads\ComicPruebas\datasets\index.json" ^
  --clip-backend clip ^
  --epochs 50 ^
  --batch-size 64 ^
  --learning-rate 0.0003 ^
  --dropout 0.2 ^
  --negatives-per-positive 6 ^
  --output outputs/learned_relation_model.pt ^
  --metrics-output outputs/learned_relation_metrics.json
```

Para evaluar ranking:

```bash
python -m models.evaluate ^
  --checkpoint outputs/learned_relation_model.pt ^
  --dataset "C:\Users\nico4\Downloads\ComicPruebas\datasets\index.json" ^
  --clip-backend clip ^
  --output outputs/learned_relation_ranking.json
```

## Magi, PaddleOCR Y Evidencia OCR

Se agrego una capa experimental para entender paginas de comic con herramientas
externas y convertir sus salidas a estructuras auditables.

Magi se usa como extractor estructural:

- paneles
- regiones de texto
- personajes
- colas de globos
- asociaciones texto-personaje cuando estan disponibles
- OCR propio de Magi

PaddleOCR se usa como OCR complementario:

- detecta bloques de texto independientes
- permite comparar contra las regiones de texto de Magi
- ayuda a encontrar texto que Magi no detecto

El agrupador OCR combina PaddleOCR con el contexto de Magi para construir grupos
tipo frase/globo. Esto no reemplaza una correccion humana, pero hace mucho mas
facil comparar resultados visualmente.

La fusion OCR compara:

- OCR propio de Magi por region de texto.
- Grupos generados desde PaddleOCR.
- Confianza de PaddleOCR.
- Limpieza del texto y desacuerdos entre herramientas.

El resultado queda en `page_understanding_report.json` como `ocr_fusion` y en el
reporte HTML como capa `Fusion`. Al seleccionar una caja, el panel lateral muestra
la sugerencia del sistema, alternativas Magi/PaddleOCR y flags como
`ocr_disagreement`, `possible_noise`, `missing_magi_text` o
`missing_paddle_text`.

Las correcciones descargadas desde el reporte se pueden normalizar para
calibracion/entrenamiento:

```bash
python -m tools.normalize_review_corrections ^
  --input "C:\Users\nico4\Downloads\comic_ocr_corrections.json" ^
  --output-dir annotations/review_corrections
```

Para inspeccionar Magi localmente:

```bash
python -m tools.inspect_magi_dataset ^
  --input "C:\ruta\dataset\test_1_clean" ^
  --output-dir outputs/runs/comic_sample/magi ^
  --limit 1 ^
  --device cpu ^
  --dtype float32
```

En la primera prueba, Magi logro detectar textos, personajes, asociaciones y OCR
en portadas. En una pagina interior compleja detecto textos, personajes y colas
de globos, pero marco la pagina como un solo panel. Por eso Magi es una base
valiosa, pero todavia necesita complementarse con deteccion de vinetas mas
robusta y postprocesamiento propio.

Ver detalles en [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md).

Para correr el flujo completo en Colab con GPU, abrir
[notebooks/COMIC_ANALYSIS_COLAB.ipynb](notebooks/COMIC_ANALYSIS_COLAB.ipynb).
Ese notebook es el flujo oficial unico: ejecuta Magi con detecciones + OCR propio,
genera el reporte de calidad, corre PaddleOCR complementario, genera overlays
`ocr_boxes` y `ocr_groups`, exporta evidencia OCR para calibracion y descarga un
ZIP estandar del run.

## Estandar De Outputs

Las nuevas herramientas escriben salidas en este esquema:

```text
outputs/
  packages/              # ZIPs/datasets preparados para Colab
  legacy/                # salidas historicas previas al esquema de runs
  runs/<run_name>/       # salida consolidada de una corrida
    manifest.json
    magi/
      magi_results.json
      metrics.json
      summary.json
    analysis/
      magi_analysis_report.json
      paddle_magi_ocr_comparison.json
      page_understanding_report.json
      # page_understanding_report.json incluye ocr_fusion por pagina
      ocr_evidence/
        evidence.jsonl
        correction_template.jsonl
        evidence_index.json
        assets/<comic_id>/<page>/
    visuals/
      magi_boxes/<comic_id>/  # paginas completas con cajas Magi
      ocr_boxes/<comic_id>/   # paginas completas con cajas PaddleOCR
      ocr_groups/<comic_id>/  # frases/globos agrupados con apoyo de Magi
    report/
      index.html              # reporte visual navegable del run
annotations/
  review_corrections/
    review_corrections.jsonl
    review_corrections_index.json
```

La separacion por `comic_id` permite comparar visualmente dos comics sin mezclar
paginas ni overlays. Los JSON tambien conservan `comic_id`, `file_name`,
`page_id` y `path` para enlazar cada resultado visual con sus datos crudos.

Los resultados historicos sueltos se consideran legado de pruebas y deben quedar
en `outputs/legacy/`. Para consolidar outputs antiguos descargados desde Colab:

```bash
python -m tools.standardize_magi_outputs ^
  --magi-input "C:\Users\nico4\Downloads\magi_colab_full_results.zip" ^
  --ocr-comparison outputs/analysis/paddle_magi_ocr_comparison.json ^
  --image-root "C:\Users\nico4\Downloads\ComicPruebas\datasets\by_comic" ^
  --run-name colab_clean_full
```

## Analisis De Resultados Magi

Un ZIP descargado desde Colab se puede convertir en un reporte normalizado:

```bash
python -m tools.analyze_magi_results ^
  --input "C:\Users\nico4\Downloads\magi_colab_full_results.zip" ^
  --output outputs/analysis/magi_analysis_report.json
```

Para comparar una muestra de paginas contra PaddleOCR:

```bash
python -m tools.compare_magi_paddleocr ^
  --magi-input "C:\Users\nico4\Downloads\magi_colab_full_results.zip" ^
  --image-root "C:\Users\nico4\Downloads\ComicPruebas\datasets\by_comic" ^
  --dataset-name test_1_clean ^
  --selection random ^
  --limit 3 ^
  --visual-output-dir outputs/runs/ocr_sample/visuals/ocr_boxes ^
  --grouped-visual-output-dir outputs/runs/ocr_sample/visuals/ocr_groups ^
  --output outputs/analysis/paddle_magi_ocr_comparison.json
```

En Windows CPU, PaddleOCR puede ser muy lento. Para pruebas de mayor tamano,
usar el notebook de Colab.

## Evidencia OCR Para Calibracion

Para convertir resultados PaddleOCR en un dataset auditable de crops,
geometria, contexto Magi y plantillas de correccion:

```bash
python -m tools.export_ocr_evidence ^
  --ocr-report outputs/runs/colab_clean_full_ocr_full/analysis/paddle_magi_ocr_comparison.json ^
  --magi-input outputs/runs/colab_clean_full/magi ^
  --image-root outputs/packages/magi_clean_full/by_comic ^
  --dataset-name test_1_clean ^
  --output-dir outputs/runs/colab_clean_full_ocr_full/analysis/ocr_evidence ^
  --asset-policy priority ^
  --max-asset-blocks 500
```

Esto genera:

```text
outputs/runs/<run_name>/analysis/ocr_evidence/evidence.jsonl
outputs/runs/<run_name>/analysis/ocr_evidence/correction_template.jsonl
outputs/runs/<run_name>/analysis/ocr_evidence/evidence_index.json
outputs/runs/<run_name>/analysis/ocr_evidence/assets/<comic_id>/<page>/
```

Cada registro conserva texto OCR crudo, confianza, poligono, geometria normalizada,
metricas de pagina, contexto Magi, prioridad de revision, tags de entrenamiento y
un espacio para correccion humana posterior. La politica `priority` escribe JSONL
para todos los bloques OCR, pero genera crops solo para los casos mas utiles de
revisar: baja confianza, texto fuera de regiones Magi, posible ruido o paginas
donde PaddleOCR detecta mucho mas texto que Magi.

## Page Understanding Y Reporte HTML

Despues de tener una corrida con Magi y PaddleOCR, se puede generar una capa de
analisis por pagina:

```bash
python -m tools.generate_run_report ^
  --run-dir outputs/runs/colab_full_pipeline ^
  --image-root outputs/packages/magi_clean_full/by_comic ^
  --dataset-name test_1_clean
```

Esto crea:

```text
outputs/runs/<run_name>/analysis/page_understanding_report.json
outputs/runs/<run_name>/report/index.html
```

El reporte JSON agrega:

- candidatos de numeracion de pagina detectados por OCR;
- clasificacion heuristica de tipo de pagina:
  - `cover_or_title`
  - `interior_story`
  - `credits`
  - `ad_or_social`
  - `text_heavy`
  - `noise_or_blank`
  - `unknown`
- conteos Magi;
- resumen OCR;
- rutas a `magi_boxes`, `ocr_boxes` y `ocr_groups`.

El HTML permite revisar una corrida completa desde un solo archivo visual. Es el
punto de revision recomendado antes de calibracion manual, porque deja juntas
las tres vistas: Magi, PaddleOCR individual y PaddleOCR agrupado.

## Estado De Trabajo Actual

La base estable actual queda asi:

1. El ordenamiento narrativo global con CLIP + modelo entrenable sigue siendo el
   core de ordenamiento.
2. Magi y PaddleOCR ahora son una capa de page understanding para construir
   features estructurales.
3. `ocr_boxes` permite revisar bloques detectados individualmente.
4. `ocr_groups` permite revisar texto agrupado en frases/globos, lo que facilita
   comparar Magi contra PaddleOCR.
5. `ocr_evidence` deja preparado un dataset auditable para correcciones futuras.

Mientras no se haga calibracion manual, los siguientes avances programables son:

- mejorar reglas automaticas de agrupacion OCR;
- detectar numeros de pagina y titulos repetidos;
- clasificar pagina como portada, interior, creditos, anuncio o ruido;
- generar embeddings por panel/crop;
- agregar metricas automaticas de calidad por pagina y por comic;
- mejorar el reporte HTML para agregar filtros por tipo de pagina, sospecha y
  diferencia Magi/PaddleOCR.
