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
- Primera prueba experimental de Magi v3 para detectar paneles, textos,
  personajes, asociaciones y OCR.

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

## Prueba Experimental Con Magi

Se agrego una herramienta experimental para inspeccionar Magi v3:

```bash
python -m tools.inspect_magi_dataset ^
  --input "C:\ruta\dataset\test_1_clean" ^
  --output-dir outputs/magi_debug/comic_sample ^
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

Para probar Magi en Colab Free con GPU y una muestra pequena de paginas limpias,
ver [COLAB_MAGI_QUICKTEST.md](COLAB_MAGI_QUICKTEST.md).
