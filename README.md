# ComicAnalizer

Sistema modular para reconstruir el orden narrativo de paginas de comic a partir de imagenes desordenadas.

## Arquitectura

El pipeline esta separado en modulos reemplazables:

1. `core/ingestion.py`: descubre imagenes y calcula ids/hashes.
2. `features/`: extrae metadata, embeddings visuales CLIP, OCR/texto y layout.
3. `models/relation_model.py`: calcula `score(A, B)` como probabilidad de transicion narrativa.
4. `graph/`: crea el grafo dirigido, ordena y valida.
5. `analyzers/`: plugins read-only con interfaz `Analyzer.run(data)`.
6. `reports/`: serializa el output estructurado.
7. `main.py`: CLI de orquestacion.

## Primer objetivo implementado

- Carga de imagenes desde archivo o directorio.
- Extraccion de metadata con OpenCV.
- Extraccion de embedding visual con CLIP usando backend `clip` o `transformers`.
- Fallback opcional a embedding visual OpenCV si CLIP no esta instalado.
- Calculo de similitud y score dirigido `A -> B`.
- Grafo narrativo dirigido con NetworkX.
- Ordenamiento global greedy inicial sin ciclos.
- Deteccion inicial de duplicados, outliers, gaps y clusters.
- Output JSON con `pages`, `relations`, `clusters`, `anomalies`, `order` y `analysis`.

## Uso

```bash
pip install -r requirements.txt
python main.py --input ./imagenes --output ./outputs/narrative_order.json
```

Para evitar guardar vectores completos en el JSON:

```bash
python main.py --input ./imagenes --output ./outputs/narrative_order.json --no-embeddings
```

## Decision clave

El modelo inicial es heuristico para mantener el sistema funcional desde el primer paso. La frontera de reemplazo es `HeuristicTransitionScorer.score(A, B)`: en una siguiente fase puede sustituirse por un modelo entrenado con pares positivos/negativos sin tocar ingestion, features, grafo, ordenamiento, validacion ni reportes.

## Entrenamiento direccional

El modelo entrenable vive en `models/`:

- `feature_extractor.py`: CLIP real con cache de embeddings; no usa fallback OpenCV.
- `dataset_pairs.py`: carga `metadata.json` y genera pares positivos/negativos.
- `pairwise_model.py`: MLP direccional sobre `[emb_A, emb_B, emb_B - emb_A, abs(emb_B - emb_A)]`.
- `train.py`: entrenamiento y evaluacion.

Ejemplo con `test_1_clean`:

```bash
python -m models.train ^
  --dataset "C:\Users\nico4\Downloads\ComicPruebas\datasets\test_1_clean" ^
  --clip-backend clip ^
  --epochs 20 ^
  --batch-size 8 ^
  --output outputs/learned_relation_model.pt ^
  --metrics-output outputs/learned_relation_metrics.json
```

Para entrenar con todos los escenarios, repite `--dataset` por cada carpeta.

Si existe un indice global generado por `dataset_builder.py`, tambien puedes
entrenar directamente desde `datasets/index.json`; el loader descubre todos los
comics y escenarios declarados en `datasets/by_comic/<comic_slug>/manifest.json`:

```bash
python -m models.train ^
  --dataset "C:\Users\nico4\Downloads\ComicPruebas\datasets\index.json" ^
  --clip-backend clip ^
  --epochs 50 ^
  --batch-size 64 ^
  --learning-rate 0.0003 ^
  --dropout 0.2 ^
  --negatives-per-positive 6 ^
  --output outputs/learned_relation_model_index_v1.pt ^
  --metrics-output outputs/learned_relation_metrics_index_v1.json
```

Para evaluar ranking full-candidate contra ese mismo indice:

```bash
python -m models.evaluate ^
  --checkpoint outputs/learned_relation_model_index_v1.pt ^
  --dataset "C:\Users\nico4\Downloads\ComicPruebas\datasets\index.json" ^
  --clip-backend clip ^
  --output outputs/learned_relation_logical_ranking_index_v1.json
```
