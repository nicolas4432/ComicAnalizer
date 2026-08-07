# Hoja De Ruta De ComicAnalizer

## Vision

Construir un sistema capaz de entender y reconstruir continuidad narrativa en
comics desordenados.

El sistema debe manejar:

- Paginas sin texto.
- Paginas visualmente similares.
- Multiples comics mezclados.
- Paginas faltantes.
- Imagenes irrelevantes o duplicadas.
- Portadas, creditos, anuncios y paginas no narrativas.
- Numeracion visible, titulos repetidos y marcas de capitulo.
- Continuidad entre vinetas, globos de texto, personajes y escenas.

La idea central sigue siendo:

```text
score(A, B) = P(A -> B)
```

pero el score futuro no debe depender solo de embeddings globales. Debe usar una
representacion estructural de cada pagina.

## Estado Actual Del Sistema

### Lo Que Ya Existe

El proyecto actual ya resuelve la arquitectura base:

- Pipeline modular.
- Ingesta de imagenes.
- Validacion de archivos ilegibles.
- Metadata visual basica con OpenCV.
- Embeddings CLIP de pagina completa.
- Modelo heuristico reemplazable.
- Modelo entrenable direccional PyTorch.
- Construccion de pares positivos/negativos desde metadata.
- Grafo narrativo con NetworkX.
- Ordenamiento greedy y beam search.
- Exportacion de JSON estructurado.
- Exportacion de carpeta con paginas renombradas en orden predicho.
- Sistema de analyzers extensible.
- Integracion experimental estable de Magi v3.
- Notebook unico de Colab para Magi + OCR propio de Magi + PaddleOCR.
- Reportes normalizados por corrida en `outputs/runs/<run_name>/`.
- Visualizaciones por comic:
  - `magi_boxes`
  - `ocr_boxes`
  - `ocr_groups`
- Evidencia OCR auditable para futuras correcciones y entrenamiento.
- Fusion OCR Magi/PaddleOCR con sugerencias por region, alternativas y flags de
  desacuerdo.
- Reporte interactivo para revisar cajas, comparar OCRs, guardar correcciones y
  convertirlas en JSONL de calibracion.

### Limitacion Principal

El modelo entrenable actual usa principalmente:

```text
pagina completa -> CLIP -> embedding global
```

Luego compara pares con:

```text
[emb_A, emb_B, emb_B - emb_A, abs(emb_B - emb_A)]
```

Esto da direccionalidad, pero no comprension narrativa profunda. El modelo puede
aprender estilo, color, composicion y similitud global, pero no ve
explicitamente:

- Ultimo panel de A vs primer panel de B.
- Globos de dialogo y continuidad de texto.
- Personajes recurrentes.
- Asociacion texto-personaje.
- Numeros de pagina.
- Titulos repetidos.
- Portadas y creditos.
- Cambios de escena.

## Representacion Objetivo

La pagina debe evolucionar desde:

```text
PageFeatures:
  visual_embedding
  metadata
  text
  layout
```

hacia:

```text
PageFeatures:
  global_embedding
  metadata
  page_type
  detected_page_numbers
  repeated_title_candidates
  panels[]
  text_blocks[]
  balloons[]
  balloon_tails[]
  characters[]
  character_clusters[]
  text_character_associations[]
  ocr_fusion[]
  correction_evidence[]
  ocr_text
  layout_graph
```

Cada panel deberia tener:

```text
PanelFeatures:
  bbox
  reading_order
  crop_path
  clip_embedding
  dino_embedding
  texts_inside[]
  characters_inside[]
  scene_embedding
```

## Herramientas Candidatas

### Magi

Rol recomendado:

- Detector de textos.
- Detector de personajes.
- Detector de colas de globos.
- Asociacion texto-personaje.
- OCR especializado en comic/manga.
- Clustering de personajes.

Magi debe ser usado como extractor estructural, no como reemplazo del proyecto.

Resultado de la primera prueba local:

- En portadas detecto textos, personajes y OCR correctamente.
- En una pagina interior compleja detecto textos, personajes y tails, pero solo
  detecto un panel grande.

Conclusion:

```text
Magi es util como ojo semantico.
No basta como unico detector de vinetas.
```

### PaddleOCR

Rol recomendado:

- OCR local alternativo si Magi falla.
- Extraccion de texto con bounding boxes.
- Comparacion con Magi OCR para robustez.

No asocia texto con personajes por si solo.

### Fusion OCR Propia

Rol recomendado:

- Comparar Magi OCR contra PaddleOCR agrupado.
- Elegir una sugerencia textual inicial.
- Detectar desacuerdos y posibles falsos positivos.
- Guardar alternativas para correccion humana y entrenamiento posterior.

La fusion no reemplaza una revision humana: produce candidatos y flags para que
la calibracion sea mas rapida y trazable.

### Manga OCR

Rol recomendado:

- OCR especializado para manga japones.
- Usarlo solo cuando el idioma detectado o el dataset lo justifique.

### SAM

Rol recomendado:

- Segmentacion fina.
- Apoyo para separar personajes/objetos cuando Magi o deteccion por cajas sea
  insuficiente.

No resuelve por si solo paneles ni narrativa.

### Grounding DINO

Rol recomendado:

- Deteccion abierta por prompts.
- Experimentos con prompts como `comic panel`, `speech bubble`, `character`,
  `page number`, `title`.

Debe considerarse experimental para este dominio.

### CLIP / DINOv2

Rol recomendado:

- Embeddings globales de pagina.
- Embeddings por panel.
- Comparacion visual de continuidad.
- Clustering de estilo y comic.

DINOv2 puede ser especialmente util para similitud visual mas fina que CLIP.

### Qwen2.5-VL / OpenAI Vision

Rol recomendado:

- Analisis auxiliar.
- Captions o pseudo-etiquetado.
- Auditoria de errores.
- Generacion de datos de entrenamiento supervisados.

No deben ser el core si se busca un sistema local, reproducible y robusto.

## Modelo Futuro De Transicion

El nuevo `score(A, B)` deberia considerar:

```text
global_page_similarity
global_page_direction_delta
last_panel_A -> first_panel_B
last_k_panels_A -> first_k_panels_B
text_tail_A -> text_head_B
character_overlap
character_identity_continuity
scene_continuity
layout_compatibility
page_number_delta
title/header repetition
page_type_prior
```

Ejemplo conceptual:

```text
score(A, B) =
  model(
    global_A,
    global_B,
    last_panel_A,
    first_panel_B,
    text_tail_A,
    text_head_B,
    character_overlap,
    page_number_features,
    page_type_features,
    layout_features
  )
```

## Pipeline Futuro

```text
1. Ingesta
2. Preprocesamiento
3. Page understanding
   - portada/interior/creditos/anuncio
   - OCR
   - titulos
   - numeracion
4. Panel understanding
   - deteccion de vinetas
   - orden de lectura interno
   - crops
   - embeddings por panel
5. Character/Text understanding
   - personajes
   - globos
   - colas
   - asociaciones texto-personaje
6. Modelo P(A -> B)
7. Grafo
8. Ordenamiento global
9. Validacion
10. Exportacion JSON + carpeta ordenada
```

## Calibracion De Magi

Antes de fine-tuning, conviene calibrar:

- Thresholds de cajas.
- Filtros por tamano minimo/maximo.
- Merge de cajas cercanas.
- Separacion de textos decorativos vs dialogo.
- Reglas para portada/creditos.
- Asociaciones texto-personaje dudosas.
- Orden de lectura de paneles.

Fine-tuning solo deberia evaluarse despues de medir fallas reales con suficientes
paginas anotadas.

## Consideraciones Sobre Contenido Adulto

Si Magi corre localmente, no deberia bloquear imagenes por politicas de API. Sin
embargo, puede rendir peor si el contenido esta fuera del dominio visual donde
fue entrenado.

Para APIs externas, puede haber restricciones de uso o filtros de contenido. Por
eso las APIs deben ser auxiliares, no dependencia central.

El sistema debe incluir validacion y exclusion de contenido ilegal o ambiguo.

## Proximos Pasos Recomendados

### Fase 1: Evaluacion De Magi Y OCR Complementario

- Ejecutar Magi sobre paginas limpias variadas.
- Separar portadas, interiores, paginas con mucho texto y paginas sin texto.
- Comparar Magi contra PaddleOCR.
- Generar overlays visuales para revision:
  - `magi_boxes`
  - `ocr_boxes`
  - `ocr_groups`
- Medir:
  - paneles detectados vs paneles reales
  - textos detectados
  - OCR correcto
  - personajes detectados
  - asociaciones texto-personaje

Estado:

```text
Implementado como base funcional.
```

La evaluacion ya puede correrse desde:

```text
notebooks/COMIC_ANALYSIS_COLAB.ipynb
```

El resultado esperado queda en:

```text
outputs/runs/<run_name>/magi/
outputs/runs/<run_name>/analysis/
outputs/runs/<run_name>/visuals/magi_boxes/
outputs/runs/<run_name>/visuals/ocr_boxes/
outputs/runs/<run_name>/visuals/ocr_groups/
outputs/runs/<run_name>/analysis/ocr_evidence/
```

### Fase 2: Normalizador De Features

Crear:

```text
features/magi_normalizer.py
```

para convertir outputs crudos de Magi a estructuras internas estables.

Estado inicial implementado:

```text
features/magi_schema.py
features/magi_postprocess.py
features/ocr_paddle.py
features/ocr_grouping.py
features/ocr_evidence.py
tools/analyze_magi_results.py
tools/compare_magi_paddleocr.py
tools/export_ocr_evidence.py
tools/calibrate_ocr_grouping.py
tools/standardize_magi_outputs.py
notebooks/COMIC_ANALYSIS_COLAB.ipynb
```

La salida de trabajo principal ahora usa una ruta estandar por corrida:

```text
outputs/runs/<run_name>/manifest.json
outputs/runs/<run_name>/analysis/magi_analysis_report.json
outputs/runs/<run_name>/analysis/paddle_magi_ocr_comparison.json
outputs/runs/<run_name>/visuals/magi_boxes/<comic_id>/
outputs/runs/<run_name>/visuals/ocr_boxes/<comic_id>/
outputs/runs/<run_name>/visuals/ocr_groups/<comic_id>/
outputs/runs/<run_name>/analysis/ocr_evidence/
outputs/runs/<run_name>/analysis/page_understanding_report.json
outputs/runs/<run_name>/report/index.html
```

Los overlays visuales se guardan separados por comic para revisar resultados
lado a lado sin mezclar paginas de historias distintas.

### Fase 2B: Calibracion OCR Sin Correccion Manual

Mientras no exista correccion humana pagina por pagina, aun se puede avanzar con
programacion automatica:

- mejorar agrupacion de palabras en frases usando:
  - region de texto Magi
  - cercania geometrica
  - alineacion por lineas
  - tamano de fuente estimado
  - distancia vertical/horizontal normalizada
- detectar texto sospechoso fuera de globos o fuera de regiones Magi;
- marcar falsos positivos probables cuando PaddleOCR lea patrones de fondo como
  texto;
- extraer crops de evidencia por prioridad;
- producir un reporte visual por comic con paginas ordenadas por riesgo;
- guardar plantillas de correccion para cuando exista revision manual.

Objetivo de esta fase:

```text
Reducir el trabajo manual futuro y preparar datos mas limpios para entrenar.
```

### Fase 3: Detector De Paneles Complementario

Evaluar:

- OpenCV/layout heuristico.
- Modelo entrenado con Manga109/eBDtheque.
- Grounding DINO experimental.
- Segmentacion asistida con SAM.

### Fase 4: Embeddings Por Panel

Crear:

```text
features/panel_embeddings.py
```

para generar CLIP/DINOv2 sobre crops de vinetas.

### Fase 5: Nuevo Modelo Narrativo

Crear un modelo que aprenda con features enriquecidas:

```text
models/narrative_relation_model.py
```

y que reemplace el MLP global actual.

### Fase 6: Evaluacion Por Escenario

Medir separadamente:

- comics limpios
- shuffled
- sin texto
- visualmente similares
- ruido
- multi-comic
- paginas faltantes
- comics externos no vistos

## Siguientes Pasos Programables Sin Calibracion Manual

Orden recomendado:

1. Cerrar commit de la base actual:
   - notebook unico
   - `ocr_groups`
   - evidencia OCR
   - documentacion actualizada
2. Crear un reporte HTML local para revisar una corrida completa sin abrir
   imagen por imagen.
3. Agregar detector automatico de numeros de pagina:
   - zonas de borde inferior/superior
   - OCR filtrado por regex numerica
   - confianza y posicion
4. Agregar clasificador heuristico de tipo de pagina:
   - portada
   - interior
   - creditos
   - anuncio/ruido
5. Agregar detector de titulos repetidos:
   - texto OCR frecuente por comic
   - posicion estable en pagina
   - similitud textual
6. Crear `features/panel_embeddings.py` para embeddings por panel cuando haya
   paneles detectados por Magi u otro detector.
7. Empezar a alimentar el modelo de transicion con features estructurales, no
   solo CLIP global.

Estado de los puntos 1-4:

```text
Implementado como primera version programable.
```

Archivos principales:

```text
features/page_numbers.py
features/page_type.py
tools/generate_run_report.py
notebooks/COMIC_ANALYSIS_COLAB.ipynb
```

Limitaciones actuales:

- La numeracion de pagina es heuristica y puede confundir numeros de dialogo o
  efectos visuales con numeros reales.
- El tipo de pagina es una clasificacion inicial basada en conteos Magi, OCR y
  palabras clave, no un modelo entrenado.
- El HTML es una vista de inspeccion; todavia no incluye filtros interactivos ni
  correccion manual integrada.

Siguiente avance programable recomendado:

```text
detector de titulos repetidos + filtros interactivos en el HTML
```

## Decision Tecnica Actual

La direccion recomendada es:

```text
No reemplazar ComicAnalizer por Magi.
Usar Magi como extractor estructural.
Complementar Magi con OCR/panel detection/embeddings.
Mantener un modelo propio para P(A -> B).
```

Magi puede darnos ojos. ComicAnalizer debe aprender la continuidad narrativa.
