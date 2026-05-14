# Magi En Colab Free: Prueba Rapida

Esta guia corre una prueba corta de Magi con GPU gratuita de Colab, usando el codigo del proyecto desde GitHub y un ZIP pequeno de paginas limpias generado localmente.

Para el flujo completo con reporte de calidad y comparacion PaddleOCR, abre
`notebooks/MAGI_ANALYSIS_COLAB.ipynb` directamente en Colab. Esta guia queda
como version rapida/manual.

Para probar solo OCR complementario sobre resultados Magi ya existentes, abre
`notebooks/OCR_COMPARISON_COLAB.ipynb`.

La idea es probar primero deteccion sin OCR:

- paneles
- cuadros/globos de texto como regiones
- personajes
- colas/asociaciones visuales cuando Magi las entregue
- tiempos por pagina y cache

> Nota: el primer arranque puede tardar por la descarga del modelo. La prueba de inferencia apunta a menos de 5 minutos si usas 2 o 3 paginas con GPU.

## 0. Preparar ZIP Local Antes De Abrir Colab

En tu maquina, desde el proyecto:

```powershell
python -m tools.export_magi_cloud_sample `
  --by-comic-root "C:\Users\nico4\Downloads\ComicPruebas\datasets\by_comic" `
  --dataset-name test_1_clean `
  --selection middle `
  --max-comics 8 `
  --output-dir outputs\packages\magi_cloud_sample `
  --zip-path outputs\packages\magi_cloud_sample.zip
```

Esto crea:

```text
outputs/packages/magi_cloud_sample.zip
```

Ese ZIP queda fuera de Git porque `outputs/` esta ignorado. Es lo recomendado si los comics son privados, pesados, con copyright o contenido sensible.

Para preparar todos los comics limpios completos:

```powershell
python -m tools.export_magi_cloud_sample `
  --by-comic-root "C:\Users\nico4\Downloads\ComicPruebas\datasets\by_comic" `
  --dataset-name test_1_clean `
  --all-pages `
  --max-comics 0 `
  --output-dir outputs\packages\magi_clean_full `
  --zip-path outputs\packages\magi_clean_full.zip
```

Esto crea:

```text
outputs/packages/magi_clean_full.zip
```

Para preparar solo el comic nuevo `nekkorarekko` completo:

```powershell
python -m tools.export_magi_cloud_sample `
  --by-comic-root "C:\Users\nico4\Downloads\ComicPruebas\datasets\by_comic" `
  --dataset-name test_1_clean `
  --all-pages `
  --comic-id nekkorarekko `
  --max-comics 0 `
  --output-dir outputs\packages\magi_nekkorarekko_clean `
  --zip-path outputs\packages\magi_nekkorarekko_clean.zip
```

Esto crea:

```text
outputs/packages/magi_nekkorarekko_clean.zip
```

## 1. Activar GPU En Colab

En Colab:

```text
Runtime > Change runtime type > Hardware accelerator > T4 GPU
```

Luego ejecuta:

```python
import torch

print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 2. Clonar El Proyecto Desde GitHub

Cambia `REPO_URL` por tu repo real. Si todavia no subiste el commit a GitHub, primero haz `git remote add ...` y `git push`.

```python
REPO_URL = "https://github.com/TU_USUARIO/ComicAnalizer.git"

!git clone "$REPO_URL" /content/ComicAnalizer
%cd /content/ComicAnalizer
!git log --oneline -5
```

## 3. Instalar Dependencias De Magi

Colab ya trae PyTorch con CUDA. No reinstales `torch` salvo que sea necesario.

```python
%cd /content/ComicAnalizer

!pip -q install \
  "transformers==4.49.0" \
  "huggingface_hub<1.0" \
  timm \
  einops \
  pytorch-metric-learning \
  shapely
```

## 4. Subir El ZIP De Muestra

Sube el archivo:

```text
outputs/packages/magi_cloud_sample.zip
```

Para el comic nuevo sube:

```text
outputs/packages/magi_nekkorarekko_clean.zip
```

```python
from google.colab import files
from pathlib import Path

uploaded = files.upload()
zip_name = next(iter(uploaded))
package_name = zip_name.replace(".zip", "")
print("ZIP subido:", zip_name)
print("Package:", package_name)

!rm -rf /content/magi_sample
!mkdir -p /content/magi_sample
!unzip -q -o "$zip_name" -d /content/magi_sample
!find /content/magi_sample -maxdepth 4 -type f | head -20
```

## 5. Ejecutar Prueba Corta Sin OCR

Usa `--max-comics 3` para una prueba rapida. Si tarda poco, sube a 5 u 8.

```python
%cd /content/ComicAnalizer

!python -m tools.inspect_magi_dataset \
  --input /content/magi_sample/magi_cloud_sample/by_comic \
  --output-dir outputs/runs/colab_middle_sample/magi \
  --sample-one-per-comic \
  --dataset-name test_1_clean \
  --selection middle \
  --task detections \
  --cache-dir outputs/cache/magi \
  --device cuda \
  --dtype float16 \
  --max-comics 3
```

Si Colab no te asigna GPU, cambia:

```text
--device cuda --dtype float16
```

por:

```text
--device cpu --dtype float32
```

pero no esperes que entre en 5 minutos.

## 5B. Ejecutar Todos Los Comics Limpios Sin OCR

Usa esta version cuando subas `magi_clean_full.zip`. Procesa todas las paginas
de `test_1_clean` en todos los comics incluidos en el ZIP.

```python
%cd /content/ComicAnalizer

!python -m tools.inspect_magi_dataset \
  --input /content/magi_sample/magi_clean_full/by_comic \
  --output-dir outputs/runs/colab_clean_full/magi \
  --visual-output-dir outputs/runs/colab_clean_full/visuals/magi_boxes \
  --all-pages-per-comic \
  --no-panel-crops \
  --dataset-name test_1_clean \
  --task detections \
  --cache-dir outputs/cache/magi \
  --device cuda \
  --dtype float16 \
  --max-comics 0
```

## 5C. Ejecutar Solo Nekkorarekko Sin OCR

Usa esta version cuando subas `magi_nekkorarekko_clean.zip`.
Los resultados visuales quedan separados por comic en:

```text
outputs/runs/nekkorarekko_clean/visuals/magi_boxes/nekkorarekko/
```

```python
%cd /content/ComicAnalizer

RUN_NAME = "nekkorarekko_clean"
PACKAGE_NAME = "magi_nekkorarekko_clean"

!python -m tools.inspect_magi_dataset \
  --input /content/magi_sample/$PACKAGE_NAME/by_comic \
  --output-dir outputs/runs/$RUN_NAME/magi \
  --visual-output-dir outputs/runs/$RUN_NAME/visuals/magi_boxes \
  --all-pages-per-comic \
  --no-panel-crops \
  --dataset-name test_1_clean \
  --task detections \
  --cache-dir outputs/cache/magi \
  --device cuda \
  --dtype float16 \
  --comic-id nekkorarekko \
  --max-comics 0
```

Para comparar una muestra con PaddleOCR:

```python
!python -m tools.compare_magi_paddleocr \
  --magi-input outputs/runs/$RUN_NAME/magi \
  --image-root /content/magi_sample/$PACKAGE_NAME/by_comic \
  --dataset-name test_1_clean \
  --selection random \
  --limit 12 \
  --seed 42 \
  --lang en \
  --comic-id nekkorarekko \
  --visual-output-dir outputs/runs/$RUN_NAME/visuals/ocr_boxes \
  --output outputs/runs/$RUN_NAME/analysis/paddle_magi_ocr_comparison.json
```

Los overlays de OCR quedan en:

```text
outputs/runs/nekkorarekko_clean/visuals/ocr_boxes/nekkorarekko/
```

Si quieres probar el flujo completo pero sin gastar toda la sesion de Colab,
limita primero a 1 o 2 comics:

```python
!python -m tools.inspect_magi_dataset \
  --input /content/magi_sample/magi_clean_full/by_comic \
  --output-dir outputs/runs/colab_clean_two_comics/magi \
  --visual-output-dir outputs/runs/colab_clean_two_comics/visuals/magi_boxes \
  --all-pages-per-comic \
  --no-panel-crops \
  --dataset-name test_1_clean \
  --task detections \
  --cache-dir outputs/cache/magi \
  --device cuda \
  --dtype float16 \
  --max-comics 2
```

Para una corrida larga conviene guardar resultados en Drive o descargar el ZIP
final, porque el runtime de Colab se puede reiniciar.

## 6. Leer Metricas

```python
import json
from pathlib import Path

metrics_path = Path("outputs/runs/colab_middle_sample/magi/metrics.json")
metrics = json.loads(metrics_path.read_text())

print("Paginas:", metrics["page_count"])
print("Cache hits:", metrics["cache_hits"])
print("Cache misses:", metrics["cache_misses"])
print("Tiempo total:", round(metrics["wall_elapsed_seconds"], 2), "s")

for page in metrics["pages"]:
    print(
        page["comic_id"],
        "tiempo=", round(page["elapsed_seconds"], 2),
        "paneles=", page.get("panels_count"),
        "textos=", page.get("texts_count"),
        "personajes=", page.get("characters_count"),
        "colas=", page.get("tails_count"),
    )
```

## 7. Repetir Para Probar Cache

La segunda corrida deberia ser mucho mas rapida porque lee `outputs/cache/magi`.

```python
!python -m tools.inspect_magi_dataset \
  --input /content/magi_sample/magi_cloud_sample/by_comic \
  --output-dir outputs/runs/colab_middle_sample_cached/magi \
  --sample-one-per-comic \
  --dataset-name test_1_clean \
  --selection middle \
  --task detections \
  --cache-dir outputs/cache/magi \
  --device cuda \
  --dtype float16 \
  --max-comics 3
```

## 8. Descargar Resultados

```python
from google.colab import files

RUN_NAME = "nekkorarekko_clean"  # cambia si usaste otro run
zip_out = f"{RUN_NAME}_magi_ocr_outputs.zip"
!zip -qr "$zip_out" outputs/runs/$RUN_NAME outputs/cache/magi
files.download(zip_out)
```

## Que Mirar Primero

- `metrics.json`: tiempos, cache, conteos por pagina.
- `summary.json`: resumen legible por pagina.
- `magi_results.json`: salida completa normalizada.
- `visuals/magi_boxes/<comic>/*_magi_boxes.jpg`: cajas Magi por comic.
- `visuals/ocr_boxes/<comic>/*_ocr_boxes.jpg`: cajas PaddleOCR por comic.
- carpetas `001_*`, `002_*`: recortes de paneles detectados.

Para calibrar Magi despues, usa estas salidas como evidencia: falsos negativos de personajes, globos sin cola, colas sin asociacion, paneles partidos o paneles fusionados.
