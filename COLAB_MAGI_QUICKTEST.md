# Magi En Colab Free: Prueba Rapida

Esta guia corre una prueba corta de Magi con GPU gratuita de Colab, usando el codigo del proyecto desde GitHub y un ZIP pequeno de paginas limpias generado localmente.

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
  --output-dir outputs\magi_cloud_sample `
  --zip-path outputs\magi_cloud_sample.zip
```

Esto crea:

```text
outputs/magi_cloud_sample.zip
```

Ese ZIP queda fuera de Git porque `outputs/` esta ignorado. Es lo recomendado si los comics son privados, pesados, con copyright o contenido sensible.

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
outputs/magi_cloud_sample.zip
```

```python
from google.colab import files
from pathlib import Path

uploaded = files.upload()
zip_name = next(iter(uploaded))
print("ZIP subido:", zip_name)

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
  --output-dir outputs/magi_debug/colab_middle_detections \
  --sample-one-per-comic \
  --dataset-name test_1_clean \
  --selection middle \
  --task detections \
  --cache-dir outputs/magi_cache \
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

## 6. Leer Metricas

```python
import json
from pathlib import Path

metrics_path = Path("outputs/magi_debug/colab_middle_detections/metrics.json")
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

La segunda corrida deberia ser mucho mas rapida porque lee `outputs/magi_cache`.

```python
!python -m tools.inspect_magi_dataset \
  --input /content/magi_sample/magi_cloud_sample/by_comic \
  --output-dir outputs/magi_debug/colab_middle_detections_cached \
  --sample-one-per-comic \
  --dataset-name test_1_clean \
  --selection middle \
  --task detections \
  --cache-dir outputs/magi_cache \
  --device cuda \
  --dtype float16 \
  --max-comics 3
```

## 8. Descargar Resultados

```python
from google.colab import files

!zip -qr magi_colab_results.zip outputs/magi_debug/colab_middle_detections outputs/magi_cache
files.download("magi_colab_results.zip")
```

## Que Mirar Primero

- `metrics.json`: tiempos, cache, conteos por pagina.
- `summary.json`: resumen legible por pagina.
- `magi_results.json`: salida completa normalizada.
- `*_boxes.jpg`: imagen con cajas dibujadas.
- carpetas `001_*`, `002_*`: recortes de paneles detectados.

Para calibrar Magi despues, usa estas salidas como evidencia: falsos negativos de personajes, globos sin cola, colas sin asociacion, paneles partidos o paneles fusionados.
