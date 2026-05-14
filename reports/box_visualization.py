from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from features.magi_schema import BoundingBox, MagiPageAnalysis
from features.ocr_paddle import PaddleOCRPageResult


@dataclass(frozen=True)
class OverlayBox:
    label: str
    box: BoundingBox
    color: str
    polygon: list[list[float]] | None = None


MAGI_COLORS = {
    "panel": "lime",
    "text": "dodgerblue",
    "character": "red",
    "tail": "magenta",
}
OCR_COLOR = "orange"
OCR_BOX_COLOR = (255, 184, 28)
OCR_BADGE_FILL = (255, 214, 70)
OCR_TEXT_COLOR = (0, 0, 0)


def draw_overlay(
    image_path: str | Path,
    output_path: str | Path,
    boxes: list[OverlayBox],
    line_width: int = 4,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in boxes:
        if item.polygon:
            points = [(point[0], point[1]) for point in item.polygon]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill=item.color, width=line_width)
        else:
            box = item.box
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=item.color, width=line_width)
        label = item.label[:80]
        draw.text((item.box.x1 + 4, item.box.y1 + 4), label, fill=item.color)

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def draw_magi_overlay(
    page: MagiPageAnalysis,
    image_path: str | Path,
    output_path: str | Path,
    include_panels: bool = True,
    include_texts: bool = True,
    include_characters: bool = True,
    include_tails: bool = True,
) -> None:
    boxes: list[OverlayBox] = []
    if include_panels:
        boxes.extend(
            OverlayBox(f"panel:{item.index}", item.box, MAGI_COLORS["panel"])
            for item in page.panels
        )
    if include_texts:
        boxes.extend(
            OverlayBox(f"text:{item.index}", item.box, MAGI_COLORS["text"])
            for item in page.texts
        )
    if include_characters:
        boxes.extend(
            OverlayBox(f"char:{item.index}", item.box, MAGI_COLORS["character"])
            for item in page.characters
        )
    if include_tails:
        boxes.extend(
            OverlayBox(f"tail:{item.index}", item.box, MAGI_COLORS["tail"])
            for item in page.tails
        )
    draw_overlay(image_path=image_path, output_path=output_path, boxes=boxes)


def draw_paddle_ocr_overlay(
    ocr_result: PaddleOCRPageResult,
    output_path: str | Path,
) -> None:
    boxes = [
        OverlayBox(
            label=f"{block.index}:{block.text}",
            box=block.box,
            color=OCR_COLOR,
            polygon=block.polygon,
        )
        for block in ocr_result.blocks
    ]
    draw_overlay(image_path=ocr_result.path, output_path=output_path, boxes=boxes)


def draw_paddle_ocr_overlay_from_dict(
    ocr_result: dict[str, Any],
    output_path: str | Path,
) -> None:
    boxes: list[OverlayBox] = []
    for block in ocr_result.get("blocks") or []:
        box_payload = block.get("box") or {}
        box = BoundingBox(
            x1=float(box_payload.get("x1", 0.0)),
            y1=float(box_payload.get("y1", 0.0)),
            x2=float(box_payload.get("x2", 0.0)),
            y2=float(box_payload.get("y2", 0.0)),
        )
        boxes.append(
            OverlayBox(
                label=f"{block.get('index', 0)}:{block.get('text', '')}",
                box=box,
                color=OCR_COLOR,
                polygon=block.get("polygon"),
            )
        )
    draw_overlay(
        image_path=ocr_result["path"],
        output_path=output_path,
        boxes=boxes,
    )


def draw_paddle_ocr_readable_overlay_from_dict(
    ocr_result: dict[str, Any],
    output_path: str | Path,
    min_confidence: float = 0.0,
    sidebar_width: int = 520,
) -> None:
    image = Image.open(ocr_result["path"]).convert("RGB")
    blocks = readable_ocr_blocks(ocr_result, min_confidence=min_confidence)

    badge_font = load_font(size=max(18, image.width // 55), bold=True)
    sidebar_font = load_font(size=max(18, image.width // 65), bold=False)
    sidebar_bold = load_font(size=max(20, image.width // 58), bold=True)
    line_height = max(28, sidebar_font.size + 10 if hasattr(sidebar_font, "size") else 28)
    header_height = 96
    sidebar_needed_height = header_height + max(1, len(blocks)) * line_height + 32
    canvas_height = max(image.height, sidebar_needed_height)
    canvas_width = image.width + sidebar_width

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for display_index, block in enumerate(blocks, 1):
        draw_ocr_box_with_badge(
            draw=draw,
            block=block,
            display_index=display_index,
            font=badge_font,
        )

    sidebar_x = image.width
    draw.rectangle((sidebar_x, 0, canvas_width, canvas_height), fill=(248, 248, 248))
    draw.line((sidebar_x, 0, sidebar_x, canvas_height), fill=(210, 210, 210), width=3)
    draw.text((sidebar_x + 24, 24), "PaddleOCR blocks", fill=OCR_TEXT_COLOR, font=sidebar_bold)
    draw.text(
        (sidebar_x + 24, 56),
        f"{len(blocks)} bloques detectados",
        fill=(70, 70, 70),
        font=sidebar_font,
    )

    y = header_height
    for display_index, block in enumerate(blocks, 1):
        text = block.get("text", "").strip()
        confidence = block.get("confidence")
        prefix = f"{display_index:02d}"
        confidence_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "--"
        line = f"{prefix}  [{confidence_text}]  {text}"
        draw.text((sidebar_x + 24, y), line[:64], fill=OCR_TEXT_COLOR, font=sidebar_font)
        y += line_height

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def readable_ocr_blocks(
    ocr_result: dict[str, Any],
    min_confidence: float,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in ocr_result.get("blocks") or []:
        text = str(block.get("text") or "").strip()
        confidence = block.get("confidence")
        if not text:
            continue
        if isinstance(confidence, (int, float)) and confidence < min_confidence:
            continue
        blocks.append(block)
    return sorted(
        blocks,
        key=lambda item: (
            float((item.get("box") or {}).get("y1", 0.0)),
            float((item.get("box") or {}).get("x1", 0.0)),
        ),
    )


def draw_ocr_box_with_badge(
    draw: ImageDraw.ImageDraw,
    block: dict[str, Any],
    display_index: int,
    font: ImageFont.ImageFont,
) -> None:
    box_payload = block.get("box") or {}
    x1 = float(box_payload.get("x1", 0.0))
    y1 = float(box_payload.get("y1", 0.0))
    x2 = float(box_payload.get("x2", 0.0))
    y2 = float(box_payload.get("y2", 0.0))
    polygon = block.get("polygon")
    if polygon:
        points = [(float(point[0]), float(point[1])) for point in polygon]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=OCR_BOX_COLOR, width=5)
    else:
        draw.rectangle((x1, y1, x2, y2), outline=OCR_BOX_COLOR, width=5)

    label = f"{display_index:02d}"
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0] + 12
    label_height = label_box[3] - label_box[1] + 8
    badge_x1 = max(0, x1)
    badge_y1 = max(0, y1 - label_height - 2)
    if badge_y1 < 4:
        badge_y1 = y1 + 2
    badge_x2 = badge_x1 + label_width
    badge_y2 = badge_y1 + label_height
    draw.rectangle((badge_x1, badge_y1, badge_x2, badge_y2), fill=OCR_BADGE_FILL, outline=OCR_TEXT_COLOR, width=2)
    draw.text((badge_x1 + 6, badge_y1 + 3), label, fill=OCR_TEXT_COLOR, font=font)


def load_font(size: int, bold: bool) -> ImageFont.ImageFont:
    candidates = (
        ["DejaVuSans-Bold.ttf", "arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"]
        if bold
        else ["DejaVuSans.ttf", "arial.ttf", "C:/Windows/Fonts/arial.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def safe_visual_name(comic_id: str | None, file_name: str, suffix: str) -> str:
    safe_comic = sanitize_filename(comic_id or "unknown")
    safe_file = sanitize_filename(Path(file_name).stem)
    return f"{safe_comic}_{safe_file}_{suffix}.jpg"


def visual_comic_dir(root: str | Path, comic_id: str | None) -> Path:
    return Path(root) / sanitize_filename(comic_id or "unknown")


def visual_page_name(file_name: str | Path, suffix: str) -> str:
    return f"{sanitize_filename(Path(file_name).stem)}_{suffix}.jpg"


def sanitize_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
