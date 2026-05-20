from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from features.magi_schema import BoundingBox, MagiPageAnalysis
from features.ocr_grouping import OCRGroupedText
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
OCR_GROUP_COLOR = (0, 170, 90)


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

    badge_font = load_font(size=max(10, image.width // 110), bold=True)
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
            image_size=image.size,
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


def draw_ocr_grouped_overlay(
    image_path: str | Path,
    groups: list[OCRGroupedText],
    output_path: str | Path,
    sidebar_width: int = 620,
) -> None:
    image = Image.open(image_path).convert("RGB")
    sidebar_font = load_font(size=max(18, image.width // 65), bold=False)
    sidebar_bold = load_font(size=max(20, image.width // 58), bold=True)
    badge_font = load_font(size=max(14, image.width // 90), bold=True)
    line_height = max(30, sidebar_font.size + 12 if hasattr(sidebar_font, "size") else 30)
    header_height = 104
    sidebar_needed_height = header_height + max(1, len(groups)) * line_height * 2 + 32
    canvas_height = max(image.height, sidebar_needed_height)
    canvas_width = image.width + sidebar_width
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for display_index, group in enumerate(groups, 1):
        block_label_width = max(30, badge_font.size + 18 if hasattr(badge_font, "size") else 30)
        display_box = expanded_group_box_for_blocks(
            group.box,
            image_size=image.size,
            left_extra=block_label_width + 12,
        )
        group_badge_box = group_badge_outside_box(
            display_box,
            display_index=display_index,
            font=badge_font,
            draw=draw,
            image_size=image.size,
        )
        draw.rectangle(
            (display_box.x1, display_box.y1, display_box.x2, display_box.y2),
            outline=OCR_GROUP_COLOR,
            width=5,
        )
        for block in group.blocks:
            draw_ocr_block_outline_with_inline_number(
                draw=draw,
                block=block,
                font=badge_font,
                label_width=block_label_width,
            )
        draw_group_badge_inside(draw, group_badge_box, display_index, badge_font)

    sidebar_x = image.width
    draw.rectangle((sidebar_x, 0, canvas_width, canvas_height), fill=(248, 248, 248))
    draw.line((sidebar_x, 0, sidebar_x, canvas_height), fill=(210, 210, 210), width=3)
    draw.text((sidebar_x + 24, 24), "OCR grouped bubbles", fill=OCR_TEXT_COLOR, font=sidebar_bold)
    draw.text(
        (sidebar_x + 24, 58),
        f"{len(groups)} grupos calibrados",
        fill=(70, 70, 70),
        font=sidebar_font,
    )

    y = header_height
    for display_index, group in enumerate(groups, 1):
        confidence = f"{group.confidence:.2f}" if group.confidence is not None else "--"
        block_ids = ",".join(str(index) for index in group.block_indices)
        header = f"{display_index:02d} [{confidence}] blocks={block_ids}"
        draw.text((sidebar_x + 24, y), header[:72], fill=OCR_TEXT_COLOR, font=sidebar_bold)
        y += line_height
        draw.text((sidebar_x + 24, y), group.text[:78], fill=OCR_TEXT_COLOR, font=sidebar_font)
        y += line_height

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def draw_ocr_block_outline(draw: ImageDraw.ImageDraw, block: dict[str, Any]) -> None:
    polygon = block.get("polygon")
    if polygon:
        points = [(float(point[0]), float(point[1])) for point in polygon]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=OCR_BOX_COLOR, width=3)
            return
    box_payload = block.get("box") or {}
    draw.rectangle(
        (
            float(box_payload.get("x1", 0.0)),
            float(box_payload.get("y1", 0.0)),
            float(box_payload.get("x2", 0.0)),
            float(box_payload.get("y2", 0.0)),
        ),
        outline=OCR_BOX_COLOR,
        width=3,
    )


def draw_ocr_block_outline_with_inline_number(
    draw: ImageDraw.ImageDraw,
    block: dict[str, Any],
    font: ImageFont.ImageFont,
    label_width: int,
    padding: int = 4,
) -> None:
    box_payload = block.get("box") or {}
    x1 = float(box_payload.get("x1", 0.0))
    y1 = float(box_payload.get("y1", 0.0))
    x2 = float(box_payload.get("x2", 0.0))
    y2 = float(box_payload.get("y2", 0.0))
    box_height = max(1.0, y2 - y1)
    expanded_x1 = max(0.0, x1 - label_width)

    draw.rectangle((expanded_x1, y1, x2, y2), outline=OCR_BOX_COLOR, width=3)
    draw.line((x1, y1, x1, y2), fill=OCR_BOX_COLOR, width=2)

    label = str(int(block.get("index") or 0) + 1)
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_x = expanded_x1 + max(2, (label_width - text_width) / 2)
    text_y = y1 + max(0, (box_height - text_height) / 2) - 1
    draw.text((text_x, text_y), label, fill=OCR_TEXT_COLOR, font=font)


def draw_group_badge(
    draw: ImageDraw.ImageDraw,
    box: BoundingBox,
    display_index: int,
    font: ImageFont.ImageFont,
) -> None:
    label = f"{display_index:02d}"
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    diameter = max(22, max(text_width, text_height) + 10)
    x = max(2, box.x1 - diameter - 3)
    y = max(2, box.y1 - diameter - 3)
    draw.ellipse((x, y, x + diameter, y + diameter), fill=(120, 235, 170), outline=OCR_TEXT_COLOR, width=2)
    draw.text(
        (x + (diameter - text_width) / 2, y + (diameter - text_height) / 2 - 1),
        label,
        fill=OCR_TEXT_COLOR,
        font=font,
    )


def expanded_group_box_for_blocks(
    box: BoundingBox,
    image_size: tuple[int, int],
    left_extra: int,
    padding: int = 8,
) -> BoundingBox:
    return BoundingBox(
        x1=max(0.0, box.x1 - left_extra - padding),
        y1=max(0.0, box.y1 - padding),
        x2=min(float(image_size[0]), box.x2 + padding),
        y2=min(float(image_size[1]), box.y2 + padding),
    )


def group_badge_outside_box(
    box: BoundingBox,
    display_index: int,
    font: ImageFont.ImageFont,
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    margin: int = 4,
) -> tuple[float, float, float, float]:
    label = f"{display_index:02d}"
    label_box = draw.textbbox((0, 0), label, font=font)
    text_width = label_box[2] - label_box[0]
    text_height = label_box[3] - label_box[1]
    diameter = max(24, max(text_width, text_height) + 12)
    x1 = box.x1 - diameter - margin
    y1 = box.y1 - diameter - margin
    if x1 < margin:
        x1 = box.x1 + margin
    if y1 < margin:
        y1 = box.y1 + margin
    x1 = min(max(margin, x1), max(margin, image_size[0] - diameter - margin))
    y1 = min(max(margin, y1), max(margin, image_size[1] - diameter - margin))
    return (x1, y1, x1 + diameter, y1 + diameter)


def draw_group_badge_inside(
    draw: ImageDraw.ImageDraw,
    badge_box: tuple[float, float, float, float],
    display_index: int,
    font: ImageFont.ImageFont,
) -> None:
    label = f"{display_index:02d}"
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x1, y1, x2, y2 = badge_box
    draw.ellipse((x1, y1, x2, y2), fill=OCR_BADGE_FILL, outline=OCR_TEXT_COLOR, width=3)
    draw.text(
        (x1 + ((x2 - x1) - text_width) / 2, y1 + ((y2 - y1) - text_height) / 2 - 1),
        label,
        fill=OCR_TEXT_COLOR,
        font=font,
    )


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
    image_size: tuple[int, int],
) -> None:
    box_payload = block.get("box") or {}
    x1 = float(box_payload.get("x1", 0.0))
    y1 = float(box_payload.get("y1", 0.0))
    x2 = float(box_payload.get("x2", 0.0))
    y2 = float(box_payload.get("y2", 0.0))
    polygon = block.get("polygon")
    points: list[tuple[float, float]] = []
    if polygon:
        points = [(float(point[0]), float(point[1])) for point in polygon]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=OCR_BOX_COLOR, width=5)
    else:
        draw.rectangle((x1, y1, x2, y2), outline=OCR_BOX_COLOR, width=5)

    label = f"{display_index:02d}"
    label_box = draw.textbbox((0, 0), label, font=font)
    text_width = label_box[2] - label_box[0]
    text_height = label_box[3] - label_box[1]
    diameter = max(18, max(text_width, text_height) + 8)
    badge_x1, badge_y1 = choose_badge_position(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        diameter=diameter,
        polygon_points=points,
        image_size=image_size,
    )
    badge_x2 = badge_x1 + diameter
    badge_y2 = badge_y1 + diameter
    draw.ellipse((badge_x1, badge_y1, badge_x2, badge_y2), fill=OCR_BADGE_FILL, outline=OCR_TEXT_COLOR, width=2)
    text_x = badge_x1 + (diameter - text_width) / 2
    text_y = badge_y1 + (diameter - text_height) / 2 - 1
    draw.text((text_x, text_y), label, fill=OCR_TEXT_COLOR, font=font)


def choose_badge_position(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    diameter: int,
    polygon_points: list[tuple[float, float]] | None = None,
    image_size: tuple[int, int] | None = None,
    margin: int = 2,
) -> tuple[float, float]:
    if polygon_points and len(polygon_points) >= 3:
        polygon_position = choose_polygon_badge_position(
            points=polygon_points,
            diameter=diameter,
            image_size=image_size,
            margin=margin,
        )
        if polygon_position is not None:
            return polygon_position

    # Preferred position: attached to the outside top-left corner of the OCR box.
    preferred_x = x1 - diameter - margin
    preferred_y = y1 - diameter - margin
    if preferred_x >= margin and preferred_y >= margin:
        return preferred_x, preferred_y

    # If the top-left outside corner would leave the image, keep the badge
    # attached to the nearest outside edge instead of letting it float away.
    candidates = [
        (x1, y1 - diameter - margin),  # above, aligned with left edge
        (x1 - diameter - margin, y1),  # left, aligned with top edge
        (x2 + margin, y1),  # right, aligned with top edge
        (x1, y2 + margin),  # below, aligned with left edge
    ]
    for candidate_x, candidate_y in candidates:
        if candidate_x >= margin and candidate_y >= margin:
            return candidate_x, candidate_y

    # Last resort for boxes touching both top and left borders.
    return max(margin, x1 + margin), max(margin, y1 + margin)


def choose_polygon_badge_position(
    points: list[tuple[float, float]],
    diameter: int,
    image_size: tuple[int, int] | None,
    margin: int,
) -> tuple[float, float] | None:
    corner_x, corner_y = min(points, key=lambda point: (point[0] + point[1], point[1], point[0]))
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    vector_x = corner_x - center_x
    vector_y = corner_y - center_y
    magnitude = max((vector_x**2 + vector_y**2) ** 0.5, 1.0)
    unit_x = vector_x / magnitude
    unit_y = vector_y / magnitude
    radius = diameter / 2

    badge_center_x = corner_x + unit_x * (radius + margin)
    badge_center_y = corner_y + unit_y * (radius + margin)
    badge_x = badge_center_x - radius
    badge_y = badge_center_y - radius

    if image_size is not None:
        max_x = max(margin, image_size[0] - diameter - margin)
        max_y = max(margin, image_size[1] - diameter - margin)
        badge_x = min(max(badge_x, margin), max_x)
        badge_y = min(max(badge_y, margin), max_y)

    return badge_x, badge_y


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
