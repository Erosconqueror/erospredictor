"""Compact descriptive chips beside prediction, with evidence in tooltips."""

from html import escape
from functools import lru_cache


@lru_cache(maxsize=512)
def _chip_image(text, tone, score):
    """Qt rich text has no rounded inline borders; use a cached, high-DPI chip."""
    from PyQt6.QtCore import QBuffer, QIODevice, QRectF, Qt
    from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
    level = max(0, min(1, ((3-score) if tone == "negative" else (score-3))/2))
    if tone == "descriptive":
        level = abs(score-3)/2
    muted, vivid = {"positive": ("#769B80", "#9AF7AE"), "negative": ("#AF7B80", "#FF929D"),
                    "descriptive": ("#719A9D", "#91E4E6")}[tone]
    a, b = QColor(muted), QColor(vivid)
    tint = QColor(*(round(x+(y-x)*level) for x,y in zip(a.getRgb()[:3], b.getRgb()[:3])))
    font = QFont("Segoe UI")
    font.setPixelSize(11)
    font.setBold(level >= .75)
    metrics = QFontMetrics(font)
    width, height = metrics.horizontalAdvance(text)+14, metrics.height()+8
    image = QImage(width*2, height*2, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.scale(2, 2)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(tint, 1.5 if level >= .75 else 1))
    background = QColor(tint)
    background.setAlpha(18 if level < .75 else 35)
    painter.setBrush(background)
    painter.drawRoundedRect(QRectF(1, 1, width-2, height-2), 4, 4)
    painter.setFont(font)
    painter.drawText(QRectF(7, 3, width-14, height-6), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return "data:image/png;base64,"+bytes(buffer.data().toBase64()).decode("ascii"), width, height


def chip_html(label, tone, score=3, tooltip="", href=None):
    prefix = "+ " if tone == "positive" else "− " if tone == "negative" else "↔ "
    uri, width, height = _chip_image(prefix+label, tone, round(score, 2))
    html = (f"<img src='{uri}' width='{width}' height='{height}' "
            f"alt='{escape(prefix+label, quote=True)}' title='{escape(tooltip, quote=True)}'>")
    return (f"<a href='{escape(href, quote=True)}' title='{escape(tooltip, quote=True)}'>{html}</a>"
            if href else html)


def summary_html(summary, title, color):
    html = f"<b style='color:{color};'>{escape(title)}</b>"
    tooltip = [summary.get("note", "")]
    if summary.get("tags"):
        html += "<br>"
    for tag in summary.get("tags", []):
        tone = "descriptive" if tag["descriptive"] else "positive" if tag["direction"] == "high" else "negative"
        html += chip_html(tag["label"], tone, tag["score"]) + " "
        tooltip.append(f"{tag['label']}: {tag['score']:.2f}/5; referenciaátlag {tag['mean']:.2f}/5; "
                       f"eltérés {tag['delta']:+.2f}; {tag['known']}/{summary['picks']} ismert profil.")
    if not summary.get("tags"):
        html += "<br><span style='color:#A0A0B5;'>" + escape(summary.get("note", "Nincs adat")) + "</span>"
    html += f"<br><span style='font-size:10px;color:#9CA7BB;'>Kísérleti profil · {summary.get('picks', 0)}/5 hős</span>"
    return html, "\n".join(tooltip)


def digest_html(digest):
    if not digest.get("character") and not digest.get("needs"):
        return "<span style='color:#9CA7BB;'>" + escape(digest.get("note", "Kevés profiladat.")) + "</span>"
    lines = []
    for key, fallback, caption, tone in (("character_tags", "character", "Csapat", "positive"),
                                         ("need_tags", "needs", "Szükség lehet", "negative")):
        tags = digest.get(key, [{"label": label, "score": 3} for label in digest.get(fallback, [])])
        if tags:
            lines.append(f"<b>{caption}:</b> " + " ".join(chip_html(tag["label"], tag.get("tone", tone),
                         tag.get("score", 3), f"{tag.get('score', 3):.2f}/5 · kísérleti csapatprofil") for tag in tags))
    return "<br>".join(lines)
