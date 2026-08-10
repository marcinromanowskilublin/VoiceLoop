from __future__ import annotations

import argparse
import html
import re
import subprocess
import tempfile
from pathlib import Path


def inline_markup(value: str) -> str:
    code_spans: list[str] = []

    def save_code(match: re.Match[str]) -> str:
        code_spans.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00CODE{len(code_spans) - 1}\x00"

    value = re.sub(r"`([^`]+)`", save_code, value)
    value = html.escape(value)
    value = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(match.group(2), quote=True)}">{match.group(1)}</a>'
        ),
        value,
    )
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", value)
    for index, code in enumerate(code_spans):
        value = value.replace(f"\x00CODE{index}\x00", code)
    return value


def slugify(value: str, used: set[str]) -> str:
    slug = value.casefold().replace("ł", "l")
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-") or "section"
    candidate = slug
    counter = 2
    while candidate in used:
        candidate = f"{slug}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def markdown_to_html(markdown: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = markdown.replace("\r\n", "\n").split("\n")
    output: list[str] = []
    headings: list[tuple[int, str, str]] = []
    used_ids: set[str] = set()
    paragraph: list[str] = []
    in_code = False
    code_language = ""
    code_lines: list[str] = []
    list_type: str | None = None
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{inline_markup(' '.join(part.strip() for part in paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            output.append(f"</{list_type}>")
            list_type = None

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if in_code:
            if stripped.startswith("```"):
                language_class = (
                    f' class="language-{html.escape(code_language, quote=True)}"'
                    if code_language
                    else ""
                )
                output.append(
                    f"<pre><code{language_class}>"
                    + html.escape("\n".join(code_lines))
                    + "</code></pre>"
                )
                in_code = False
                code_language = ""
                code_lines.clear()
            else:
                code_lines.append(line)
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            in_code = True
            code_language = stripped[3:].strip()
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            identifier = slugify(re.sub(r"[*_`]", "", title), used_ids)
            headings.append((level, title, identifier))
            output.append(
                f'<h{level} id="{identifier}">{inline_markup(title)}</h{level}>'
            )
            index += 1
            continue

        if stripped in {"---", "***", "___"}:
            flush_paragraph()
            close_list()
            output.append("<hr>")
            index += 1
            continue

        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*\|?\s*$", lines[index + 1])
        ):
            flush_paragraph()
            close_list()
            headers = [cell.strip() for cell in stripped.strip("|").split("|")]
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(
                    [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                )
                index += 1
            output.append("<table><thead><tr>")
            output.extend(f"<th>{inline_markup(cell)}</th>" for cell in headers)
            output.append("</tr></thead><tbody>")
            for row in rows:
                output.append("<tr>")
                padded = row + [""] * max(0, len(headers) - len(row))
                output.extend(
                    f"<td>{inline_markup(cell)}</td>" for cell in padded[: len(headers)]
                )
                output.append("</tr>")
            output.append("</tbody></table>")
            continue

        unordered = re.match(r"^\s*[-+*]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if unordered or ordered:
            flush_paragraph()
            desired = "ul" if unordered else "ol"
            if list_type != desired:
                close_list()
                output.append(f"<{desired}>")
                list_type = desired
            item = (unordered or ordered).group(1)
            checkbox = re.match(r"^\[([ xX])\]\s+(.+)$", item)
            if checkbox:
                checked = " checked" if checkbox.group(1).lower() == "x" else ""
                rendered = (
                    f'<input type="checkbox" disabled{checked}> '
                    f"{inline_markup(checkbox.group(2))}"
                )
            else:
                rendered = inline_markup(item)
            output.append(f"<li>{rendered}</li>")
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            close_list()
            index += 1
            continue

        close_list()
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    close_list()
    if in_code:
        output.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    return "\n".join(output), headings


def build_document(markdown: str, source_name: str) -> str:
    body, headings = markdown_to_html(markdown)
    title = next((title for level, title, _ in headings if level == 1), source_name)
    toc_items = [
        f'<li class="level-{level}"><a href="#{identifier}">{inline_markup(text)}</a></li>'
        for level, text, identifier in headings
        if level in {2, 3}
    ]
    toc = (
        '<section class="toc"><h2>Spis treści</h2><ul>'
        + "\n".join(toc_items)
        + "</ul></section>"
    )
    first_h1_end = body.find("</h1>")
    if first_h1_end >= 0:
        first_rule_end = body.find("<hr>", first_h1_end)
        insert_at = (
            first_rule_end + len("<hr>")
            if first_rule_end >= 0
            else first_h1_end + len("</h1>")
        )
        body = body[:insert_at] + toc + body[insert_at:]
    else:
        body = toc + body

    css = """
    :root {
      --ink: #172033;
      --muted: #5c667a;
      --accent: #2459d1;
      --line: #dce2ec;
      --soft: #f4f6fa;
      --code: #eef1f6;
    }
    @page { size: A4; margin: 16mm 15mm 18mm; }
    * { box-sizing: border-box; }
    html { font-family: "Segoe UI", Arial, sans-serif; color: var(--ink); }
    body { margin: 0; font-size: 10.4pt; line-height: 1.46; }
    article { max-width: 180mm; margin: 0 auto; }
    h1 {
      font-size: 27pt; line-height: 1.12; color: var(--accent);
      margin: 32mm 0 8mm; letter-spacing: -0.4pt;
    }
    article > hr:first-of-type { margin-top: 18mm; break-after: page; }
    h2 {
      font-size: 17pt; color: var(--accent); margin: 13mm 0 4mm;
      padding-bottom: 2mm; border-bottom: 1px solid var(--line);
      break-after: avoid;
    }
    h3 { font-size: 13pt; margin: 7mm 0 2.5mm; break-after: avoid; }
    h4 { font-size: 11pt; margin: 5mm 0 2mm; break-after: avoid; }
    p { margin: 0 0 3.2mm; orphans: 3; widows: 3; }
    a { color: var(--accent); text-decoration: none; }
    strong { font-weight: 650; }
    hr { border: 0; border-top: 1px solid var(--line); margin: 9mm 0; }
    ul, ol { margin: 1.5mm 0 4mm 6mm; padding-left: 5mm; }
    li { margin: 1.2mm 0; }
    code {
      font-family: Consolas, "Cascadia Mono", monospace; font-size: 9pt;
      background: var(--code); padding: 0.25mm 1mm; border-radius: 1mm;
    }
    pre {
      background: var(--code); border-left: 3px solid var(--accent);
      padding: 4mm; margin: 3mm 0 5mm; white-space: pre-wrap;
      overflow-wrap: anywhere; break-inside: avoid;
    }
    pre code { padding: 0; background: transparent; font-size: 8.6pt; }
    table {
      width: 100%; border-collapse: collapse; margin: 3mm 0 6mm;
      font-size: 9.1pt; break-inside: avoid;
    }
    th, td {
      border: 1px solid var(--line); padding: 2.2mm 2.5mm;
      vertical-align: top; text-align: left;
    }
    th { background: var(--soft); font-weight: 650; }
    tr { break-inside: avoid; }
    .toc {
      background: var(--soft); padding: 7mm 8mm; margin: 12mm 0;
      break-after: page;
    }
    .toc h2 { margin-top: 0; border: 0; color: var(--ink); }
    .toc ul {
      columns: 2; column-gap: 10mm; margin: 0; padding: 0;
      list-style: none;
    }
    .toc li { break-inside: avoid; margin: 1mm 0; }
    .toc .level-3 { margin-left: 3mm; font-size: 9pt; color: var(--muted); }
    input[type="checkbox"] { width: 3.5mm; height: 3.5mm; vertical-align: -0.6mm; }
    @media print {
      h2 { break-before: auto; }
      .toc { break-after: page; }
      a { color: inherit; }
    }
    """
    return f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <article>{body}</article>
</body>
</html>
"""


def locate_chrome(explicit: str | None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError("Nie znaleziono Chrome ani Edge do wygenerowania PDF.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VoiceLoop HTML/PDF documentation.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--html", dest="html_path", type=Path)
    parser.add_argument("--pdf", dest="pdf_path", type=Path)
    parser.add_argument("--chrome")
    args = parser.parse_args()

    source = args.source.resolve()
    html_path = (args.html_path or source.with_suffix(".html")).resolve()
    pdf_path = args.pdf_path.resolve() if args.pdf_path else None

    markdown = source.read_text(encoding="utf-8")
    html_document = build_document(markdown, source.stem)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_document, encoding="utf-8")
    print(f"HTML: {html_path}")

    if pdf_path:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        chrome = locate_chrome(args.chrome)
        with tempfile.TemporaryDirectory(prefix="voiceloop-docs-") as profile:
            command = [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(
                "Chrome PDF build failed: "
                + (result.stderr.strip() or result.stdout.strip() or str(result.returncode))
            )
        print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
