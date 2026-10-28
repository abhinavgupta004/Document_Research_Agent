"""Turn the model's [S1] tags into readable (file, page) citations."""
import re

_TAG = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def link_citations(text: str, passages: list[dict]) -> tuple[str, list[str]]:
    """Replace [S1]-style tags with [file.pdf, p.4].

    Returns the rewritten text and the ids that were actually cited, in order of
    first appearance. Tags pointing at passages that don't exist (the model
    occasionally invents one) are dropped rather than shown to the reader.
    """
    by_id = {p["id"]: p for p in passages}
    used: list[str] = []

    def _replace(match: re.Match) -> str:
        labels = []
        for tag in (t.strip() for t in match.group(1).split(",")):
            p = by_id.get(tag)
            if p is None:
                continue
            if tag not in used:
                used.append(tag)
            labels.append(f"{p['source']}, p.{p['page']}")
        return f"[{'; '.join(labels)}]" if labels else ""

    return _TAG.sub(_replace, text), used
