"""
Vision module — uses Claude claude-sonnet-4-6 to extract dimensions from
package/sorter images (images include a ruler).
"""

from __future__ import annotations
import base64
import os
import re
from dataclasses import dataclass

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


@dataclass
class DimExtraction:
    length_cm: float | None
    width_cm: float | None
    height_cm: float | None
    confidence: float   # 0–1
    raw_response: str
    error: str | None = None


def _fetch_image_b64(url: str) -> tuple[str, str]:
    """Download image and return (base64_data, media_type)."""
    resp = httpx.get(url, timeout=15, follow_redirects=True)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return base64.standard_b64encode(resp.content).decode(), content_type


EXTRACTION_PROMPT = """
You are analyzing a product/package image that may contain a ruler for scale.
Extract the physical dimensions of the main package/parcel in centimetres.

Return ONLY a JSON object with these keys:
{
  "length_cm": <number or null>,
  "width_cm": <number or null>,
  "height_cm": <number or null>,
  "confidence": <0.0–1.0>,
  "notes": "<brief explanation>"
}

If no ruler is visible or dimensions cannot be determined, set values to null and confidence to 0.
Do not return any text outside the JSON.
"""


def extract_dims_from_url(url: str) -> DimExtraction:
    """
    Download image from URL and ask Claude to extract L×W×H in cm.
    """
    try:
        b64, media_type = _fetch_image_b64(url)
    except Exception as e:
        return DimExtraction(None, None, None, 0.0, "", error=f"fetch_error: {e}")

    try:
        client = _get_client()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }],
        )
        raw = msg.content[0].text.strip()
    except Exception as e:
        return DimExtraction(None, None, None, 0.0, "", error=f"api_error: {e}")

    # Parse JSON from response
    try:
        import json
        # strip code fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(cleaned)
        return DimExtraction(
            length_cm=data.get("length_cm"),
            width_cm=data.get("width_cm"),
            height_cm=data.get("height_cm"),
            confidence=float(data.get("confidence", 0)),
            raw_response=raw,
        )
    except Exception as e:
        return DimExtraction(None, None, None, 0.0, raw, error=f"parse_error: {e}")


def extract_dims_from_urls(urls: list[str]) -> DimExtraction:
    """
    Try each URL in order; return the first result with confidence > 0.5,
    otherwise return the highest-confidence result.
    """
    results = [extract_dims_from_url(u) for u in urls]
    good = [r for r in results if r.confidence > 0.5]
    if good:
        return max(good, key=lambda r: r.confidence)
    if results:
        return max(results, key=lambda r: r.confidence)
    return DimExtraction(None, None, None, 0.0, "", error="no_urls")
