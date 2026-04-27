from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

class TraceLoader:
    def __init__(self, file_path: str | Path):
        self.path = Path(file_path)
        self.tree = self._load_trc_tree()

    def list_traces(self) -> list[str]:
        """Return the trace names available in a TRC file."""
        return self._available_trace_tags()

    def _available_trace_tags(self) -> list[str]:
        """Return all TRACE* tags present under TRACE_DATA."""
        trace_data = self._find_trace_data_root()
        return [child.tag for child in trace_data if child.tag.upper().startswith("TRACE")]

    def _normalize_trace_tag(self, trace: int | str) -> str:
        """Normalize user input like 1 or 'trace1' into a TRACE* tag name."""
        if isinstance(trace, int):
            return f"TRACE{trace}"
        text = trace.strip()
        if text.upper().startswith("TRACE"):
            return text.upper()
        return text

    def _find_trace_data_root(self) -> ET.Element:
        """Find the TRACE_DATA element in the TRC XML tree."""
        root = self.tree.getroot()
        trace_data = root.find(".//TRACE_DATA")
        if trace_data is None:
            raise ValueError("TRACE_DATA section not found in TRC file")
        return trace_data
   
    def _load_trc_tree(self) -> ET.ElementTree:
        """Load TRC XML and tolerate trailing non-XML text after the root element."""
        path = self.path

        try:
            return ET.parse(path)
        except ET.ParseError:
            text = path.read_text(encoding="utf-8", errors="ignore")

            # Remove control chars that are invalid in XML 1.0.
            text = re.sub(r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]", "", text)

            # Keep only the first complete root element if trailing junk exists.
            root_match = re.search(r"<\s*([A-Za-z_][\w\-.]*)\b[^>]*>", text)
            if root_match is not None:
                root_tag = root_match.group(1)
                end_tag = f"</{root_tag}>"
                end_idx = text.find(end_tag)
                if end_idx != -1:
                    text = text[: end_idx + len(end_tag)]

            try:
                root = ET.fromstring(text)
            except ET.ParseError as exc:
                raise ValueError(f"Unable to parse TRC XML in {path}: {exc}") from exc

            return ET.ElementTree(root)

    def _parse_float_array(self, raw: str | None) -> np.ndarray:
        if not raw:
            return np.array([], dtype=float)

        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            values.append(float(token))
        return np.asarray(values, dtype=float)

    def load_trace(self, trace: int | str = 1,
        freq_scale: float = 1.0,
        spectrum_scale: float = 1.0,
        spectrum_offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load one trace from an XML-based .trc file.

        Parameters
        ----------
        trace:
            Trace number like 1 or tag name like "TRACE1".
        freq_scale:
            Scale factor applied to the frequency axis.
        spectrum_scale:
            Scale factor applied to the spectrum values.
        spectrum_offset:
            Offset added after scaling the spectrum values.
        """ 
        trace_tag = self._normalize_trace_tag(trace)
        trace_data = self.load_traces(
            traces=[trace],
            freq_scale=freq_scale,
            spectrum_scale=spectrum_scale,
            spectrum_offset=spectrum_offset,
        )
        series = trace_data[trace_tag]
        return series["frequency"], series["spectrum"]

    def load_traces(self, traces: list[int | str] | None = None,
        freq_scale: float = 1.0, spectrum_scale: float = 1.0, spectrum_offset: float = 0.0,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Load multiple traces and return a dict of frequency/spectrum arrays by trace tag."""
        trace_data = self._find_trace_data_root()

        x_values = self._parse_float_array(trace_data.get("X")) * freq_scale
        if x_values.size == 0:
            raise ValueError("Empty frequency axis (TRACE_DATA X attribute)")

        available = self._available_trace_tags()
        if traces is None:
            trace_tags = available
        else:
            trace_tags = [self._normalize_trace_tag(t) for t in traces]

        data: dict[str, dict[str, np.ndarray]] = {}
        for trace_tag in trace_tags:
            trace_element = trace_data.find(trace_tag)
            if trace_element is None:
                avail_text = ", ".join(available)
                raise ValueError(f"Trace {trace_tag!r} not found. Available traces: {avail_text}")

            y_values = self._parse_float_array(trace_element.get("Data"))
            y_values = y_values * spectrum_scale + spectrum_offset
            if y_values.size == 0:
                raise ValueError(f"Empty spectrum data in trace {trace_tag!r}")

            sample_count = min(x_values.size, y_values.size)
            data[trace_tag] = {
                "frequency": x_values[:sample_count],
                "spectrum": y_values[:sample_count],
            }

        return data


