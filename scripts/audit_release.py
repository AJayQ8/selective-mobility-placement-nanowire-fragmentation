#!/usr/bin/env python3
"""Validate the compact public-repository release."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "science_lab" / "papers" / "selective_mobility_scripta"
DATA = PAPER / "source_data"
FIGURES = PAPER / "figures"
SKIPPED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "reproduced_artifacts",
}
FORBIDDEN_DIRECTORY_NAMES = {
    ".idea",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".vscode",
    "__pycache__",
    "htmlcov",
    "raw_results",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".npy", ".tex", ".bib", ".docx"}
FORBIDDEN_BACKUP_SUFFIXES = (".bak", ".orig", ".rej", ".swp", ".swo", "~")
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".svg",
    ".yml",
    ".yaml",
}
PRIVATE_PATH_PATTERNS = (
    re.compile(r"/" + "Users/"),
    re.compile(r"/" + "home/"),
    re.compile(r"/" + "tmp/"),
    re.compile("file" + "://"),
)
SECRET_PATTERNS = (
    re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    re.compile("github_" + r"pat_[A-Za-z0-9_]{20,}"),
    re.compile("sk-" + r"[A-Za-z0-9]{20,}"),
    re.compile("AKIA" + r"[0-9A-Z]{16}"),
    re.compile("BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY"),
)
MAX_PUBLIC_FILE_BYTES = 50 * 1024 * 1024
REQUIRED_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "ARCHIVE_SCOPE.md",
    "CITATION.cff",
    "CODE_MAP.md",
    "LICENSE",
    "LICENSE.md",
    "LICENSES/CC-BY-4.0.txt",
    "LICENSES/GPL-3.0.txt",
    "Makefile",
    "README.md",
    "REPRODUCIBILITY.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements-lock.txt",
    ".github/workflows/validate.yml",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def check_manifest() -> list[str]:
    failures: list[str] = []
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        return ["MANIFEST.sha256 is missing"]
    listed: set[Path] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            failures.append(f"malformed manifest line: {line!r}")
            continue
        expected, relative = match.groups()
        path = ROOT / relative
        listed.add(path)
        if not path.is_file():
            failures.append(f"manifest target missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"manifest hash mismatch: {relative}")
    actual = {
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest
        and not {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "reproduced_artifacts"}.intersection(
            path.relative_to(ROOT).parts
        )
        and path.name != ".DS_Store"
        and path.suffix not in {".pyc", ".pyo"}
    }
    if actual != listed:
        for path in sorted(actual - listed):
            failures.append(f"file absent from manifest: {path.relative_to(ROOT)}")
        for path in sorted(listed - actual):
            failures.append(f"manifest lists non-release file: {path.relative_to(ROOT)}")
    return failures


def check_clean_boundary() -> list[str]:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if SKIPPED_DIRECTORY_NAMES.intersection(relative.parts):
            continue
        if path.is_symlink():
            failures.append(f"symbolic link is not allowed in release: {relative}")
            continue
        if path.is_dir() and FORBIDDEN_DIRECTORY_NAMES.intersection(relative.parts):
            failures.append(f"forbidden directory: {relative}")
        if path.is_dir() and re.fullmatch(r"generated_v\d+", path.name):
            failures.append(f"superseded generated directory: {relative}")
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            failures.append(f"oversized public artifact: {relative}")
        if (
            path.suffix in FORBIDDEN_SUFFIXES
            or path.name == ".DS_Store"
            or path.name.endswith(FORBIDDEN_BACKUP_SUFFIXES)
        ):
            failures.append(f"forbidden file: {relative}")
        if FIGURES in path.parents and re.search(r"_v\d+\.py$", path.name):
            failures.append(f"superseded versioned Python file: {relative}")
        if path.name == "generation_receipt.json" and "reproduced_artifacts" not in relative.parts:
            failures.append(f"generation receipt is not a canonical artifact: {relative}")
        if (
            path.name.startswith(".env")
            or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
        ):
            failures.append(f"sensitive filename in release: {relative}")
        if path.suffix in TEXT_SUFFIXES or path.name in {
            "LICENSE",
            "Makefile",
            ".gitattributes",
            ".gitignore",
        }:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failures.append(f"non-UTF8 text artifact: {relative}")
                continue
            for pattern in PRIVATE_PATH_PATTERNS:
                if pattern.search(text):
                    failures.append(f"private path token in {relative}")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    failures.append(f"credential-like token in {relative}")
            system_font_prefix = "/System/" + "Library/Fonts/"
            windows_font_prefix = "\\\\Windows\\\\" + "Fonts\\\\"
            if system_font_prefix in text or windows_font_prefix in text:
                failures.append(f"machine font path in {relative}")
    return failures


def check_repository_metadata() -> list[str]:
    failures: list[str] = []
    for relative in sorted(REQUIRED_ROOT_FILES):
        if not (ROOT / relative).is_file():
            failures.append(f"required repository file missing: {relative}")

    root_license = ROOT / "LICENSE"
    canonical_gpl = ROOT / "LICENSES" / "GPL-3.0.txt"
    if root_license.is_file() and canonical_gpl.is_file():
        if root_license.read_bytes() != canonical_gpl.read_bytes():
            failures.append("root LICENSE is not the canonical GPL-3.0 text")

    notice_path = ROOT / "THIRD_PARTY_NOTICES.md"
    if notice_path.is_file():
        notice = notice_path.read_text(encoding="utf-8")
        required_notice_fragments = (
            "Abhinav Roy",
            "Arjun Varma R.",
            "M. P. Gururajan",
            "10.1063/5.0064917",
            "a275dc0639c8f1e3dad795234e4a5ea7a7aeb5b3",
            "GPL-3.0-only",
        )
        for fragment in required_notice_fragments:
            if fragment not in notice:
                failures.append(f"third-party notice missing: {fragment}")
        for incorrect_name in ("Rakesh Roy", "Rahul Kumar", "Chandan P. Varma"):
            if incorrect_name in notice:
                failures.append(f"incorrect upstream attribution remains: {incorrect_name}")

    citation_path = ROOT / "CITATION.cff"
    if citation_path.is_file():
        citation = citation_path.read_text(encoding="utf-8")
        required_citation_fragments = (
            'cff-version: 1.2.0',
            'family-names: "Al-Zanki"',
            'given-names: "Ayas"',
            'version: "1.0.1"',
            "license: GPL-3.0-only",
            "https://github.com/AJayQ8/selective-mobility-placement-nanowire-fragmentation",
        )
        for fragment in required_citation_fragments:
            if fragment not in citation:
                failures.append(f"citation metadata missing: {fragment}")

    requirements_path = ROOT / "requirements-lock.txt"
    if requirements_path.is_file():
        requirements = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for requirement in requirements:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", requirement):
                failures.append(f"dependency is not exactly pinned: {requirement}")
        pypdf_requirements = [
            requirement
            for requirement in requirements
            if requirement.lower().startswith("pypdf==")
        ]
        if len(pypdf_requirements) != 1:
            failures.append("requirements must contain exactly one pypdf pin")
        else:
            version_text = pypdf_requirements[0].split("==", 1)[1]
            match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version_text)
            if not match or tuple(map(int, match.groups())) < (6, 15, 0):
                failures.append("pypdf must be pinned at 6.15.0 or newer")

    stale_venv_path = "science_lab/" + ".venv/bin/python"
    for path in ROOT.rglob("*.md"):
        if SKIPPED_DIRECTORY_NAMES.intersection(path.relative_to(ROOT).parts):
            continue
        if stale_venv_path in path.read_text(encoding="utf-8"):
            failures.append(
                f"documentation uses stale virtual-environment path: {path.relative_to(ROOT)}"
            )

    readme_path = ROOT / "README.md"
    if readme_path.is_file():
        readme = readme_path.read_text(encoding="utf-8")
        for command in ("make setup", "make test", "make test-all"):
            if command not in readme:
                failures.append(f"README omits validation command: {command}")

    workflow_path = ROOT / ".github" / "workflows" / "validate.yml"
    if workflow_path.is_file():
        workflow = workflow_path.read_text(encoding="utf-8")
        for command in (
            "make test PYTHON=python",
            "make test-all PYTHON=python",
            "make figures PYTHON=python",
        ):
            if command not in workflow:
                failures.append(f"CI omits validation command: {command}")

        for forbidden in ("make document", "Tectonic", "DOCUMENT_SOURCE_DIR"):
            if forbidden in workflow:
                failures.append(f"CI contains private document-build plumbing: {forbidden}")
        if not re.search(r"(?m)^permissions:\s*\n\s+contents:\s+read\s*$", workflow):
            failures.append("CI does not declare read-only repository permissions")

    makefile_path = ROOT / "Makefile"
    if makefile_path.is_file():
        makefile = makefile_path.read_text(encoding="utf-8")
        for forbidden in ("document:", "DOCUMENT_SOURCE_DIR", "DOCUMENT_OUTPUT_DIR"):
            if forbidden in makefile:
                failures.append(f"Makefile contains private document-build plumbing: {forbidden}")
        for target in ("test:", "test-all:", "figures:", "provenance:", "manifest:", "clean:"):
            if target not in makefile:
                failures.append(f"Makefile omits public target: {target}")
    return failures


def check_local_markdown_links() -> list[str]:
    failures: list[str] = []
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    scheme_pattern = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
    for path in ROOT.rglob("*.md"):
        if SKIPPED_DIRECTORY_NAMES.intersection(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith("#") or scheme_pattern.match(target):
                continue
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            candidate = (path.parent / target).resolve()
            try:
                candidate.relative_to(ROOT.resolve())
            except ValueError:
                failures.append(
                    f"local Markdown link escapes repository in {path.relative_to(ROOT)}: {raw_target}"
                )
                continue
            if not candidate.exists():
                failures.append(
                    f"broken local Markdown link in {path.relative_to(ROOT)}: {raw_target}"
                )
    return failures


def check_core_numbers() -> list[str]:
    failures: list[str] = []
    panel = load_csv("paired_panel.csv")
    source_labels = {"2292": "A", "104729": "B", "130363": "C"}
    by_source_case = {
        (source_labels[row["seed"]], row["case_id"]): row for row in panel
    }
    expected = {
        ("A", "untreated"): 1695.0,
        ("B", "untreated"): 1685.0,
        ("C", "untreated"): 1655.0,
        ("A", "c18p5"): 750.0,
        ("B", "c18p5"): 760.0,
        ("C", "c18p5"): 770.0,
        ("A", "c26p5"): 290.0,
        ("B", "c26p5"): 270.0,
        ("C", "c26p5"): 260.0,
        ("A", "c34p5"): -110.0,
        ("B", "c34p5"): -110.0,
        ("C", "c34p5"): -100.0,
    }
    for key, value in expected.items():
        row = by_source_case.get(key)
        if row is None:
            failures.append(f"paired-panel row missing: {key}")
            continue
        column = "event_midpoint" if key[1] == "untreated" else "delta_vs_paired_untreated"
        if float(row[column]) != value:
            failures.append(f"paired-panel mismatch {key}: {row[column]} != {value}")

    contrast = {
        (row["source_label"], float(row["m_min"]), row["placement_role"]): float(row["paired_shift"])
        for row in load_csv("mobility_contrast.csv")
    }
    expected_contrast = {
        ("B", 0.1, "intermediate"): 270.0,
        ("C", 0.1, "intermediate"): 260.0,
        ("B", 0.1, "outboard"): -110.0,
        ("C", 0.1, "outboard"): -100.0,
        ("B", 0.3, "intermediate"): 200.0,
        ("C", 0.3, "intermediate"): 200.0,
        ("B", 0.3, "outboard"): -60.0,
        ("C", 0.3, "outboard"): -60.0,
    }
    if contrast != expected_contrast:
        failures.append("mobility-contrast table does not match frozen values")

    transport = load_csv("transport_response.csv")
    maximum_full = max(float(row["max_corrected_relative_error"]) for row in transport)
    maximum_axial = max(
        float(row["max_spectral_axial_only_relative_error"]) for row in transport
    )
    if abs(maximum_full - 0.008545907723271252) > 1e-14:
        failures.append(f"unexpected full-CV maximum error: {maximum_full}")
    if abs(maximum_axial - 0.21291566643466878) > 1e-14:
        failures.append(f"unexpected spectral axial-only maximum error: {maximum_axial}")

    spatial = {(row["grid_level"], row["case_id"]): row for row in load_csv("spatial_refinement.csv")}
    fine = spatial.get(("F", "c34p5"))
    if fine is None or float(fine["paired_effect_c34_minus_untreated"]) != -100.0:
        failures.append("fine-grid paired effect is not -100")
    elif (float(fine["paired_interval_lower"]), float(fine["paired_interval_upper"])) != (-110.0, -90.0):
        failures.append("fine-grid propagated interval is not [-110,-90]")

    checks = load_csv("numerical_checks.csv")
    if len(checks) == 0 or any(row.get("passed", "").lower() != "true" for row in checks):
        failures.append("numerical-check ledger contains a nonpassing record")
    return failures


def check_portable_provenance() -> list[str]:
    failures: list[str] = []
    manifest = load_json(DATA / "provenance.json")
    if manifest.get("manifest_id") != "selective_mobility_scripta_public_provenance_v1":
        failures.append("public provenance manifest ID is wrong")
        return failures
    sources = manifest.get("sources", [])
    if {record.get("id") for record in sources} != {f"S{index:02d}" for index in range(1, 17)}:
        failures.append("public provenance does not contain exactly S01-S16")
    for record in sources:
        public = record.get("public_copy")
        if public:
            path = ROOT / public["path"]
            if not path.is_file():
                failures.append(f"missing public provenance copy: {record['id']}")
            elif sha256(path) != public["sha256"] or path.stat().st_size != public["bytes"]:
                failures.append(f"public provenance identity mismatch: {record['id']}")

    paired_extract = load_json(
        PAPER
        / "provenance"
        / "paired_repeat_protocol"
        / "preflight_extract.json"
    )
    protocol_record = paired_extract["original_artifacts"]["executable_protocol"]
    protocol_path = ROOT / protocol_record["relative_name"]
    if protocol_record.get("included_as_exact_copy") is not False:
        failures.append("paired-repeat public protocol is incorrectly marked as an exact copy")
    if not protocol_path.is_file():
        failures.append("paired-repeat public protocol is missing")
    elif sha256(protocol_path) != protocol_record.get("public_sha256"):
        failures.append("paired-repeat public protocol hash is wrong")
    original_digest = protocol_record.get("original_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", original_digest):
        failures.append("paired-repeat original protocol hash is malformed")
    return failures


def check_figures() -> list[str]:
    failures: list[str] = []
    manifests = [
        FIGURES / "artwork" / "main" / f"figure{index}_manifest.json"
        for index in range(1, 5)
    ] + [FIGURES / "artwork" / "graphical_abstract" / "graphical_abstract_manifest.json"]
    for path in manifests:
        if not path.is_file():
            failures.append(f"missing figure manifest: {path.name}")
            continue
        record = load_json(path)
        for key in ("compact_outputs", "full_width_outputs", "outputs"):
            for output in record.get(key, []):
                output_path = Path(output["path"])
                if output_path.parts and output_path.parts[0] == "science_lab":
                    candidate = ROOT / output_path
                elif path.name == "graphical_abstract_manifest.json":
                    candidate = path.parent / output_path
                else:
                    candidate = PAPER / output_path
                if not candidate.is_file():
                    failures.append(f"missing figure output: {output['path']}")
                elif sha256(candidate) != output["sha256"]:
                    failures.append(f"figure hash mismatch: {output['path']}")
    ga = load_json(
        FIGURES / "artwork" / "graphical_abstract" / "graphical_abstract_manifest.json"
    )
    if ga.get("master_pixel_size") != [3000, 1200]:
        failures.append("graphical abstract master dimensions changed")
    supporting = load_json(
        FIGURES / "artwork" / "supporting" / "supporting_figures_manifest.json"
    )
    for figure in supporting.get("figures", {}).values():
        for key in ("compact_outputs", "full_width_outputs"):
            for output in figure.get(key, []):
                path = ROOT / output["path"]
                if not path.is_file():
                    failures.append(f"missing supporting figure: {output['path']}")
                elif sha256(path) != output["sha256"] or path.stat().st_size != output["bytes"]:
                    failures.append(f"supporting-figure identity mismatch: {output['path']}")
    obsolete = [
        path.relative_to(ROOT)
        for path in FIGURES.iterdir()
        if re.search(r"(?:^generated_v\d+$|_v\d+\.py$)", path.name)
    ]
    if obsolete:
        failures.append(f"obsolete figure revision artifacts remain: {obsolete}")

    for path in FIGURES.rglob("*.pdf"):
        if dict(PdfReader(str(path)).metadata or {}):
            failures.append(f"nontechnical PDF metadata remains: {path.relative_to(ROOT)}")
    for path in FIGURES.rglob("*.png"):
        with Image.open(path) as image:
            unexpected = {
                key: value
                for key, value in image.info.items()
                if key.lower() != "dpi"
            }
        if unexpected:
            failures.append(f"nontechnical PNG metadata remains: {path.relative_to(ROOT)}")
    for path in FIGURES.rglob("*.svg"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in ("<cc:Agent>", "<dc:date>", "<dc:creator>")):
            failures.append(f"creator/date SVG metadata remains: {path.relative_to(ROOT)}")

    expected_python = {
        "__init__.py",
        "figure1.py",
        "figure2.py",
        "figure3.py",
        "figure4.py",
        "figure_common.py",
        "generate_all.py",
        "graphical_abstract.py",
        "supporting_figures.py",
        "test_figures.py",
    }
    actual_python = {path.name for path in FIGURES.glob("*.py")}
    if actual_python != expected_python:
        failures.append(
            "figure Python surface differs from the semantic module set: "
            f"{sorted(actual_python ^ expected_python)}"
        )
    return failures


def main() -> None:
    checks = {
        "clean_boundary": check_clean_boundary(),
        "repository_metadata": check_repository_metadata(),
        "local_markdown_links": check_local_markdown_links(),
        "manifest": check_manifest(),
        "core_numbers": check_core_numbers(),
        "portable_provenance": check_portable_provenance(),
        "figures": check_figures(),
    }
    failures = [f"{name}: {message}" for name, messages in checks.items() for message in messages]
    report = {
        "schema_version": 1,
        "release": "selective-mobility-placement-nanowire-fragmentation-1.0.1",
        "passed": not failures,
        "checks": {name: not messages for name, messages in checks.items()},
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
