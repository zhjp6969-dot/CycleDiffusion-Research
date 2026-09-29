#!/usr/bin/env python3
"""Static audit for gradient-blocking decorators in the archived cycle path."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def decorated_methods(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, list[str]] = {}
    class_name = ""
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_name = node.name
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [ast.unparse(item) for item in child.decorator_list]
            if decorators:
                result[f"{class_name}.{child.name}"] = decorators
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to a lawfully obtained historical CycleDiffusion source snapshot.",
    )
    args = parser.parse_args()
    vc = decorated_methods(args.root / "model" / "vc.py")
    diffusion = decorated_methods(args.root / "model" / "diffusion.py")
    findings = {
        "model.vc.DiffVC.forward": vc.get("DiffVC.forward", []),
        "model.diffusion.Diffusion.forward": diffusion.get("Diffusion.forward", []),
        "model.diffusion.Diffusion.reverse_diffusion": diffusion.get(
            "Diffusion.reverse_diffusion", []
        ),
    }
    blocked = all("torch.no_grad()" in value for value in findings.values())
    print(json.dumps({"cycle_gradient_blocked": blocked, "findings": findings}, indent=2))
    return 1 if not blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
