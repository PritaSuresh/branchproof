"""Command-line tools for test strengthening and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import AzureProvider
from .config import Settings
from .sandbox import execution_image, resolve_image


def repository_map(values: list[str]) -> dict[str, Path]:
    repositories = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Repositories must be NAME=/path/to/clone.")
        if name in repositories:
            raise ValueError(f"Duplicate repository: {name}")
        repositories[name] = Path(path).resolve()
    return repositories


def _provider(settings: Settings):
    if not settings.has_azure():
        raise SystemExit(
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT."
        )
    return AzureProvider(
        settings.azure_endpoint or "",
        settings.azure_api_key or "",
        settings.azure_deployment or "",
        settings.azure_api_version,
        model=settings.azure_model, reasoning_effort=settings.azure_reasoning_effort,
        max_completion_tokens=settings.azure_max_completion_tokens,
    )


SKIP_DIRS = {".git", ".venv", "__pycache__", "tests", "test", "build", "dist", "docs", "examples"}
SKIP_NAMES = {"setup.py", "conf.py", "__init__.py", "__main__.py", "_version.py", "version.py"}


def cmd_strengthen(args, settings: Settings) -> int:
    from .strengthen import strengthen

    directory, result = strengthen(
        Path(args.repo), Path(args.env), args.module,
        None if args.screen_only else _provider(settings), Path(args.output),
        limit=args.limit, candidate_limit=args.candidate_limit,
        max_attempts=args.max_attempts, max_calls=args.max_calls,
        stability_runs=settings.stability_runs, screen_only=args.screen_only,
        screen_timeout=args.screen_timeout, suite_timeout=settings.suite_timeout_s,
        test_timeout=settings.test_timeout_s,
        provider_metadata={"kind": "none" if args.screen_only else "azure",
                           "model": settings.azure_model,
                           "deployment": settings.azure_deployment,
                           "reasoning_effort": settings.azure_reasoning_effort,
                           "max_completion_tokens": settings.azure_max_completion_tokens},
    )
    if args.screen_only:
        print(f"{result['eligible_targets']} surviving targets; no model calls. Saved: {directory}")
        return 0
    print(f"{result['accepted']}/{result['eligible_targets']} targets accepted; "
          f"{result['model_calls']} model calls. Saved independent patches: {directory}")
    return 0


def cmd_test_prepare(args, settings: Settings) -> int:
    from .test_study import combine, prepare

    repositories = repository_map(args.repo_map)
    if args.include_manifest:
        if args.module:
            raise SystemExit("Merge completed preparations separately from new module screening")
        directory, manifest = combine([Path(p) for p in args.include_manifest],
                                      repositories, Path(args.output))
        print(f"{len(manifest['cases'])} combined targets; saved {directory}")
        return 0

    modules = []
    for value in args.module:
        name, separator, module = value.partition(":")
        if not separator or not module:
            raise SystemExit("Use --module NAME:package/module.py")
        modules.append((name, module))
    directory, manifest = prepare(
        repositories, modules, Path(sys.prefix), Path(args.output),
        candidate_limit=args.candidate_limit, per_module=args.per_module,
        additional_limit=args.additional_limit, seed=args.seed,
        development=not args.evaluation_set, screen_timeout=args.screen_timeout)
    eligible = sum(e.get("eligible", False) for e in manifest["modules"])
    print(f"{len(manifest['cases'])} frozen targets; {eligible}/{len(manifest['modules'])} "
          "offered modules eligible")
    print(f"Saved API-free preparation: {directory}")
    return 0


def cmd_test_study(args, settings: Settings) -> int:
    from .agent import model_configuration
    from .test_study import run, validate

    repositories = repository_map(args.repo_map)
    manifest = json.loads(Path(args.manifest).read_text())
    validate(manifest, repositories)
    profile = model_configuration(settings.azure_deployment or "", settings.azure_model,
                                  settings.azure_reasoning_effort,
                                  settings.azure_max_completion_tokens)
    plan = {"manifest_sha256": manifest["sha256"], "purpose": manifest["purpose"],
            "primary_jobs": len(manifest["cases"]) * 2,
            "repeat_jobs": len(manifest["repeat_cases"]) * args.extra_repeats * 2,
            "max_total_calls": args.max_total_calls, "max_total_tokens": args.max_total_tokens,
            "model_configuration": profile, "paid_calls": args.live}
    print(json.dumps(plan, indent=2))
    if not args.live:
        return 0
    directory, result = run(
        manifest, repositories, Path(sys.prefix), _provider(settings), Path(args.output),
        max_total_calls=args.max_total_calls, max_total_tokens=args.max_total_tokens,
        completion_cap=profile["max_completion_tokens"], provider_metadata=profile,
        extra_repeats=args.extra_repeats)
    print("Preliminary acceptance only; run test-score for independently graded results.")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved generation study (run test-score next): {directory}")
    return 0


def cmd_test_score(args, settings: Settings) -> int:
    from .test_study import score

    result = score(json.loads(Path(args.manifest).read_text()), repository_map(args.repo_map),
                   Path(sys.prefix), Path(args.study).resolve())
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved API-free independent grades: {result['grading_directory']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="branchproof",
                                    description="Mutation-guided, verified Python test patches")
    sub = parser.add_subparsers(dest="command", required=True)

    strengthen = sub.add_parser("strengthen", help="add verified tests; export patches only")
    strengthen.add_argument("--repo", required=True)
    strengthen.add_argument("--module", required=True)
    strengthen.add_argument("--env", default=str(Path(sys.executable).parent.parent),
                            help="host virtualenv used only with --unsafe-local")
    strengthen.add_argument("--output", default="runs/strengthen")
    strengthen.add_argument("--limit", type=int, default=3)
    strengthen.add_argument("--candidate-limit", type=int, default=20)
    strengthen.add_argument("--max-attempts", type=int, default=3)
    strengthen.add_argument("--max-calls", type=int, default=30)
    strengthen.add_argument("--screen-timeout", type=int, default=30,
                            help="seconds per original-suite mutation screen")
    strengthen.add_argument("--screen-only", action="store_true",
                            help="API-free baseline/mutation screening; no credentials needed")

    test_prepare = sub.add_parser("test-prepare", help="freeze model-free test-generation targets")
    test_prepare.add_argument("--repo-map", action="append", required=True, metavar="NAME=PATH")
    test_prepare.add_argument("--module", action="append", default=[], metavar="NAME:MODULE")
    test_prepare.add_argument("--include-manifest", action="append",
                              help="union completed preparations without rescreening or filtering")
    test_prepare.add_argument("--output", default="runs/test-preparation")
    test_prepare.add_argument("--candidate-limit", type=int, default=20)
    test_prepare.add_argument("--screen-timeout", type=int, default=30)
    test_prepare.add_argument("--per-module", type=int, default=2)
    test_prepare.add_argument("--additional-limit", type=int, default=2)
    test_prepare.add_argument("--seed", type=int, default=17)
    test_prepare.add_argument("--evaluation-set", action="store_true",
                              help="declare untouched evaluation material, not development")

    test_study = sub.add_parser("test-study", help="paired generation; API-free dry run by default")
    test_study.add_argument("--repo-map", action="append", required=True, metavar="NAME=PATH")
    test_study.add_argument("--manifest", required=True)
    test_study.add_argument("--output", default="runs/test-studies")
    test_study.add_argument("--max-total-calls", type=int, default=60)
    test_study.add_argument("--max-total-tokens", type=int, default=200000)
    test_study.add_argument("--extra-repeats", type=int, choices=[0, 2], default=0)
    study_mode = test_study.add_mutually_exclusive_group()
    study_mode.add_argument("--live", action="store_true", help="permit paid model calls")
    study_mode.add_argument("--dry-run", action="store_true", help="default; no model client")

    test_score = sub.add_parser("test-score", help="replay patches and grade withheld mutations")
    test_score.add_argument("--repo-map", action="append", required=True, metavar="NAME=PATH")
    test_score.add_argument("--manifest", required=True)
    test_score.add_argument("--study", required=True)

    for command in sub.choices.values():
        options = command.add_mutually_exclusive_group()
        options.add_argument("--sandbox-image", help="trusted Docker image, pinned at run start")
        options.add_argument("--unsafe-local", action="store_true",
                             help="run trusted fixture code with host permissions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    handlers = {
        "strengthen": cmd_strengthen,
        "test-prepare": cmd_test_prepare,
        "test-study": cmd_test_study,
        "test-score": cmd_test_score,
    }
    image = None
    if args.unsafe_local:
        print("WARNING: target code will execute with your host permissions.", file=sys.stderr)
    else:
        try:
            image = resolve_image(args.sandbox_image or
                                  os.getenv("HARDENER_SANDBOX_IMAGE", "hardener-sandbox:tests-v1"))
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from None
    with execution_image(image):
        return handlers[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
