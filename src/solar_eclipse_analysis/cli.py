import argparse
from collections.abc import Sequence
from pathlib import Path

from solar_eclipse_analysis.analyze import analyze_raws, print_rankings
from solar_eclipse_analysis.edge import calculate_edge_spread, print_edge_rankings
from solar_eclipse_analysis.extract import extract_raw_archives
from solar_eclipse_analysis.inventory import create_inventory, print_inventory
from solar_eclipse_analysis.libraw import LibRaw
from solar_eclipse_analysis.paths import ProjectPaths
from solar_eclipse_analysis.render import render_all


def project_paths(arguments: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths.create(
        root=arguments.root,
        archives=arguments.archives,
        raw=arguments.raw,
        data=arguments.data,
        crops=arguments.crops,
        renders=arguments.renders,
    )


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project root (default: current directory)",
    )
    parser.add_argument(
        "--archives", type=Path, help="RAW ZIP directory (default: source/raw-archives)"
    )
    parser.add_argument(
        "--raw", type=Path, help="extracted CR3 directory (default: source/raw)"
    )
    parser.add_argument(
        "--data", type=Path, help="CSV/checkpoint directory (default: analysis/data)"
    )
    parser.add_argument(
        "--crops", type=Path, help="analysis crop directory (default: analysis/crops)"
    )
    parser.add_argument(
        "--renders",
        type=Path,
        help="render output directory (default: analysis/renders)",
    )
    parser.add_argument(
        "--libraw", type=Path, help="LibRaw shared library; auto-detected by default"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="solar-eclipse-analysis",
        description="Portable Canon CR3 eclipse analysis pipeline",
    )
    add_path_arguments(root)
    subcommands = root.add_subparsers(dest="command", required=True)
    subcommands.add_parser("extract", help="extract CR3 files from source archives")
    subcommands.add_parser("inventory", help="write RAW metadata inventory")
    analyze_parser = subcommands.add_parser(
        "analyze", help="measure RAW clipping and edge detail"
    )
    analyze_parser.add_argument("--crop-size", type=int, default=1400)
    subcommands.add_parser(
        "edge-spread", help="measure 10-90%% edge width from analysis crops"
    )
    subcommands.add_parser("render", help="create selected eclipse renders and montage")
    subcommands.add_parser(
        "all", help="extract, inventory, analyze, edge-spread, and render"
    )
    return root


def run(arguments: argparse.Namespace) -> None:
    paths = project_paths(arguments)
    paths.ensure_outputs()
    if arguments.command in {"extract", "all"}:
        extracted = extract_raw_archives(paths.archives, paths.raw)
        print(f"RAW files ready: {len(extracted)} in {paths.raw}")
        if arguments.command == "extract":
            return
    decoder = LibRaw(arguments.libraw)
    if arguments.command in {"inventory", "all"}:
        rows = create_inventory(paths.raw, paths.data / "inventory.csv", decoder)
        print_inventory(rows)
        if arguments.command == "inventory":
            return
    if arguments.command in {"analyze", "all"}:
        rows = analyze_raws(
            paths.raw,
            paths.data / "inventory.csv",
            paths.data / "metrics.csv",
            paths.data / "metrics.jsonl",
            paths.crops,
            decoder,
            getattr(arguments, "crop_size", 1400),
        )
        print_rankings(rows)
        if arguments.command == "analyze":
            return
    if arguments.command in {"edge-spread", "all"}:
        rows = calculate_edge_spread(
            paths.data / "metrics.csv", paths.crops, paths.data / "edge_spread.csv"
        )
        print_edge_rankings(rows)
        if arguments.command == "edge-spread":
            return
    if arguments.command in {"render", "all"}:
        created = render_all(
            paths.raw,
            paths.data / "metrics.csv",
            paths.crops,
            paths.renders,
            paths.data / "8507-8509_alignment.txt",
            decoder,
        )
        print("Created:")
        for path in created:
            print(path)


def main(argv: Sequence[str] | None = None) -> None:
    run(parser().parse_args(argv))
