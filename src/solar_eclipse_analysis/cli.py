import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from solar_eclipse_analysis.analyze import analyze_raws, print_rankings
from solar_eclipse_analysis.edge import calculate_edge_spread, print_edge_rankings
from solar_eclipse_analysis.extract import extract_raw_archives
from solar_eclipse_analysis.inventory import create_inventory, print_inventory
from solar_eclipse_analysis.paths import ProjectPaths
from solar_eclipse_analysis.raw import RawDecoder
from solar_eclipse_analysis.render import render_all

Command = Literal["extract", "inventory", "analyze", "edge-spread", "render", "all"]


class CliArguments(argparse.Namespace):
    root: Path
    archives: Path | None
    raw: Path | None
    data: Path | None
    crops: Path | None
    renders: Path | None
    final: Path | None
    crop_size: int
    command: Command | None

    def __init__(self) -> None:
        super().__init__()
        self.root = Path.cwd()
        self.archives = None
        self.raw = None
        self.data = None
        self.crops = None
        self.renders = None
        self.final = None
        self.crop_size = 1400
        self.command = None


def project_paths(arguments: CliArguments) -> ProjectPaths:
    return ProjectPaths.create(
        root=arguments.root,
        archives=arguments.archives,
        raw=arguments.raw,
        data=arguments.data,
        crops=arguments.crops,
        renders=arguments.renders,
        final=arguments.final,
    )


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project root (default: current directory)",
    )
    _ = parser.add_argument(
        "--archives", type=Path, help="RAW ZIP directory (default: source/raw-archives)"
    )
    _ = parser.add_argument(
        "--raw", type=Path, help="extracted CR3 directory (default: source/raw)"
    )
    _ = parser.add_argument(
        "--data", type=Path, help="CSV/checkpoint directory (default: analysis/data)"
    )
    _ = parser.add_argument(
        "--crops", type=Path, help="analysis crop directory (default: analysis/crops)"
    )
    _ = parser.add_argument(
        "--renders",
        type=Path,
        help="render output directory (default: analysis/renders)",
    )
    _ = parser.add_argument(
        "--final", type=Path, help="final output directory (default: final)"
    )
    _ = parser.add_argument("--crop-size", type=int, default=1400)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="solar-eclipse-analysis",
        description="Portable Canon CR3 eclipse analysis pipeline",
    )
    add_path_arguments(root)
    subcommands = root.add_subparsers(dest="command", required=True)
    _ = subcommands.add_parser("extract", help="extract CR3 files from source archives")
    _ = subcommands.add_parser("inventory", help="write RAW metadata inventory")
    _ = subcommands.add_parser("analyze", help="measure RAW clipping and edge detail")
    _ = subcommands.add_parser(
        "edge-spread", help="measure 10-90%% edge width from analysis crops"
    )
    _ = subcommands.add_parser(
        "render", help="create analysis renders and final progression"
    )
    _ = subcommands.add_parser(
        "all", help="extract, inventory, analyze, edge-spread, and render"
    )
    return root


def parse_arguments(argv: Sequence[str] | None) -> CliArguments:
    arguments = CliArguments()
    _ = parser().parse_args(argv, namespace=arguments)
    if arguments.command is None:
        raise RuntimeError("No command selected")
    return arguments


def run(arguments: CliArguments) -> None:
    paths = project_paths(arguments)
    if arguments.command in {"extract", "all"}:
        extracted = extract_raw_archives(paths.archives, paths.raw)
        print(f"RAW files ready: {len(extracted)} in {paths.raw}")
        if arguments.command == "extract":
            return
    if arguments.command in {"inventory", "all"}:
        decoder = RawDecoder()
        rows = create_inventory(paths.raw, paths.data / "inventory.csv", decoder)
        print_inventory(rows)
        if arguments.command == "inventory":
            return
    if arguments.command in {"analyze", "all"}:
        decoder = RawDecoder()
        rows = analyze_raws(
            paths.raw,
            paths.data / "inventory.csv",
            paths.data / "metrics.csv",
            paths.data / "metrics.jsonl",
            paths.crops,
            decoder,
            arguments.crop_size,
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
        decoder = RawDecoder()
        created = render_all(
            paths.raw,
            paths.data / "metrics.csv",
            paths.renders,
            paths.final,
            paths.data / "8507-8509_alignment.txt",
            decoder,
        )
        print("Created:")
        for path in created:
            print(path)


def main(argv: Sequence[str] | None = None) -> None:
    try:
        run(parse_arguments(argv))
    except (FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
