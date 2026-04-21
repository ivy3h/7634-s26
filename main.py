"""Entry point.

Modes:
  build    Run Phase I + story_to_plan + world_builder once; save everything.
  play     Load saved plan+world and launch the interactive game loop.
  replay   Read a scripted list of commands from a file and run them
           against the saved plan+world (for transcripts / regression tests).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from game_engine import EngineConfig, GameEngine
from phase1_story_generator import generate_full_story, load_checkpoint
from story_to_plan import build_plan, load_plan
from world_builder import build_world, load_world, save_world


def cmd_build(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_story or not (data_dir / "plot_points.json").exists():
        print("==> Phase I: generating story")
        generate_full_story(user_prompt=args.prompt, out_dir=data_dir, min_points=args.min_points)
    else:
        print("==> Reusing existing Phase I artifacts")

    case_file = load_checkpoint(data_dir / "case_file.json")
    plot_points = load_checkpoint(data_dir / "plot_points.json")

    print("==> Building plan")
    plan = build_plan(case_file, plot_points, out_path=data_dir / "plan.json")

    print("==> Building world")
    world = build_world(plan)
    save_world(world, data_dir / "world.json")
    # Persist plan again so detective.location is set correctly.
    (data_dir / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    print(f"All artifacts saved to {data_dir}/")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    plan = load_plan(data_dir / "plan.json")
    world = load_world(data_dir / "world.json")
    engine = GameEngine(plan, world, EngineConfig(log_dir=Path(args.log_dir)))
    status = engine.run()
    print(f"[game ended: {status}]")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    plan = load_plan(data_dir / "plan.json")
    world = load_world(data_dir / "world.json")
    cmds = Path(args.script).read_text(encoding="utf-8").splitlines()
    cmds_iter = iter(cmds)

    def scripted_input(prompt: str) -> str:
        try:
            line = next(cmds_iter)
        except StopIteration:
            raise EOFError
        print(prompt + line)
        return line

    engine = GameEngine(plan, world, EngineConfig(log_dir=Path(args.log_dir)))
    status = engine.run(get_input=scripted_input)
    print(f"[replay ended: {status}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="cmd", required=True)

    sp_build = subs.add_parser("build", help="Generate story + plan + world")
    sp_build.add_argument("--prompt", default="A poisoning murder at a prestigious 1920s London art gallery opening")
    sp_build.add_argument("--data-dir", default="data")
    sp_build.add_argument("--min-points", type=int, default=20)
    sp_build.add_argument("--skip-story", action="store_true",
                          help="Skip Phase I if data/plot_points.json already exists.")
    sp_build.set_defaults(func=cmd_build)

    sp_play = subs.add_parser("play", help="Launch interactive game")
    sp_play.add_argument("--data-dir", default="data")
    sp_play.add_argument("--log-dir", default="logs")
    sp_play.set_defaults(func=cmd_play)

    sp_replay = subs.add_parser("replay", help="Run a scripted transcript")
    sp_replay.add_argument("script")
    sp_replay.add_argument("--data-dir", default="data")
    sp_replay.add_argument("--log-dir", default="logs")
    sp_replay.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
