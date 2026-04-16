"""CLI entrypoint and command dispatch."""

from __future__ import annotations

from . import commands
from .parser import build_parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "webcam":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_webcam(args)
            return
        commands._run_webcam(args)
        return
    if args.command == "video":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_video(args)
            return
        commands._run_video(args)
        return
    if args.command == "replay":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_replay(args)
            return
        commands._run_replay(args)
        return
    if args.command == "dump-video":
        if args.hand == "both":
            commands._run_bihand_dump_video(args)
            return
        commands._run_dump_video(args)
        return
    if args.command == "pico":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_pico(args)
            return
        commands._run_pico(args)
        return
    if args.command == "hc-mocap":
        if args.hand == "both":
            if args.backend != "viewer":
                raise ValueError("Controller backends are currently only supported for single-hand commands")
            commands._run_bihand_hc_mocap_udp(args)
            return
        commands._run_hc_mocap_udp(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")
