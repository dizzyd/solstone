# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified observer entry point with platform detection.

Detects the current platform and delegates to the appropriate
platform-specific observer implementation.
"""

import sys


def main() -> None:
    """Platform-aware observer entry point.

    Detects the current platform and calls the appropriate observer:
    - macOS (darwin): observe.macos.observer
    - Linux: observe.linux.observer

    All command-line arguments are passed through to the platform-specific
    implementation via its main() function.
    """
    platform = sys.platform

    if platform == "darwin":
        from observe.macos.observer import main as platform_main
    elif platform == "linux":
        from observe.linux.observer import main as platform_main
    else:
        print(
            f"Error: Observer not available for platform '{platform}'", file=sys.stderr
        )
        print("Supported platforms: macOS (darwin), Linux", file=sys.stderr)
        sys.exit(1)

    platform_main()


if __name__ == "__main__":
    main()
