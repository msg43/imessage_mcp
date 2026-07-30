"""LaunchAgent installation (SPEC §5.5): renders the 7 thin,
content-free `com.imsgindex.*` plist templates and installs them into
`~/Library/LaunchAgents`. `plists.py` is pure dict-building + rendering
(no filesystem writes); `imsg.cli`'s `install-agents` command is the
only place that actually writes files.
"""

from __future__ import annotations
