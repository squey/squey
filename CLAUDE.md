@AGENTS.md

## Claude Code

- **Commit attribution.** `.claude/settings.json` turns off the attribution Claude Code adds to
  commits and merge requests by default: CONTRIBUTING.md wants messages that describe the change
  and mention no tool.

### C/C++ code intelligence

`.claude/skills/squey-clangd` is a Claude Code plugin that provides the LSP tool for C and C++
files, through the `buildstream/clangd.sh` script that CONTRIBUTING.md describes.

- **Which build tree.** It follows the tree of the session's checkout:
  `builds/x86_64-linux-gnu/Clang/RelWithDebInfo`, else `builds/linux-release`.
- **What it writes.** It rewrites the compile commands (unity builds, precompiled headers) into
  `<build tree>/clangd/`, and keeps its index there.
- **Requirements.** The sandbox must be running and reachable without a password. If ssh needs
  options, put them in `SQUEY_SANDBOX_SSH_OPTS`, for instance in the `env` of
  `.claude/settings.local.json`.
- **Official plugin disabled.** `.claude/settings.json` disables the official `clangd-lsp`
  plugin: it runs the host clangd, which cannot see `/app`.
- **When it loads.** The plugin loads for sessions started at the root of a checkout that
  contains it, a worktree root included. A personal plugin of the same name in
  `~/.claude/skills/` shadows it.
- **Limits.**
  - Definitions in third-party headers point to `/app/...`, a path that exists only inside the
    sandbox.
  - Diagnostics only cover the open files, with the Linux flags, so the build remains the
    reference.
  - The first indexing of a build tree takes about ten minutes.
