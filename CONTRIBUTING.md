# Contributing to Squey

These rules apply to every contribution, whoever writes it and whatever tools they use.

## Development environment

Development currently requires a Linux machine: the BuildStream development sandbox only runs
on Linux, and the Windows and macOS versions are cross-compiled from sandboxes running there.

The toolchain and every dependency (Qt, Arrow, sigc++, DuckDB…) live under `/app`, which two
environments provide. The host has none of them, and the CMake configuration stops there.
[buildstream/README.md](buildstream/README.md) explains how to set each one up.

### The BuildStream development sandbox

The reference environment, and the only one that cross-compiles and packages.

- `cd buildstream && ./dev_shell.sh` starts the Linux sandbox and keeps an interactive shell open.
  `--target_triple=` selects another target: `x86_64-w64-mingw32`, `aarch64-apple-darwin` or
  `x86_64-apple-darwin`. Sandboxes for different targets can run side by side.
- Each sandbox runs an ssh server. `buildstream/sshd/ssh_config.squey` names them `SqueyLinux`
  (port 6666), `SqueyWin` (6667) and `SqueyMac` (6668); authentication is by key only.
- From the host, run commands with `ssh SqueyLinux '<command>'`. The session gets the
  environment of the sandbox shell, and the sandbox sees the host filesystem at the same paths:
  `cd` to the same absolute path.

### The devcontainer

`.devcontainer/` describes an image of the same `/app`, for the editors that support development
containers and for the `devcontainer` CLI. It is quicker to enter than the sandbox, but only
builds and tests the native Linux version.

- On creation, it configures `builds/x86_64-linux-gnu/Clang/RelWithDebInfo`.
- It binds the Wayland socket of the host, to show the GUI: the host needs a Wayland session.
- `.devcontainer/devcontainer.json` pins the image by a digest of the dependency graph. When a
  branch changes a dependency, its merge request pipeline publishes the new image, and the
  merge commits the new pin to `main`. Until then, the branch keeps the image of `main`.

### Editors and coding assistants

A clangd started on the host cannot see the headers under `/app`, and reports errors that do not
exist. Use `buildstream/clangd.sh` as the clangd binary of your editor or coding assistant. It
runs clangd on the build tree of the checkout it is started from: through the ssh server of the
sandbox from the host, directly inside the sandbox or the devcontainer. The devcontainer also
points `src/compile_commands.json` at the compile commands it gives clangd, for the editors that
start the clangd they find.

## Building

- On first start, `dev_shell.sh` configures `builds/<target triple>/<Clang|GCC>/<Debug|RelWithDebInfo>`.
  Run `ninja` inside one of them; it runs `cmake` again by itself when a `CMakeLists.txt` changes.
- Validate changes in `builds/x86_64-linux-gnu/Clang/RelWithDebInfo`. It builds like the CI
  (Clang, `-O3 -ffast-math`), which a GCC or Debug build does not reproduce.
- The CI treats warnings as errors: a build log must contain no `warning:`.
- `src/CMakePresets.json` describes the same configurations. `cmake --preset linux-release`,
  run from `<checkout>/src`, configures `<checkout>/builds/linux-release`.

## Testing

- `ctest` first builds what the tests need, so it never runs a stale test executable. While
  iterating, run only the relevant tests (`ctest -R <pattern>`). Keep the full suite
  (`ninja squey_run_testsuite`) for before a commit.
- `ctest` runs each test through `buildstream/files/flatpak/run_cmd.sh`, which sets up OpenCL and
  the library path as it does for the packaged application. Run a test executable by hand
  through it too.
- `src/libpvcop` is a repository of its own, tested there. Squey only registers its tests to
  measure the code coverage.

## Other platforms and the CI

- The CI builds Linux, Windows (mingw) and macOS (osxcross). Some code only breaks in the
  Windows and macOS sandboxes, so compile it there before pushing:
  - code under `#ifdef`;
  - code depending on type sizes (`long` is 32 bits on Windows);
  - code using recent standard library features (the macOS SDK's libc++, deployment target
    10.15).
- Pushing a branch does not start a pipeline: pipelines run for merge requests, schedules,
  manual runs and release tags.

## Working on several branches at once

Several changes can progress side by side, each in its own git worktree: for instance, a recent
feature branch next to a bugfix on an older base.

- `buildstream/new_worktree.sh <branch> [<base>]` creates the worktree next to the current
  checkout, from `<base>` (`origin/main` by default). It clones the submodules from the local
  copies rather than from the network, and configures `builds/linux-release` in the sandbox.
- Build and test a change only in its own checkout (`git rev-parse --show-toplevel`), never in
  the main checkout or in another worktree.
- All worktrees share the running Linux sandbox, whose `/app` comes from the branch that started
  it.

## Coding standards

- The project follows the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
  by Bjarne Stroustrup and Herb Sutter.
- Code formatting follows the [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) file
  [src/.clang-format](src/.clang-format).
- In case of doubt, follow the conventions the project has already established.
- Code, comments and identifiers are written in English.
- New files carry the license header of the existing ones, with `© Squey, <current year>`.

### Comments

Follow the naming-and-layout rules of the C++ Core Guidelines:

- **NL.1**: don't say in a comment what the code already states clearly. Compilers don't read
  comments, comments are less precise than code, and they are not maintained as consistently.
- **NL.2**: state intent. Code says what is done, not what was supposed to be done.
- **NL.3**: keep comments crisp. Verbosity slows the reader down.

If a comment and the code disagree, assume both are wrong.

Write a comment only for what the code cannot express on its own:

- **Contract**: preconditions, postconditions, invariants of a public entity.
- **Ownership and lifetime**: who owns a raw pointer, how long a reference or span stays valid,
  what must outlive what.
- **Thread-safety**: which mutex guards which member, which functions may be called
  concurrently and which may not, which are re-entrant.
- **Non-obvious rationale**: why a slower-looking path is actually faster, why a standard
  facility is avoided, alignment, SIMD or ABI constraints, workarounds for a compiler or library
  bug (name the bug, link the issue).
- **Units and ranges the type does not carry**: milliseconds or seconds, half-open or closed
  intervals, 0-based or 1-based.

Do not comment:

- **Restatements of the code.** Bad: `auto x = m * v1 + vv; // multiply m by v1 and add vv`.
- **Step narration**: `// Step 1: validate input`, `// Loop over the rows`. If a function needs
  section headers, extract functions instead.
- **Change history**: "replaces the old implementation", "added to fix the crash on startup",
  "previously used std::map". It belongs in the commit message, and git already records it.
- **Session or review artefacts**: "as requested", "per the review", "refactored for clarity".
- **Commented-out code.** Delete it.
- **Decorative banners and separator lines.**
- **Bare `TODO` or `FIXME`.** A TODO names an owner and an issue:
  `// TODO(#412): drop once the Arrow 22 upgrade lands`.
- **Code you did not change.** Do not annotate the surrounding code as a side effect of your
  edit, and do not reformat existing comments.

Doxygen:

- Document the public API in headers only. Do not repeat the block in the `.cpp`.
- Document the contract, not the signature. `\param count The count` adds nothing: omit it.
- Document non-obvious behaviour: what the function does on failure, whether it allocates,
  whether it blocks, what it throws.

Form:

- English, present tense, full sentences for block comments.
- Comment lines go above the code they describe, not after it. Short trailing comments are
  acceptable for a single value or case label.
- Use the comment style already dominant in the file.
- When you change code, update or delete the comment attached to it. A stale comment is a bug,
  and leaving one is worse than having written none.

## Commit messages

- Subject line in the imperative mood, 50 characters or fewer, no trailing period:
  `Fix overflow in row index computation`, not `Fixed...` or `Fixes...`.
- Blank line, then a body wrapped at 72 columns.
- The body explains why: what problem existed, why this approach, what was considered and
  rejected. The diff already shows what changed.
- One logical change per commit. Do not mix a refactor with a behaviour change, and do not fold
  unrelated formatting into a functional commit.
- Reference issues with trailers: `Closes: #123`, `Refs: #98`.
- The message describes the change, and nothing else. Whatever tools helped write it, do not
  mention them: no `Co-authored-by:` for a tool, no "Generated with" line, no emoji.

Do not:

- Generate the message from the diff. `Update parser.cpp, add helper function` is a
  restatement, not a message.
- Describe the working session: "as asked", "after feedback", "second attempt", "final version".
- Claim that tests or builds pass unless they were actually run. State what was run, on what
  configuration.
- Amend or reword commits that are already pushed to a shared branch.

For example:

```
Fix overflow in row index computation

The 32-bit accumulator wrapped for tables above 2^31 rows, producing a
negative offset and an out-of-range read. Widen it to std::size_t and add
a regression test on the boundary.

Closes: #412
```

## Branching model

Active development happens on short-lived branches that start from `main` and are merged back
into it. A release is marked by a tag in `main` (for instance `release-5.1.3`). Each major and
minor release must have a branch for backports and bug fixes (for instance `branch-4.9`).

### Merge requests

Since `main` is protected, merging a short-lived branch goes through a merge request. Merge
requests ensure that:

1. The CI/CD pipeline passes and no regression was introduced.
2. Someone else with a Developer, Maintainer or Owner role reviewed the code.
3. The commits are clean: squash some of them with
   [git rebase -i](https://gitlab.com/squey/squey/-/wikis/rebase) if necessary.

If you spot something wrong after opening a merge request, switch it to "Draft" for a while to
prevent it from being merged. Do not use the draft status otherwise: it prevents GitLab's
"Merge when pipeline succeeds" feature.

### External contributions

Without a [Developer](https://docs.gitlab.com/ee/user/permissions.html) role on this project,
follow the [forking workflow](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html)
and [request your branch to be merged upstream](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#merging-upstream).
A Developer then reviews your changes before running a
[CI pipeline in the project](https://docs.gitlab.com/ee/ci/pipelines/merge_request_pipelines.html#run-pipelines-in-the-parent-project).

## Code reviews

Code reviews catch problems early in the development process, and share the knowledge of what
changed through the team. At the very least, check that:

1. Nothing looks suspicious.
2. There is **no code duplication**: it is a **no go** for merging.
3. The code has been tested, automatically or manually.

[Google's code review best practices](https://google.github.io/eng-practices/review/) are a good
source of inspiration.

## Version management

Versions follow [Semantic Versioning](https://semver.org/):

1. MAJOR version for breaking changes.
2. MINOR version for functionality added in a backwards compatible manner.
3. PATCH version for bug fixes in a backwards compatible manner.

Pushing a `release-X.Y.Z` tag starts the pipeline that produces the release.

### Releasing bug fixes

Fix bugs on `main` first, then cherry-pick the fix onto the release branch. For instance, if the
current minor version is 4.9 and a fix targets the future 4.9.10:

1. Make the fix on a short-lived branch from `main`, and merge it into `main` through a merge
   request.
2. Cherry-pick its commits onto a branch from `branch-4.9`, and merge that branch into
   `branch-4.9`.
3. Tag the release on `branch-4.9` with `release-4.9.10`.

```
git checkout main && git pull
git checkout -b bugfix
git commit -m "Fix the bug"
git push
# Then create a merge request to merge "bugfix" into "main"

git checkout branch-4.9 && git pull
git checkout -b bugfix_backport
git cherry-pick 1fd20f2c32c2bdf7d5c6df6ae2bbbf55d5e24235
git push
# Then create a merge request to merge "bugfix_backport" into "branch-4.9"
```

## Before committing

- [ ] No comment restates the code it sits above.
- [ ] Every comment left behind is still true after the change.
- [ ] No commented-out code, no bare TODO, no step narration.
- [ ] Public API changes are reflected in the header documentation.
- [ ] The subject is imperative and 50 characters or fewer; the body explains why.
- [ ] The commit holds one logical change.
- [ ] The message describes the change only, and mentions no tool.
- [ ] The build and the tests were actually run, and the message says which.
