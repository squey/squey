
# Contributing guidelines

## Coding standards

The project should follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) by Bjarne Stroustrup & Herb Sutter. \
Code formatting style should follow the [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) file [.clang-format](https://gitlab.com/inendi/inspector/-/blob/main/.clang-format). \
I case of doubt always follow what appears to be the project established conventions.

## Branching model

Active development should be done on short-lived branches originating from main and merged back to main. \
Release are symbolized by a tag in main (eg tag = "v4.9.10"). \
Major and minor release must be associated with a branch for backports & bugfixes (eg branch = "branch-4.9").

### Merge requests

Since branch "main" is protected, any code located on a short-lived branch being merged is therefore subject to a Merge Request. \
Merge Requests aim is to ensure that :

1. The CI/CD pipeline is passing and no regression was introduced.
2. The code has been reviewed by someone else with Developer, Maintainer or Owner role.
3. The commits are clean (squash some of them if using [git rebase -i](https://gitlab.com/inendi/inspector/-/wikis/rebase) if necessary).

If after opening a Merge Request you spot something wrong, temporarily change its status to "Draft" to prevent it to be merged. \
Do not use the draft status otherwise because it prevents to use Gitlab "Merge when pipeline succeeds" feature.

## Code reviews

Code reviews are a solid way to catch problems early in the development process, but are also a good way to share the knowledge through the team on what has been changed. \
At the lowest level of review, we should at least check that :

1. Nothing look suspicious
2. There is **no code duplication** (this should be a **no go** to merge !)
3. The code has properly been tested (in an automated and/or manual way)

[Google's code review best practices](https://google.github.io/eng-practices/review/) can be a source of inspiration for reviews.

## Version management

Version management is following [Semantic Versioning (SemVer)](https://semver.org/) version numbering format.

1. MAJOR version when breaking changes are introduced
2. MINOR version when functionalities are added in a backwards compatible manner
3. PATCH version when fixing bugs in backwards compatible manner

### Releasing bug fixes

Bug fixes should be done on the main branch and then cherry-picked to the released version branch. \
For example, if the current minor version is version 4.9 and a bug fix is targeting future version 4.9.10, the fix should be done on a short-lived branch originating from main, be merged back to main through a Merge Request and then the associated commits must be cherry-picked to a branch originating from "branch-4.9" and then merged to "branch-4.9". \
When releasing a fix, a new tag "v4.9.10" should be added to the "branch-4.9" branch.

```
git checkout main && git pull
git checkout -b bugfix
git commit -m "This is the bugfix"
git push
# Then create a MR and merge branch "bugfix" to "main"

git checkout branch-4.9 && git pull
git branch -b bugfix_backport
git cherry-pick 1fd20f2c32c2bdf7d5c6df6ae2bbbf55d5e24235
git push
# Then create a MR and merge branch "bugfix_backport" to "branch-4.9"
```
