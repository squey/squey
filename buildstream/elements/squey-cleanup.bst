kind: script

description: |

    Remove unwanted files.

(@): buildstream/elements/base/target.yml

config:
  root-read-only: False
  commands:
    - bash /buildstream-scripts/prune_prefix.sh "%{prefix}" "%{target-platform}" "%{host}" "%{python-version}"
    - cp -r %{prefix} %{install-root}

build-depends:
- base.bst
- squey.bst
- filename: squey-cleanup-script.bst
  config:
    location: /buildstream-scripts
