kind: manual

(@): buildstream/elements/base/target.bst

variables:
    target-env-vars-path: "%{install-root}/etc/target_env_vars.sh"

config:
  configure-commands:
  - |
    mkdir -p $(dirname "%{target-env-vars-path}")

  install-commands:
  - |
    echo "export PREFIX=%{prefix}" >> %{target-env-vars-path}
    echo "export TARGET_TRIPLE=%{target_triple}" >> %{target-env-vars-path}
    echo "export HOST=%{host}" >> %{target-env-vars-path}
    echo "export TOOLCHAIN_DIR=%{toolchain-dir}" >> %{target-env-vars-path}
    echo "export TARGET_PLATFORM=%{target-platform}" >> %{target-env-vars-path}
    echo "export DEPLOYMENT_TARGET_VERSION=%{deployment-target-version}" >> %{target-env-vars-path}

build-depends:
- sdk.bst
