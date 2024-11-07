kind: stack
(?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      depends:
        - base/osxcross.bst
  - (target_triple == "x86_64-linux-gnu"):
      depends:
        - base.bst