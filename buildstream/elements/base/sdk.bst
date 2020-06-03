kind: stack
depends:
- filename: bootstrap-import.bst
  junction: freedesktop-sdk.bst
- filename: public-stacks/buildsystem-autotools.bst
  junction: freedesktop-sdk.bst
- filename: public-stacks/buildsystem-cmake.bst
  junction: freedesktop-sdk.bst
- filename: components/ninja.bst
  junction: freedesktop-sdk.bst
#- filename: components/ccache.bst
#  junction: freedesktop-sdk.bst
 
