kind: manual
description: |

    gcovr generates code coverage reports out of the gcov data produced by an
    instrumented build, both as a browsable HTML report and as a Cobertura XML
    report for the GitLab coverage widget.

sources:
# Wheels rather than source archives: gcovr builds with hatchling, which the
# runtime does not ship and which would have to be packaged just to unpack a
# pure Python distribution
- kind: remote
  url: pythonhosted:packages/92/be/f722c843e7875c7cf92cf0e0c1604cddda55a70278c768c6327a78fdba79/gcovr-8.6-py3-none-any.whl
  ref: dbf9d87c38042752ad6f530aa8210427e22b526611bb7b7bfed0e81977d1f1ef
# The only gcovr runtime dependency the runtime does not already provide
- kind: remote
  url: pythonhosted:packages/ec/a1/6b71004ab0fea510230be9ce05a4059029ac847c009fcc80b1b73d6fa5ab/colorlog-6.11.0-py3-none-any.whl
  ref: f1e27d75aa2cb138f3f640c0e305b65b680ccbef6ecc034eba7e03494ffcd2a1

config:
  install-commands:
  - python3 -m pip install --no-index --no-deps --root="%{install-root}" --prefix=/usr *.whl

depends:
- base.bst
- freedesktop-sdk.bst:components/python3-jinja2.bst
- freedesktop-sdk.bst:components/python3-lxml.bst
- freedesktop-sdk.bst:components/python3-pygments.bst

build-depends:
- freedesktop-sdk.bst:components/python3-pip.bst
