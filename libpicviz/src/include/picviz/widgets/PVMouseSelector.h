#ifndef PVMOUSESELECTOR_H_
#define PVMOUSESELECTOR_H_


#include <tulip/InteractorComponent.h>

class QMouseEvent;
class QKeyEvent;

using namespace std;
using namespace tlp;

namespace tlp {
class Graph;
}

/** \addtogroup Mouse_interactor */
/*@{*/
class PVMouseSelector:public InteractorComponent {
protected:
  Qt::MouseButton mButton;
  Qt::KeyboardModifier kModifier;
  Qt::KeyboardModifiers mousePressModifier;
  unsigned int x,y;
  int w,h;
  bool started;
  Graph *graph;
public:
  PVMouseSelector(Qt::MouseButton button = Qt::LeftButton, Qt::KeyboardModifier modifier = Qt::NoModifier);
  ~PVMouseSelector() {}
  bool draw(GlMainWidget *);
  bool eventFilter(QObject *, QEvent *);
  InteractorComponent *clone() {
    return new PVMouseSelector(mButton, kModifier, _mode);
  }

  //all of this stands here to keep binary compatibility. here should be only one constructor on the next minor version bump.
  enum SelectionMode {
    EdgesAndNodes = 0,
    EdgesOnly,
    NodesOnly
  };
  PVMouseSelector(Qt::MouseButton button, Qt::KeyboardModifier modifier, SelectionMode mode);
protected:
  SelectionMode _mode;
};
/*@}*/

#endif /* PVMOUSESELECTOR_H_ */
