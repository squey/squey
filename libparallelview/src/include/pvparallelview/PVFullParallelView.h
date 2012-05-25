#ifndef __PVFULLPARALLELVIEW_H__
#define __PVFULLPARALLELVIEW_H__

#include <QGraphicsView>
#include <QScrollBar>
#include <QFuture>

namespace PVParallelView {

class PVParallelScene;
class PVRenderingJob;

class PVFullParallelView : public QGraphicsView
{
	Q_OBJECT
public:
	virtual void translate_viewport(int translation)
	{
		QScrollBar *hBar = horizontalScrollBar();
		hBar->setValue(hBar->value() + (isRightToLeft() ? -translation : translation));
	}
};

}

#endif // __PVFULLPARALLELVIEW_H__
