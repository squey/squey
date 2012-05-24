#ifndef __PVFULLPARALLELVIEW_H__
#define __PVFULLPARALLELVIEW_H__

#include <QGraphicsView>
#include <QScrollBar>

namespace PVParallelView {

class PVParallelScene;

class PVFullParallelView : public QGraphicsView
{
	Q_OBJECT
public:
	PVFullParallelView()
	{
		//PVParallelView::PVParallelScene* s = (PVParallelView::PVParallelScene*) scene();
		//connect(horizontalScrollBar(), SIGNAL(valueChanged()), s, SLOT(translate_and_update_zones_position));
	}

	virtual void translate_viewport(int translation)
	{
		QScrollBar *hBar = horizontalScrollBar();
		hBar->setValue(hBar->value() + (isRightToLeft() ? -translation : translation));
	}
};

}

#endif // __PVFULLPARALLELVIEW_H__
