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
};

}

#endif // __PVFULLPARALLELVIEW_H__
