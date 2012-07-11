#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVParallelScene.h>

PVParallelView::PVSelectionSquareGraphicsItem::PVSelectionSquareGraphicsItem(PVParallelScene* s)
{
	_selection_square = new PVSelectionSquare(s->get_lines_view()->get_zones_manager());
	setPen(QPen(Qt::red, 2));
	setZValue(std::numeric_limits<qreal>::max());
	if (s) {
		s->addItem(this);
	}
}
