#ifndef __PVPARALLELSCENE_h__
#define __PVPARALLELSCENE_h__

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <picviz/PVAxis.h>
#include <pvparallelview/PVAxisWidget.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVLinesView.h>


#define CRAND() (127 + (random() & 0x7F))

namespace PVParallelView {

class PVParallelScene : public QGraphicsScene
{
	Q_OBJECT
public:
	PVParallelScene(QObject* parent, PVParallelView::PVLinesView* lines_view) : QGraphicsScene(parent), _lines_view(lines_view)
	{
		_lines_view->render_all_imgs(PVParallelView::ImageWidth);
		PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();

		// Add visible zones
		int pos = 0;
		for (PVZoneID z = 0; z < (PVZoneID) images.size() ; z++) {
			QGraphicsPixmapItem* zone_image = addPixmap(QPixmap::fromImage(images[z].bg->qimage()));
			zone_image->setOpacity(0.5);
			_zones.push_back(zone_image);
			if (z < _lines_view->get_zones_manager().get_number_zones()) {
				zone_image->setPos(QPointF(_lines_view->get_zone_absolute_pos(z), 0));
			}
		}

		// Add ALL axes
		PVZoneID nzones = (PVZoneID) _lines_view->get_zones_manager().get_number_cols();
		for (PVZoneID z = 0; z < nzones; z++) {
			Picviz::PVAxis* axis = new Picviz::PVAxis();
			axis->set_name(QString("axis ") + QString::number(z));
			axis->set_color(PVCore::PVColor::fromRgba(CRAND(), CRAND(), CRAND(), 0));
			axis->set_titlecolor(PVCore::PVColor::fromRgba(CRAND(), CRAND(), CRAND(), 0));

			if (z < nzones-1) {
				pos = _lines_view->get_zones_manager().get_zone_absolute_pos(z);
			}
			else {
				// Special case for last axis
				pos += _lines_view->get_zones_manager().get_zone_width(z-1);
			}

			PVParallelView::PVAxisWidget* axisw = new PVParallelView::PVAxisWidget(axis);
			axisw->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
			addItem(axisw);
			_axes.push_back(axisw);
			axisw->add_range_sliders(768, 1000);
		}
	}

	PVParallelView::PVFullParallelView* view()
	{
		return (PVParallelView::PVFullParallelView*)parent() ;
	}

	void update_zones_position(bool update_all = true)
	{
		PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();
		for (PVZoneID zid = _lines_view->get_first_drawn_zone(); zid <= _lines_view->get_last_drawn_zone(); zid++) {
			const PVZoneID img_id = zid-_lines_view->get_first_drawn_zone();
			_zones[img_id]->setPixmap(QPixmap::fromImage(images[img_id].bg->qimage()));
			_zones[img_id]->setPos(QPointF(_lines_view->get_zone_absolute_pos(zid), 0));
		}

		// Update axes position
		PVZoneID nzones = (PVZoneID) _lines_view->get_zones_manager().get_number_cols();
		uint32_t pos = 0;

		PVZoneID z = 1;
		if (! update_all) {
			uint32_t view_x = view()->horizontalScrollBar()->value();
			z = _lines_view->get_zone_from_scene_pos(view_x) + 1;
		}
		for (; z < nzones; z++) {
			if (z < nzones-1) {
				pos = _lines_view->get_zones_manager().get_zone_absolute_pos(z);
			}
			else {
				// Special case for last axis
				pos += _lines_view->get_zones_manager().get_zone_width(z-1);
			};

			_axes[z]->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
		}
	}

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->buttons() == Qt::RightButton) {
			// Translate viewport
			view()->translate_viewport(_translation_start_x - event->scenePos().x());
		}
	}

	void mousePressEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			// Store view position to compute translation
			_translation_start_x = event->scenePos().x();
		}
	}

	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			// translate zones
			translate_and_update_zones_position();
		}
	}

	void wheelEvent(QGraphicsSceneWheelEvent* event)
	{
		int zoom = event->delta() / 2;

		// Local zoom
		if (event->modifiers() == Qt::ControlModifier) {
			PVZoneID zid = _lines_view->get_zone_from_scene_pos(event->scenePos().x());
			uint32_t z_width = _lines_view->get_zone_width(zid);
			if (_lines_view->set_zone_width_and_render(zid, z_width + zoom)) {
				update_zones_position(false);
			}
		}
		//Global zoom
		else
		{
			uint32_t view_x = view()->horizontalScrollBar()->value();
			_lines_view->set_all_zones_width_and_render(view_x, view()->width(), [=](uint32_t width){ return width+zoom; });
			update_zones_position();
		}
	}

public slots:
	void translate_and_update_zones_position()
	{
		uint32_t view_x = view()->horizontalScrollBar()->value();
		_lines_view->translate(view_x, view()->width());
		update_zones_position();
	}

private:
    PVParallelView::PVLinesView* _lines_view;
    qreal _translation_start_x;

    QList<QGraphicsPixmapItem*> _zones;
    QList<PVParallelView::PVAxisWidget*> _axes;
};

}

#endif // __PVPARALLELSCENE_h__
