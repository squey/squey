
#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QScrollBar64>
#include <QScrollBar>
#include <QPainter>
#include <QMouseEvent>

#include <pvparallelview/PVZoomableDrawingArea.h>

#include <iostream>


#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

class MyPlottingZDA : public PVParallelView::PVZoomableDrawingArea
{
public:
	MyPlottingZDA(QWidget *parent = nullptr) :
		PVParallelView::PVZoomableDrawingArea(parent),
		_first_resize(true)
	{
		QGraphicsScene *scn = get_scene();

		for(long i = 0; i < (1L<<32); i += 1024 * 1024) {
			long v = i;
			scn->addLine(0, -v, 1L << 32,    -v, QPen(Qt::red, 0));
			scn->addLine(v,  0,    v, -(1L << 32), QPen(Qt::blue, 0));
		}

		_top = _right = 20;
		_left = _bottom = 50;
		QRectF r(0, -(1L << 32), (1L << 32), (1L << 32));
		set_scene_rect(r);

		print_rect(scn->sceneRect());

		set_scene_margins(_left, _right, _top, _bottom);
		set_alignment(Qt::AlignLeft | Qt::AlignBottom);
		set_transformation_anchor(PVWidgets::PVGraphicsView::AnchorUnderMouse);
		set_zoom_range(-200, 200);
	}

	qreal zoom_to_scale(const int z) const
	{
		constexpr static int zoom_steps = 5;
		constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);

		return pow(2, z / zoom_steps) * pow(root_step, z % zoom_steps);
	}


	void drawForeground(QPainter *painter, const QRectF &rect)
	{
		int top = _top;
		int left = _left;
		int right = rect.width() - _right;
		int bottom = rect.height() - _bottom;
		painter->save();
		painter->resetTransform();
		painter->drawLine(left, bottom, right, bottom);
		painter->drawText(left, bottom, get_real_viewport_width(), 50,
		                  Qt::AlignRight | Qt::AlignBottom,
		                  QString("axis N"), nullptr);

		painter->drawLine(left, top, left, bottom);
		painter->drawText(left, 0, get_real_viewport_width(), 20,
		                  Qt::AlignLeft | Qt::AlignTop,
		                  QString("occurences number"), nullptr);


		QPen pen(Qt::blue);
		painter->setPen(pen);
		painter->drawLine(left, bottom, left, bottom + 5);

		QRectF s = map_to_scene(QRectF(left, top, right - left - 1, bottom - top));

		painter->drawText(left, bottom + 5, get_real_viewport_width(), 20,
		                  Qt::AlignLeft | Qt::AlignTop,
		                  QString::number((long)(s.x())) + " " + QString::number(log2(s.x())), nullptr);


		painter->drawLine(right, bottom, right, bottom + 5);
		painter->drawText(left, bottom + 5, get_real_viewport_width(), 20,
		                  Qt::AlignRight | Qt::AlignTop,
		                  QString::number((long)((s.x() + s.width()))) + " " +
		                  QString::number(log2(s.x() + s.width())), nullptr);


		painter->restore();
	}

	void keyPressEvent(QKeyEvent *event)
	{
		if (event->key() == Qt::Key_Space) {
			center_on(0., 0.);
			event->accept();
		} else {
			PVParallelView::PVZoomableDrawingArea::keyPressEvent(event);
		}
	}

	void resizeEvent(QResizeEvent *event)
	{
		PVParallelView::PVZoomableDrawingArea::resizeEvent(event);
		if (_first_resize) {
			center_on(0., 0.);
			_first_resize = false;
		}
	}
private:
	bool _first_resize;
	int _top, _left, _bottom, _right;
};


class MyZoomingZDA : public PVParallelView::PVZoomableDrawingArea
{
public:
	MyZoomingZDA(QWidget *parent = nullptr) :
		PVParallelView::PVZoomableDrawingArea(parent)
	{
		QGraphicsScene *scn = get_scene();

		for(int i = 0; i < 255; ++i) {
			int v = i * 4;
			scn->addLine(-10, v, 10, v, QColor(255 - i, i, 0));
		}
		scn->addLine(-10, 1024, 10, 1024, QColor(0, 255, 0));

		QRectF r(-512, 0, 1024, 1024);
		set_scene_rect(r);
		scn->setSceneRect(r);

		set_pan_policy(PVParallelView::PVZoomableDrawingArea::AlongY);
		set_transformation_anchor(PVWidgets::PVGraphicsView::AnchorUnderMouse);
		set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);
		set_vertical_scrollbar_policy(Qt::ScrollBarAlwaysOn);
	}

	qreal zoom_to_scale(const int w) const
	{
		constexpr static int zoom_steps = 5;
		constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);

		return pow(2, w / zoom_steps) * pow(root_step, w % zoom_steps);
	}

	void drawForeground(QPainter *painter, const QRectF &rect)
	{
		int c = rect.width() / 2;
		QPen pen(Qt::red);
		pen.setWidth(3);

		painter->save();
		painter->resetTransform();
		painter->setPen(pen);
		painter->drawLine(c, 0, c, rect.height());
		painter->restore();
	}
};



int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	PVParallelView::PVZoomableDrawingArea *pzda = new MyPlottingZDA;
	pzda->resize(600, 400);
	pzda->show();
	pzda->setWindowTitle("Plotting test");

	PVParallelView::PVZoomableDrawingArea *zzda = new MyZoomingZDA;
	zzda->resize(600, 400);
	zzda->show();
	zzda->setWindowTitle("Zooming test");

	app.exec();

	return 0;
}
