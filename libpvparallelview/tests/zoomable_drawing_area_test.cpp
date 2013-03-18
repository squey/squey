
#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QScrollBar64>
#include <QScrollBar>
#include <QPainter>
#include <QMouseEvent>

#include <pvparallelview/PVZoomableDrawingArea.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>
#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>

#include <iostream>
#include <stdio.h>

#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}


#define print_scalar(V) __print_scalar(#V, V)

template <typename V>
void __print_scalar(const char *text, const V &v)
{
	std::cout << text << ": "
	          << v
	          << std::endl;
}

class MyPlottingZDAWA : public PVParallelView::PVZoomableDrawingAreaWithAxes
{
	constexpr static int zoom_steps = 5;
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);

public:
	MyPlottingZDAWA(QWidget *parent = nullptr) :
		PVParallelView::PVZoomableDrawingAreaWithAxes(parent)
	{
		QGraphicsScene *scn = get_scene();

		for(long i = 0; i < (1L<<32); i += 1024 * 1024) {
			long v = i;
			scn->addLine(0, -v, 1L << 32,    -v, QPen(Qt::red, 0));
			scn->addLine(v,  0,    v, -(1L << 32), QPen(Qt::blue, 0));
		}

		QRectF r(0, -(1L << 32), (1L << 32), (1L << 32));
		set_scene_rect(r);

		set_zoom_range(-110, 30);
		set_zoom_value(-110);

		set_x_legend("Axis N");
		set_y_legend("occurrence count");

		set_decoration_color(Qt::white);

		set_ticks_count(8);
	}

protected:
	qreal zoom_to_scale(const int z) const
	{
		return pow(2, z / zoom_steps) * pow(root_step, z % zoom_steps);
	}

	void drawBackground(QPainter *painter, const QRectF &rect)
	{
		painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));

		PVParallelView::PVZoomableDrawingAreaWithAxes::drawBackground(painter, rect);
	}


	void drawForeground(QPainter *painter, const QRectF &rect)
	{
		PVParallelView::PVZoomableDrawingAreaWithAxes::drawForeground(painter, rect);

		int top = get_scene_left_margin();
		int bottom = get_scene_bottom_margin();
		int left = get_scene_left_margin();
		int right = get_scene_right_margin();

		QRectF screen = QRectF(left, top,
		                       rect.width() - left -right,
		                       rect.height() - top -bottom);

		QRectF scene_in_screen = map_from_scene(get_scene_rect());

		QRectF screen_in_scene = map_to_scene(screen);

		print_rect(scene_in_screen);
		print_rect(screen_in_scene);

		qreal scene_width  = get_scene_rect().width();
		qreal scene_width_in_screen = scene_in_screen.width();

		//printf("sw/swis: %f\n", scene_width / scene_width_in_screen);

		qreal tick_gap_0 = scene_width / (qreal)get_ticks_count();
		print_scalar(tick_gap_0);

		qreal log_tick_gap_0 = log(tick_gap_0) / log(get_ticks_count());
		print_scalar(log_tick_gap_0);

		qreal tick_gap_n = scene_in_screen.width() / (qreal)get_ticks_count();
		print_scalar(tick_gap_n);

		qreal log_tick_gap_n = log(tick_gap_n) / log(get_ticks_count());
		print_scalar(log_tick_gap_n);



		// print_scalar(scene_width_in_screen / scene_width);

		// int ticks = get_ticks_count();
		// qreal r = (get_scene_rect().right() - get_scene_rect().left()) / ticks;

		// print_scalar(r);
		// print_scalar(log(r) / log(ticks));
	}

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

	PVParallelView::PVZoomableDrawingAreaWithAxes *pzdawa = new MyPlottingZDAWA;
	pzdawa->install_interactor<PVParallelView::PVZoomableDrawingAreaInteractorSameZoom>();
	pzdawa->resize(600, 400);
	pzdawa->show();
	pzdawa->setWindowTitle("PV Plotting test");

	// PVParallelView::PVZoomableDrawingArea *zzda = new MyZoomingZDA;
	// zzda->resize(600, 400);
	// zzda->show();
	// zzda->setWindowTitle("My Zooming test");

	app.exec();

	return 0;
}
