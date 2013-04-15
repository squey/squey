
#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QScrollBar64>
#include <QScrollBar>
#include <QPainter>
#include <QMouseEvent>

#include <pvkernel/core/general.h>

#include <pvparallelview/PVAxisZoom.h>
#include <pvparallelview/PVZoomConverterScaledPowerOfTwo.h>
#include <pvparallelview/PVZoomableDrawingArea.h>
#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>

#include <pvparallelview/PVZoomableDrawingAreaConstraints.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

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

#define print_s(V) print_scalar(V)
#define print_scalar(V) __print_scalar(#V, V)

template <typename V>
void __print_scalar(const char *text, const V &v)
{
	std::cout << text << ": "
	          << v
	          << std::endl;
}

/*****************************************************************************
 * homothetic control
 *****************************************************************************/

class PVZoomableDrawingAreaInteractorHomothetic : public PVParallelView::PVZoomableDrawingAreaInteractor
{
public:
	PVZoomableDrawingAreaInteractorHomothetic(PVWidgets::PVGraphicsView* parent) :
		PVParallelView::PVZoomableDrawingAreaInteractor(parent)
	{}

protected:
	bool mousePressEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event)
	{
		if (event->button() == Qt::RightButton) {
			_pan_reference = event->pos();
			event->setAccepted(true);
		}
		return event->isAccepted();
	}

	bool mouseMoveEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event)
	{
		if (event->buttons() == Qt::RightButton) {

			QPoint delta = _pan_reference - event->pos();
			_pan_reference = event->pos();

			QScrollBar64 *sb;

			sb = zda->get_horizontal_scrollbar();
			sb->setValue(sb->value() + delta.x());

			sb = zda->get_vertical_scrollbar();
			sb->setValue(sb->value() + delta.y());
			pan_has_changed(zda);
			event->setAccepted(true);
		}

		return event->isAccepted();
	}

	bool wheelEvent(PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event) override
	{
		if (event->modifiers() == Qt::NoModifier) {
			int inc = (event->delta() > 0)?1:-1;
			bool ret = increment_zoom_value(zda,
			                                PVParallelView::PVZoomableDrawingAreaConstraints::X | PVParallelView::PVZoomableDrawingAreaConstraints::Y,
			                                inc);
			event->setAccepted(true);

			if (ret) {
				zda->reconfigure_view();
				zda->update();
			}
		}

		return true;
	}

private:
	QPoint _pan_reference;
};

class PVZoomableDrawingAreaConstraintsHomothetic : public PVParallelView::PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const
	{
		return true;
	}

	bool zoom_y_available() const
	{
		return true;
	}

	bool set_zoom_value(int /*axes*/, int value,
	                    PVParallelView::PVAxisZoom &zx,
	                    PVParallelView::PVAxisZoom &zy)
	{
		set_clamped_value(zx, value);
		set_clamped_value(zy, value);
		return true;
	}

	bool increment_zoom_value(int /*axes*/, int value,
	                          PVParallelView::PVAxisZoom &zx,
	                          PVParallelView::PVAxisZoom &zy)
	{
		set_clamped_value(zx, zx.get_value() + value);
		set_clamped_value(zy, zy.get_value() + value);
		return true;
	}

	void adjust_pan(QScrollBar64 */*xsb*/, QScrollBar64 */*ysb*/)
	{}
};

/*****************************************************************************
 * free 2D control
 *****************************************************************************/

class PVZoomableDrawingAreaInteractorFree : public PVParallelView::PVZoomableDrawingAreaInteractor
{
public:
	PVZoomableDrawingAreaInteractorFree(PVWidgets::PVGraphicsView* parent) :
		PVParallelView::PVZoomableDrawingAreaInteractor(parent)
	{}

protected:
	bool mousePressEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event)
	{
		if (event->button() == Qt::RightButton) {
			_pan_reference = event->pos();
			event->setAccepted(true);
		}
		return event->isAccepted();
	}

	bool mouseMoveEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event)
	{
		if (event->buttons() == Qt::RightButton) {

			QPoint delta = _pan_reference - event->pos();
			_pan_reference = event->pos();

			QScrollBar64 *sb;

			sb = zda->get_horizontal_scrollbar();
			sb->setValue(sb->value() + delta.x());

			sb = zda->get_vertical_scrollbar();
			sb->setValue(sb->value() + delta.y());
			pan_has_changed(zda);
			event->setAccepted(true);
		}

		return event->isAccepted();
	}

	bool wheelEvent(PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event) override
	{
		int inc = (event->delta() > 0)?1:-1;
		int mask = 0;

		if (event->modifiers() == Qt::NoModifier) {
			mask = PVParallelView::PVZoomableDrawingAreaConstraints::X | PVParallelView::PVZoomableDrawingAreaConstraints::Y;
		} else if (event->modifiers() == Qt::ControlModifier) {
			mask = PVParallelView::PVZoomableDrawingAreaConstraints::X;
		} else if (event->modifiers() == Qt::ShiftModifier) {
			mask = PVParallelView::PVZoomableDrawingAreaConstraints::Y;
		}

		if (mask != 0) {
			event->setAccepted(true);

			if (increment_zoom_value(zda, mask, inc)) {
				zda->reconfigure_view();
				zda->update();
				zoom_has_changed(zda);
			}
		}

		return event->isAccepted();
	}

private:
	QPoint _pan_reference;
};

class PVZoomableDrawingAreaConstraintsFree : public PVParallelView::PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const
	{
		return true;
	}

	bool zoom_y_available() const
	{
		return true;
	}

	bool set_zoom_value(int axes, int value,
	                    PVParallelView::PVAxisZoom &zx,
	                    PVParallelView::PVAxisZoom &zy)
	{
		if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::X) {
			set_clamped_value(zx, value);
		}
		if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::Y) {
			set_clamped_value(zy, value);
		}
		return true;
	}

	bool increment_zoom_value(int axes, int value,
	                          PVParallelView::PVAxisZoom &zx,
	                          PVParallelView::PVAxisZoom &zy)
	{
		if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::X) {
			set_clamped_value(zx, zx.get_value() + value);
		}
		if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::Y) {
			set_clamped_value(zy, zy.get_value() + value);
		}
		return true;
	}

	void adjust_pan(QScrollBar64 */*xsb*/, QScrollBar64 */*ysb*/)
	{}
};

/*****************************************************************************
 * homothetic zoom with Y pan control
 *****************************************************************************/

class PVZoomableDrawingAreaInteractorZPV : public PVParallelView::PVZoomableDrawingAreaInteractor
{
public:
	PVZoomableDrawingAreaInteractorZPV(PVWidgets::PVGraphicsView* parent) :
		PVParallelView::PVZoomableDrawingAreaInteractor(parent)
	{}

protected:
	bool mousePressEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event) override
	{
		if (event->button() == Qt::RightButton) {
			_pan_reference = event->pos();
			event->setAccepted(true);
		}
		return event->isAccepted();
	}

	bool mouseMoveEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event) override
	{
		if (event->buttons() == Qt::RightButton) {

			QPoint delta = _pan_reference - event->pos();
			_pan_reference = event->pos();

			QScrollBar64 *sb = zda->get_vertical_scrollbar();
			sb->setValue(sb->value() + delta.y());
			pan_has_changed(zda);
			event->setAccepted(true);
		}
		return event->isAccepted();
	}

	bool wheelEvent(PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event) override
	{
		int inc = (event->delta() > 0)?1:-1;

		if (increment_zoom_value(zda, PVParallelView::PVZoomableDrawingAreaConstraints::X | PVParallelView::PVZoomableDrawingAreaConstraints::Y, inc)) {
			std::cout << "update!" << std::endl;
			zda->reconfigure_view();
			zda->update();
			zoom_has_changed(zda);
		}

		event->setAccepted(true);

		return true;
	}

	bool resizeEvent(PVParallelView::PVZoomableDrawingArea* zda, QResizeEvent* event) override
	{
		zda->reconfigure_view();
		return true;
	}

private:
	QPoint _pan_reference;
};

class PVZoomableDrawingAreaConstraintsZPV : public PVParallelView::PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const
	{
		return true;
	}

	bool zoom_y_available() const
	{
		return true;
	}

	bool set_zoom_value(int /*axes*/, int value,
	                    PVParallelView::PVAxisZoom &zx,
	                    PVParallelView::PVAxisZoom &zy)
	{
		set_clamped_value(zx, value);
		set_clamped_value(zy, value);
		return true;
	}

	bool increment_zoom_value(int /*axes*/, int value,
	                          PVParallelView::PVAxisZoom &zx,
	                          PVParallelView::PVAxisZoom &zy)
	{
		set_clamped_value(zx, zx.get_value() + value);
		set_clamped_value(zy, zy.get_value() + value);
		return true;
	}

	void adjust_pan(QScrollBar64 *xsb, QScrollBar64 */*ysb*/)
	{
		int64_t mid = ((int64_t)xsb->maximum() + xsb->minimum()) / 2;
		xsb->setValue(mid);
	}
};

/*****************************************************************************
 * test views
 *****************************************************************************/

class MyPlottingZDAWA : public PVParallelView::PVZoomableDrawingAreaWithAxes
{
	constexpr static int zoom_steps = 5;
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);

	typedef PVParallelView::PVZoomConverterScaledPowerOfTwo<5> zoom_converter_t;
public:
	MyPlottingZDAWA(QWidget *parent = nullptr) :
		PVParallelView::PVZoomableDrawingAreaWithAxes(parent)
	{
		QGraphicsScene *scn = get_scene();

		PVWidgets::PVGraphicsViewInteractorBase *inter;
#if 0
		inter = declare_interactor<PVZoomableDrawingAreaInteractorHomothetic>();
		set_constraints(new PVZoomableDrawingAreaConstraintsHomothetic());
#else
		inter = declare_interactor<PVZoomableDrawingAreaInteractorFree>();
		set_constraints(new PVZoomableDrawingAreaConstraintsFree());
#endif
		register_front_all(inter);

		install_default_scene_interactor();

		for(long i = 0; i < (1L<<32); i += 1024 * 1024) {
			long v = i;
			scn->addLine(0, -v, 1L << 32,    -v, QPen(Qt::red, 0));
			scn->addLine(v,  0,    v, -(1L << 32), QPen(Qt::blue, 0));
		}

		QRectF r(0, -(1L << 32), (1L << 32), (1L << 32));
		set_scene_rect(r);

		// setMaximumWidth(1024);
		// setMaximumHeight(1024);

		PVParallelView::PVZoomConverter *zc = new zoom_converter_t();

		get_x_axis_zoom().set_range(-110, 30);
		get_x_axis_zoom().set_default_value(-110);
		get_x_axis_zoom().set_zoom_converter(zc);

		get_y_axis_zoom().set_range(-110, 30);
		get_y_axis_zoom().set_default_value(-110);
		get_y_axis_zoom().set_zoom_converter(zc);
		set_zoom_value(PVParallelView::PVZoomableDrawingAreaConstraints::X
		               | PVParallelView::PVZoomableDrawingAreaConstraints::Y,
		               -110);

		set_x_legend("Axis N");
		set_y_legend("occurrence count");

		set_decoration_color(Qt::white);

		set_ticks_per_level(8);
	}

	~MyPlottingZDAWA()
	{
		// PVZoomableDrawingArea does not care about the constraints deletion
		delete get_constraints();

		// delete only one because it is shared by the 2 PVAxisZoom
		delete get_x_axis_zoom().get_zoom_converter();
	}

protected:
	void drawBackground(QPainter *painter, const QRectF &rect)
	{
		painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));

		PVParallelView::PVZoomableDrawingAreaWithAxes::drawBackground(painter, rect);
	}


	void drawForeground(QPainter *painter, const QRectF &rect)
	{
		PVParallelView::PVZoomableDrawingAreaWithAxes::drawForeground(painter, rect);

#if 0
		int top = get_scene_top_margin();
		int bottom = get_scene_bottom_margin();
		int left = get_scene_left_margin();
		int right = get_scene_right_margin();

		print_scalar(left);
		QRectF screen = QRectF(left, top,
		                       rect.width() - left -right,
		                       rect.height() - top -bottom);

		QRectF scene_in_screen = map_from_scene(get_scene_rect());
		QRectF screen_in_scene = map_to_scene(screen);

		print_rect(scene_in_screen);
		print_rect(screen_in_scene);

		print_scalar(top);
		print_scalar(bottom);
		print_scalar(left);
		print_scalar(right);

		qreal scene_width  = get_scene_rect().width();
		//qreal scene_width_in_screen = scene_in_screen.width();

		qreal ref_scale = zoom_to_scale(get_x_axis_zoom().get_min());
		// print_scalar(scene_width * ref_scale);

		qreal ref_ticks_gap = scene_width * ref_scale / get_ticks_count();
		// print_scalar(ref_ticks_gap);

		qreal cur_scale = zoom_to_scale(get_zoom_value());
		qreal cur_ticks_gap = scene_width * cur_scale / get_ticks_count();
		// print_scalar(cur_ticks_gap);

		print_scalar(log(cur_ticks_gap) / log(ref_ticks_gap));
		qreal ratio = cur_scale / ref_scale;
		// print_scalar(ratio);

		qreal root_ratio = (int)ratio;
		//qreal step_tick_gap = ref_ticks_gap * root_ratio;
		// std::cout << "draw ticks for " << root_ratio << " (" << step_tick_gap << ")" << std::endl;
		if ((ratio / root_ratio) > 1.414) {
			// std::cout << "draw sub-ticks for " << ratio << std::endl;
		}
#endif
	}

};

class MyZoomingZDA : public PVParallelView::PVZoomableDrawingArea
{
	typedef PVParallelView::PVZoomConverterScaledPowerOfTwo<5> zoom_converter_t;

public:
	MyZoomingZDA(QWidget *parent = nullptr) :
		PVParallelView::PVZoomableDrawingArea(parent)
	{
		QGraphicsScene *scn = get_scene();

		set_transformation_anchor(PVWidgets::PVGraphicsView::AnchorUnderMouse);
		set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);
		set_vertical_scrollbar_policy(Qt::ScrollBarAlwaysOn);

		PVWidgets::PVGraphicsViewInteractorBase *inter =
			declare_interactor<PVZoomableDrawingAreaInteractorZPV>();
		register_front_all(inter);
		set_constraints(new PVZoomableDrawingAreaConstraintsZPV());

		install_default_scene_interactor();

		for(int i = 0; i < 255; ++i) {
			int v = i * 4;
			scn->addLine(-10, v, 10, v, QColor(255 - i, i, 0));
		}
		scn->addLine(-10, 1024, 10, 1024, QColor(0, 255, 0));

		QRectF r(-512, 0, 1024, 1024);
		set_scene_rect(r);
		scn->setSceneRect(r);

		PVParallelView::PVZoomConverter *zc = new zoom_converter_t();

		get_x_axis_zoom().set_range(0, 100);
		get_x_axis_zoom().set_default_value(0);
		get_x_axis_zoom().set_zoom_converter(zc);

		get_y_axis_zoom().set_range(0, 100);
		get_y_axis_zoom().set_default_value(0);
		get_y_axis_zoom().set_zoom_converter(zc);
		set_zoom_value(PVParallelView::PVZoomableDrawingAreaConstraints::X
		               | PVParallelView::PVZoomableDrawingAreaConstraints::Y,
		               0);
	}

	~MyZoomingZDA()
	{
		// PVZoomableDrawingArea does not care about the constraints deletion
		delete get_constraints();

		// delete only one because it is shared by the 2 PVAxisZoom
		delete get_x_axis_zoom().get_zoom_converter();
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
	pzdawa->resize(600, 600);
	pzdawa->show();
	pzdawa->setWindowTitle("PV Plotting test");

	PVParallelView::PVZoomableDrawingArea *zzda = new MyZoomingZDA;
	zzda->resize(600, 600);
	//zzda->show();
	zzda->setWindowTitle("My Zooming test");

	app.exec();

	return 0;
}
