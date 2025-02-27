//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QScrollBar>
#include <QScrollBar>
#include <QPainter>
#include <QMouseEvent>

#include <pvparallelview/PVAxisZoom.h>
#include <pvparallelview/PVZoomConverterScaledPowerOfTwo.h>
#include <pvparallelview/PVZoomableDrawingArea.h>
#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>

#include <pvparallelview/PVZoomableDrawingAreaConstraints.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

#include <iostream>
#include <cstdio>

#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char* text, const R& r)
{
	std::cout << text << ": " << r.x() << " " << r.y() << ", " << r.width() << " " << r.height()
	          << std::endl;
}

#define print_s(V) print_scalar(V)
#define print_scalar(V) __print_scalar(#V, V)

template <typename V>
void __print_scalar(const char* text, const V& v)
{
	std::cout << text << ": " << v << std::endl;
}

/*****************************************************************************
 * homothetic control
 *****************************************************************************/

class PVZoomableDrawingAreaInteractorHomothetic
    : public PVParallelView::PVZoomableDrawingAreaInteractor
{
  public:
	PVZoomableDrawingAreaInteractorHomothetic(PVWidgets::PVGraphicsView* parent)
	    : PVParallelView::PVZoomableDrawingAreaInteractor(parent)
	{
	}

  protected:
	bool mousePressEvent(PVParallelView::PVZoomableDrawingArea*, QMouseEvent* event) override
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

			QScrollBar* sb;

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
			int inc = (event->angleDelta().y() > 0) ? 1 : -1;
			bool ret =
			    increment_zoom_value(zda, PVParallelView::PVZoomableDrawingAreaConstraints::X |
			                                  PVParallelView::PVZoomableDrawingAreaConstraints::Y,
			                         inc);
			event->setAccepted(true);

			if (ret) {
				zda->reconfigure_view();
				zda->get_viewport()->update();
			}
		}

		return true;
	}

  private:
	QPoint _pan_reference;
};

class PVZoomableDrawingAreaConstraintsHomothetic
    : public PVParallelView::PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const override { return true; }

	bool zoom_y_available() const override { return true; }

	bool set_zoom_value(int /*axes*/,
	                    int value,
	                    PVParallelView::PVAxisZoom& zx,
	                    PVParallelView::PVAxisZoom& zy) override
	{
		set_clamped_value(zx, value);
		set_clamped_value(zy, value);
		return true;
	}

	bool increment_zoom_value(int /*axes*/,
	                          int value,
	                          PVParallelView::PVAxisZoom& zx,
	                          PVParallelView::PVAxisZoom& zy) override
	{
		set_clamped_value(zx, zx.get_value() + value);
		set_clamped_value(zy, zy.get_value() + value);
		return true;
	}

	void adjust_pan(QScrollBar* /*xsb*/, QScrollBar* /*ysb*/) override {}
};

/*****************************************************************************
 * free 2D control
 *****************************************************************************/

class PVZoomableDrawingAreaInteractorFree : public PVParallelView::PVZoomableDrawingAreaInteractor
{
  public:
	PVZoomableDrawingAreaInteractorFree(PVWidgets::PVGraphicsView* parent)
	    : PVParallelView::PVZoomableDrawingAreaInteractor(parent)
	{
	}

  protected:
	bool mousePressEvent(PVParallelView::PVZoomableDrawingArea*, QMouseEvent* event) override
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

			QScrollBar* sb;

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
		int inc = (event->angleDelta().y() > 0) ? 1 : -1;
		int mask = 0;

		if (event->modifiers() == Qt::NoModifier) {
			mask = PVParallelView::PVZoomableDrawingAreaConstraints::X |
			       PVParallelView::PVZoomableDrawingAreaConstraints::Y;
		} else if (event->modifiers() == Qt::ControlModifier) {
			mask = PVParallelView::PVZoomableDrawingAreaConstraints::X;
		} else if (event->modifiers() == Qt::ShiftModifier) {
			mask = PVParallelView::PVZoomableDrawingAreaConstraints::Y;
		}

		if (mask != 0) {
			event->setAccepted(true);

			if (increment_zoom_value(zda, mask, inc)) {
				zda->reconfigure_view();
				zda->get_viewport()->update();
				zoom_has_changed(zda, mask);
			}
		}

		return event->isAccepted();
	}

  private:
	QPoint _pan_reference;
};

class PVZoomableDrawingAreaConstraintsFree : public PVParallelView::PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const override { return true; }

	bool zoom_y_available() const override { return true; }

	bool set_zoom_value(int axes,
	                    int value,
	                    PVParallelView::PVAxisZoom& zx,
	                    PVParallelView::PVAxisZoom& zy) override
	{
		if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::X) {
			set_clamped_value(zx, value);
		}
		if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::Y) {
			set_clamped_value(zy, value);
		}
		return true;
	}

	bool increment_zoom_value(int axes,
	                          int value,
	                          PVParallelView::PVAxisZoom& zx,
	                          PVParallelView::PVAxisZoom& zy) override
	{
		if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::X) {
			set_clamped_value(zx, zx.get_value() + value);
		}
		if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::Y) {
			set_clamped_value(zy, zy.get_value() + value);
		}
		return true;
	}

	void adjust_pan(QScrollBar* /*xsb*/, QScrollBar* /*ysb*/) override {}
};

/*****************************************************************************
 * homothetic zoom with Y pan control
 *****************************************************************************/

class PVZoomableDrawingAreaInteractorZPV : public PVParallelView::PVZoomableDrawingAreaInteractor
{
  public:
	PVZoomableDrawingAreaInteractorZPV(PVWidgets::PVGraphicsView* parent)
	    : PVParallelView::PVZoomableDrawingAreaInteractor(parent)
	{
	}

  protected:
	bool mousePressEvent(PVParallelView::PVZoomableDrawingArea*, QMouseEvent* event) override
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

			QScrollBar* sb = zda->get_vertical_scrollbar();
			sb->setValue(sb->value() + delta.y());
			pan_has_changed(zda);
			event->setAccepted(true);
		}
		return event->isAccepted();
	}

	bool wheelEvent(PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event) override
	{
		int inc = (event->angleDelta().y() > 0) ? 1 : -1;

		if (increment_zoom_value(zda, PVParallelView::PVZoomableDrawingAreaConstraints::X |
		                                  PVParallelView::PVZoomableDrawingAreaConstraints::Y,
		                         inc)) {
			std::cout << "update!" << std::endl;
			zda->reconfigure_view();
			zda->get_viewport()->update();
			zoom_has_changed(zda, PVParallelView::PVZoomableDrawingAreaConstraints::X |
			                          PVParallelView::PVZoomableDrawingAreaConstraints::Y);
		}

		event->setAccepted(true);

		return true;
	}

	bool resizeEvent(PVParallelView::PVZoomableDrawingArea* zda, QResizeEvent*) override
	{
		zda->reconfigure_view();
		return true;
	}

  private:
	QPoint _pan_reference;
};

class PVZoomableDrawingAreaConstraintsZPV : public PVParallelView::PVZoomableDrawingAreaConstraints
{
	bool zoom_x_available() const override { return true; }

	bool zoom_y_available() const override { return true; }

	bool set_zoom_value(int /*axes*/,
	                    int value,
	                    PVParallelView::PVAxisZoom& zx,
	                    PVParallelView::PVAxisZoom& zy) override
	{
		set_clamped_value(zx, value);
		set_clamped_value(zy, value);
		return true;
	}

	bool increment_zoom_value(int /*axes*/,
	                          int value,
	                          PVParallelView::PVAxisZoom& zx,
	                          PVParallelView::PVAxisZoom& zy) override
	{
		set_clamped_value(zx, zx.get_value() + value);
		set_clamped_value(zy, zy.get_value() + value);
		return true;
	}

	void adjust_pan(QScrollBar* xsb, QScrollBar* /*ysb*/) override
	{
		int64_t mid = ((int64_t)xsb->maximum() + xsb->minimum()) / 2;
		xsb->setValue(mid);
	}
};

/*****************************************************************************
 * test views
 *****************************************************************************/

class MyScalingZDAWA : public PVParallelView::PVZoomableDrawingAreaWithAxes
{
	constexpr static int zoom_steps = 5;
	constexpr static int zoom_min = -22 * zoom_steps;
	constexpr static int zoom_max = 8 * zoom_steps;

	using zoom_converter_t = PVParallelView::PVZoomConverterScaledPowerOfTwo<zoom_steps>;

  public:
	MyScalingZDAWA(QWidget* parent = nullptr)
	    : PVParallelView::PVZoomableDrawingAreaWithAxes(parent)
	{
		set_gl_viewport();

		QGraphicsScene* scn = get_scene();

		PVWidgets::PVGraphicsViewInteractorBase* inter;
		inter = declare_interactor<PVZoomableDrawingAreaInteractorFree>();
		set_constraints(new PVZoomableDrawingAreaConstraintsFree());
		register_front_all(inter);

		install_default_scene_interactor();

		for (size_t i = 0; i < ((size_t)1 << 32); i += 1024 * 1024) {
			size_t v = i;
			scn->addLine(0, v, (size_t)1 << 32, v, QPen(Qt::red, 0));
			scn->addLine(v, 0, v, ((size_t)1 << 32), QPen(Qt::blue, 0));
		}

		QRectF r(0, 0, ((size_t)1 << 32), ((size_t)1 << 32));
		set_scene_rect(r);

		PVParallelView::PVZoomConverter* zc = new zoom_converter_t();

		get_x_axis_zoom().set_range(zoom_min, zoom_max);
		get_x_axis_zoom().set_default_value(zoom_min);
		get_x_axis_zoom().set_zoom_converter(zc);

		get_y_axis_zoom().set_range(zoom_min, zoom_max);
		get_y_axis_zoom().set_default_value(zoom_min);
		get_y_axis_zoom().set_zoom_converter(zc);
		set_zoom_value(PVParallelView::PVZoomableDrawingAreaConstraints::X |
		                   PVParallelView::PVZoomableDrawingAreaConstraints::Y,
		               zoom_min);

		set_x_legend("Axis N");
		set_y_legend("occurrence count");

		set_decoration_color(Qt::white);

		set_ticks_per_level(8);
	}

	~MyScalingZDAWA() override
	{
		// PVZoomableDrawingArea does not care about the constraints deletion
		delete get_constraints();

		// delete only one because it is shared by the 2 PVAxisZoom
		delete get_x_axis_zoom().get_zoom_converter();
	}

  protected:
	void drawBackground(QPainter* painter, const QRectF& rect) override
	{
		painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));

		PVParallelView::PVZoomableDrawingAreaWithAxes::drawBackground(painter, rect);
	}

	void drawForeground(QPainter* painter, const QRectF& rect) override
	{
		PVParallelView::PVZoomableDrawingAreaWithAxes::drawForeground(painter, rect);
	}

	void keyPressEvent(QKeyEvent* event) override
	{
		if (event->key() == Qt::Key_Home) {
			center_on(get_scene_rect().center());
		}
	}
};

class MyZoomingZDA : public PVParallelView::PVZoomableDrawingArea
{
	using zoom_converter_t = PVParallelView::PVZoomConverterScaledPowerOfTwo<5>;

  public:
	MyZoomingZDA(QWidget* parent = nullptr) : PVParallelView::PVZoomableDrawingArea(parent)
	{
		QGraphicsScene* scn = get_scene();

		set_scene_margins(40, 0, 40, 0);

		set_transformation_anchor(PVWidgets::PVGraphicsView::AnchorUnderMouse);
		set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
		set_vertical_scrollbar_policy(Qt::ScrollBarAlwaysOn);

		PVWidgets::PVGraphicsViewInteractorBase* inter =
		    declare_interactor<PVZoomableDrawingAreaInteractorZPV>();
		register_front_all(inter);
		set_constraints(new PVZoomableDrawingAreaConstraintsZPV());

		install_default_scene_interactor();

		QRectF r(-512, 0, 1024, 1024);
		scn->addRect(r.adjusted(1, 1, -1, -1), QPen(Qt::blue));
		for (int i = 0; i < 255; ++i) {
			int v = i * 4;
			scn->addLine(-10, v, 10, v, QColor(255 - i, i, 0));
		}
		scn->addLine(-10, 1024, 10, 1024, QColor(0, 255, 0));

		set_scene_rect(r);
		scn->setSceneRect(r);

		PVParallelView::PVZoomConverter* zc = new zoom_converter_t();

		get_x_axis_zoom().set_range(-10, 100);
		get_x_axis_zoom().set_default_value(0);
		get_x_axis_zoom().set_zoom_converter(zc);

		get_y_axis_zoom().set_range(-10, 100);
		get_y_axis_zoom().set_default_value(0);
		get_y_axis_zoom().set_zoom_converter(zc);
		set_zoom_value(PVParallelView::PVZoomableDrawingAreaConstraints::X |
		                   PVParallelView::PVZoomableDrawingAreaConstraints::Y,
		               0);
	}

	~MyZoomingZDA() override
	{
		// PVZoomableDrawingArea does not care about the constraints deletion
		delete get_constraints();

		// delete only one because it is shared by the 2 PVAxisZoom
		delete get_x_axis_zoom().get_zoom_converter();
	}

	void drawForeground(QPainter* painter, const QRectF&) override
	{
		/*
		int c = rect.width() / 2;
		QPen pen(Qt::red);
		pen.setWidth(3);

		//painter->save();
		//painter->resetTransform();
		painter->setPen(pen);
		painter->drawLine(c, 0, c, rect.height());
		//painter->restore();*/

		painter->save();
		QPen pen(Qt::red);
		pen.setWidth(4);
		painter->setPen(pen);

		QRectF rect_draw = map_from_scene(get_visible_scene_rect());
		painter->drawRect(rect_draw);
		painter->restore();
	}
};

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	PVParallelView::PVZoomableDrawingAreaWithAxes* pzdawa = new MyScalingZDAWA;
	pzdawa->set_y_axis_inverted(true);
	pzdawa->resize(600, 600);
	pzdawa->show();
	pzdawa->setWindowTitle("PV Scaling test");

	PVParallelView::PVZoomableDrawingArea* zzda = new MyZoomingZDA;
	zzda->set_y_axis_inverted(true);
	zzda->resize(600, 600);
	zzda->show();
	zzda->setWindowTitle("My Zooming test");

	app.exec();

	return 0;
}
