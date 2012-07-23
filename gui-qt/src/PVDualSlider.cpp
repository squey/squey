/**
 * \file PVDualSlider.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <cmath>

#include <QtCore>
#include <QtGui>


#include <pvkernel/core/general.h>

#include <PVMainWindow.h>

#include <PVDualSlider.h>

using std::max;
using std::min;

/******************************************************************************
 *
 * PVInspector::PVDualSlider::PVDualSlider
 *
 *****************************************************************************/
PVInspector::PVDualSlider::PVDualSlider(PVMainWindow *mw, QWidget *parent) : QWidget(parent)
{
	main_window = mw;

	last_mouse_press_position = QPointF(0,0);
	SELECTED_SLIDER = -1;
	SLIDER_WIDTH = 5;
	XMAX = 120.0;
	YMAX = 50.0;

	/* We create the QPainter of the PVDualSlider */
	painter = new QPainter();

	/* We define the shape of the left slider in abstract coordinates system */
	left_slider_polygon << QPointF(-10.0, 10.0) << QPointF(0,0) << QPointF(0,10) << QPointF(-0.2,10) << QPointF(-0.2,-10) << QPointF(0,-10) << QPointF(0,0) << QPointF(-10,-10) << QPointF(-10,10);
	/* We define the shape of the right slider in abstract coordinates system */
	right_slider_polygon << QPointF(10,10) << QPointF(0,0) << QPointF(0,10) << QPointF(0.2,10) << QPointF(0.2,-10) << QPointF(0,-10) << QPointF(0,0) << QPointF(10,-10) << QPointF(10,10);

	/* We set the extents of each slider in the abstract coordinates system */
	sliders_relative_extents << QPointF(-10,0) << QPointF(0,10);

	/* We set th initial positions of the sliders */
	sliders_positions[0] = 0.0;
	sliders_positions[1] = 1.0;

	/* We compute the margins needed */
	left_margin = sliders_relative_extents[0].y() - sliders_relative_extents[0].x();
	right_margin = sliders_relative_extents[1].y() - sliders_relative_extents[1].x();
}

/******************************************************************************
 *
 * PVInspector::PVDualSlider::get_selected_slider_index()
 *
 *****************************************************************************/
int PVInspector::PVDualSlider::get_selected_slider_index(float x)
{
	PVLOG_DEBUG("PVInspector::PVDualSlider::%s\n", __FUNCTION__);
	int k;
	float float_width = (float) width();
	float float_height = (float) height();

	/* We compute the scale ratio from viewport to abstract */
	kx = float_width/XMAX;
	ky = float_height/YMAX;
	/* Now we check each slider to see if one gets the hit */
	for (k=0; k<2; k++) {
		if (((left_margin + sliders_positions[k]*100 + sliders_relative_extents[k].x())*kx < x) && (x < kx*(left_margin + sliders_positions[k]*100 + sliders_relative_extents[k].y()))) {
			return k;
		}
	}

	/* If we have not returned yet, no slider gets the hit */
	return -1;
}

/******************************************************************************
 *
 * PVInspector::PVDualSlider::get_slider_value()
 *
 *****************************************************************************/
float PVInspector::PVDualSlider::get_slider_value(int index)
{
	return sliders_positions[index];
}

/******************************************************************************
 *
 * PVInspector::PVDualSlider::mouseMoveEvent()
 *
 *****************************************************************************/
void PVInspector::PVDualSlider::mouseMoveEvent(QMouseEvent *event)
{
	/* VARIABLES */
	float float_height;
	float float_width;

	/* CODE */
	/* If there is no selected slider, get out of here */
	if (SELECTED_SLIDER == -1) {
		return;
	}

	/* compute the scale ration from viewport to abstract */
	float_height = (float) height();
	float_width = (float) width();
	kx = float_width/XMAX;
	ky = float_height/YMAX;

	/* We compute the absciss of the mouse in abstract coordinates */
	float value = (event->x()/kx - left_margin)/100.0;

	/* We change the position of the selected slider according to it's constraints */
	if (SELECTED_SLIDER == 0) {
		/* check */
		if (value <= sliders_positions[1]) {
			sliders_positions[0] = max(0.0f,value);
			/* We redraw the PVDualSlider */
			update();
		} else {
			sliders_positions[0] = min(1.0f, sliders_positions[1]) ;
			/* We redraw the PVDualSlider */
			update();
		}
	} else {
		/* check */
		if (value >= sliders_positions[0]) {
			sliders_positions[1] = min(value,1.0f);
			/* We redraw the PVDualSlider */
			update();
		} else {
			sliders_positions[1] = max(0.0f, sliders_positions[0]) ;
			/* We redraw the PVDualSlider */
			update();
			
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVDualSlider::mousePressEvent()
 *
 *****************************************************************************/
void PVInspector::PVDualSlider::mousePressEvent(QMouseEvent *event)
{
	last_mouse_press_position = event->posF();
	if (event->button() == Qt::LeftButton) {
		SELECTED_SLIDER = get_selected_slider_index(event->x());
		event->accept();
	} else {
		QWidget::mousePressEvent(event);
	}
}

/******************************************************************************
 *
 * PVInspector::PVDualSlider::mouseReleaseEvent()
 *
 *****************************************************************************/
void PVInspector::PVDualSlider::mouseReleaseEvent(QMouseEvent *)
{
	emit value_changed_Signal();
}

/******************************************************************************
 *
 * PVInspector::PVDualSlider::paintEvent()
 *
 *****************************************************************************/
void PVInspector::PVDualSlider::paintEvent(QPaintEvent *)
{
	/* VARIABLES */
	float float_height;
	float float_width;
	QLinearGradient linear_gradient;

	/* CODE */
	float_height = (float) height();
	float_width = (float) width();
	kx = float_width/XMAX;
	ky = float_height/YMAX;

	linear_gradient = QLinearGradient(QPointF(10, 25), QPointF(240, 25));
	linear_gradient.setColorAt(0, Qt::green);
	linear_gradient.setColorAt(0.5, QColor(255,255,0,255));
	linear_gradient.setColorAt(1, Qt::red);

	painter->begin(this);
	painter->setRenderHint(QPainter::Antialiasing);
	painter->setRenderHint(QPainter::TextAntialiasing);
	painter->setPen(palette().color(QPalette::Mid));
	//painter->setBrush(palette().color(QPalette::AlternateBase));
	painter->setBrush(linear_gradient);
	painter->drawRect(rect());

	/* step #1 : we draw the middle line */
	painter->save();
	painter->translate(left_margin*kx,float_height/2);
	painter->scale(kx,ky);
	QColor line_color = QColor(Qt::black);
	painter->setPen(line_color);
	painter->drawLine(0,0,100,0);
	painter->restore();

	/* step #2 : we draw the left slider */
	painter->save();
	painter->translate((left_margin + sliders_positions[0]*100)*kx,float_height/2);
	painter->scale(kx,ky);
	painter->setPen(Qt::black);
	painter->setBrush(Qt::black);
	painter->drawPolygon(left_slider_polygon);
	painter->restore();

	/* step #3 : we draw the right slider */
	painter->save();
	painter->translate((left_margin + sliders_positions[1]*100)*kx,float_height/2);
	painter->scale(kx,ky);
	painter->setPen(Qt::black);
	painter->setBrush(Qt::black);
	painter->drawPolygon(right_slider_polygon);
	painter->restore();

	/* we close the painter */
	painter->end();
}

/******************************************************************************
 *
 * PVInspector::PVDualSlider::sizeHint
 *
 *****************************************************************************/
QSize PVInspector::PVDualSlider::sizeHint()
{
	//return QSize(left_margin + 100 + right_margin, YMAX);
	return QSize();
}

/******************************************************************************
 *
 * PVInspector::PVDualSlider::toggle_visibility_Slot()
 *
 *****************************************************************************/
void PVInspector::PVDualSlider::toggle_visibility_Slot()
{
	setVisible(! isVisible());
}

