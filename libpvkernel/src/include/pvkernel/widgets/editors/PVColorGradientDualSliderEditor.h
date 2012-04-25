//! \file PVDualSlider.h
//! $Id: PVDualSlider.h 2498 2011-04-25 14:27:23Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCOLORGRADIENTDUALSLIDEREDITOR_H
#define PVCOLORGRADIENTDUALSLIDEREDITOR_H

#include <QWidget>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>


namespace PVWidgets {
class PVMainWindow;

/**
 * \class PVColorGradientDualSliderEditor
 */
class PVColorGradientDualSliderEditor : public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVColorGradientDualSliderType _color_slider READ get_values WRITE set_values USER true)

	float kx;
	float ky;
	QPointF last_mouse_press_position;
	float left_margin;
	QPolygonF left_slider_polygon;
	QPainter *painter;
	int SELECTED_SLIDER;
	float right_margin;
	QPolygonF right_slider_polygon;
	float SLIDER_WIDTH;
	float sliders_positions[2];
	QPolygonF sliders_relative_extents;
	float XMAX;
	float YMAX;

public:
	/**
	 * Constructor
	 */
	PVColorGradientDualSliderEditor(QWidget *parent = 0);

	/**
	 * Gets the index of the closest sliders from position x
	 *
	 * @param x The absciss position (float) of the mouse on the slider
	 *
	 * @return The index (int) of the selected slider
	 */
	int get_selected_slider_index(float x);
	
	/**
	 * Get the actual value of the slider
	 *
	 * @param index
	 *
	 * @return The value of the given slider
	 */
	float get_slider_value(int index);
	QSize sizeHint() const;

public:
	PVCore::PVColorGradientDualSliderType get_values() const;
	void set_values(PVCore::PVColorGradientDualSliderType v);

public slots:
	void toggle_visibility_Slot();

signals:
	void value_changed_Signal();

protected:
	void mouseMoveEvent(QMouseEvent *event);
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void paintEvent(QPaintEvent *event);
};

}

#endif // PVColorGradientDualSliderEditor_H
