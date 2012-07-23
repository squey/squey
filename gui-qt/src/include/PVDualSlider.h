/**
 * \file PVDualSlider.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVDUALSLIDER_H
#define PVDUALSLIDER_H

#include <QWidget>

namespace PVInspector {
class PVMainWindow;

/**
 * \class PVDualSlider
 */
class PVDualSlider : public QWidget
{
Q_OBJECT

	PVMainWindow *main_window;
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
	float sliders_positions [2];
	QPolygonF sliders_relative_extents;
	float XMAX;
	float YMAX;
public:
	/**
	 * Constructor
	 */
	PVDualSlider(PVMainWindow *mw, QWidget *parent = 0);

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
	QSize sizeHint();

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

#endif // PVDUALSLIDER_H
