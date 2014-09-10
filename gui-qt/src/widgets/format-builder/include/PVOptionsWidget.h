/**
 * \file PVOptionsWidget.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVOPTIONSWIDGET_H_
#define PVOPTIONSWIDGET_H_

#include <pvguiqt/PVLocalizedSpinBox.h>

#include <QWidget>
#include <QSpinBox>
#include <QCheckBox>
class QLabel;

namespace PVInspector {

class PVOptionsWidget: public QWidget
{
	Q_OBJECT

public:
	PVOptionsWidget(QWidget* parent = nullptr);

public:
	int first_line() { return _ignore_first_lines_spinbox->value(); }
	int line_count() {  return _line_count_checkbox->checkState() == Qt::Checked ? _line_count_spinbox->value() : 0; }

	void set_lines_range(int first_line, int line_count);

private slots:
	void disable_specify_line_count(int checkstate);

private:
	PVGuiQt::PVLocalizedSpinBox* _ignore_first_lines_spinbox;
	QLabel*                      _line_count_label;
	QCheckBox*                   _line_count_checkbox;
	PVGuiQt::PVLocalizedSpinBox* _line_count_spinbox;
};

}

#endif // PVOPTIONSWIDGET_H_
