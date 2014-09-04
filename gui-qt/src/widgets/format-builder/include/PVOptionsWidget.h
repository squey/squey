/**
 * \file PVOptionsWidget.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVOPTIONSWIDGET_H_
#define PVOPTIONSWIDGET_H_

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
	int last_line() {  return _last_line_checkbox->checkState() == Qt::Checked ? _last_line_spinbox->value() : 0; }

	void set_lines_range(int first_line, int last_line);

private slots:
	void disable_specify_last_line(int checkstate);

private:
	QSpinBox* _ignore_first_lines_spinbox;
	QLabel* _last_line_label;
	QCheckBox* _last_line_checkbox;
	QSpinBox* _last_line_spinbox;
};

}

#endif // PVOPTIONSWIDGET_H_
