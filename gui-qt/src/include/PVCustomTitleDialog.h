/**
 * \file PVCustomTitleDialog.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCUSTOMTITLEDIALOG_H
#define PVCUSTOMTITLEDIALOG_H

#include <QDialog>

namespace PVInspector {

class PVCustomTitleDialog: public QDialog
{
public:
	PVCustomTitleDialog(QWidget* parent = 0, Qt::WindowFlags f = 0);
protected:
	void paintEvent(QPaintEvent* event);
};

}

#endif
