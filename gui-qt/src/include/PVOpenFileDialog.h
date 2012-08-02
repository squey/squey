/**
 * \file PVOpenFileDialog.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVOPENFILEDIALOG_H
#define PVOPENFILEDIALOG_H

#include <QFileDialog>

#include <picviz/general.h>

namespace PVInspector {

/**
 * \class PVOpenFileDialog
 */
class PVOpenFileDialog : public QFileDialog
{
    Q_OBJECT

public:
	PVOpenFileDialog(QWidget *parent = 0);

	QStringList getFileName();
};
}

#endif // PVOPENFILEDIALOG_H
