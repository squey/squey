/**
 * \file PVSaveFileDialog.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVSAVEFILEDIALOG_H
#define PVSAVEFILEDIALOG_H

#include <QFileDialog>

#include <picviz/general.h>

namespace PVInspector {

/**
 * \class PVSaveFileDialog
 */
class PVSaveFileDialog : public QFileDialog
{
    Q_OBJECT

public:
	PVSaveFileDialog(QWidget *parent = 0);

	QStringList getFileName();
};
}

#endif // PVSAVEFILEDIALOG_H
