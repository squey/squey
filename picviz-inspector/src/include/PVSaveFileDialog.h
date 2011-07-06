//! \file PVSaveFileDialog.h
//! $Id: PVSaveFileDialog.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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
