//! \file PVOpenFileDialog.h
//! $Id: PVOpenFileDialog.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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
