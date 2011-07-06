//! \file PVInvestigationWizard.h
//! $Id: PVInvestigationWizard.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVINVESTIGATIONWIZARD_H
#define PVINVESTIGATIONWIZARD_H

#include <QDialog>

namespace PVInspector {
class PVMainWindow;

/**
 * \class PVFilterWidget
 */
class PVInvestigationWizard : public QDialog
{
	Q_OBJECT

public:
	/**
	 * Constructor.
	 */
	PVInvestigationWizard(PVMainWindow *parent);

	void create();
};
}

#endif // PVINVESTIGATIONWIZARD_H



