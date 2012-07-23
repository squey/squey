/**
 * \file PVInvestigationWizard.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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



