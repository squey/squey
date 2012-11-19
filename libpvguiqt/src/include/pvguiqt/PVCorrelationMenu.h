/**
 * \file PVCorrelationMenu.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVCORRELATIONMENU_H__
#define __PVGUIQT_PVCORRELATIONMENU_H__

class QWidget;
#include <QMenu>

#include <picviz/PVAD2GView_types.h>

namespace Picviz
{
class PVRoot;
}

namespace PVGuiQt
{

/**
 * \class PVCorrelationMenu
 *
 * \note The menu managing PVRoot correlations.
 */
class PVCorrelationMenu : public QMenu
{
	Q_OBJECT;

public:
	PVCorrelationMenu(Picviz::PVRoot* root, QWidget* parent = 0);

public:
	/*! \brief Add a new correlation (based on its name) to the menu and show the editor.
	 */
	void add_correlation(const QString & name);

	/*! \brief Add an existing correlation to the menu.
	 */
	void add_correlation(Picviz::PVAD2GView* correlation);

	/*! \brief Load all existing correlations to the menu.
	 */
	void load_correlations();

private slots:
	/*! \brief Prompt the user for the correlation name.
	 */
	void create_new_correlation();

	/*! \brief Show the editor for a correlation based on the clicked menu.
	 */
	void show_correlation();

	/*! \brief Show the editor for a given correlation.
	 */
	void show_correlation(Picviz::PVAD2GView* correlation);

	/*! \brief Delete a correlation based on the clicked menu.
	 */
	void delete_correlation();

	/*! \brief Enable or disable all the correlations.
	 */
	void enable_correlations(bool enable);

private:
	Picviz::PVRoot* _root;

	QAction* _separator_create_correlation;
	QAction* _action_create_correlation;
};

}

#endif // __PVGUIQT_PVCORRELATIONMENU_H__
