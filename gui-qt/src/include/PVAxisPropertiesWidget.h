/**
 * \file PVAxisPropertiesWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVAXISPROPERTIESWIDGET_H
#define PVAXISPROPERTIESWIDGET_H

#include <QComboBox>
#include <QDialog>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QStringList>
#include <QDialogButtonBox>

#include <picviz/PVView_types.h>


namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVAxisPropertiesWidget
 */
class PVAxisPropertiesWidget : public QDialog
{
Q_OBJECT

private:
	QComboBox        *axes_list;
	QStringList       axes_names_list;
	QLineEdit        *axis_name;
	QLabel           *axis_name_label;
	QGridLayout      *grid_layout;
	QVBoxLayout      *main_layout;
	QDialogButtonBox *box_buttons;
	PVTabSplitter    *tab;
	PVMainWindow* main_window;
	Picviz::PVView_p _view;

public:
	/**
	 * Constructor
	 */
	PVAxisPropertiesWidget(Picviz::PVView_p view, PVTabSplitter* tab_, PVMainWindow *mw);

	/**
	 * Destructor
	 */
	~PVAxisPropertiesWidget();

protected:
	void create();

public slots:

	/**
	* update the name of the axis
	*/
	void apply_slot();
	
	/**
	* Cancel the modification, it reload the name.
	*/
	void cancel_slot();
	
	/**
	 * Refreshes the whole widget (mainly the Combo, and then the elements attached
	 *  to the current Axis selected in the Combo
	 */
	void refresh_widget();
	void refresh_widget(int index);
	
};
}

#endif // PVAXISPROPERTIESWIDGET_H
