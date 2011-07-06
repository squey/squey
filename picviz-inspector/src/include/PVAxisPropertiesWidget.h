//! \file PVAxisPropertiesWidget.h
//! $Id: PVAxisPropertiesWidget.h 3236 2011-07-04 12:40:17Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVAXISPROPERTIESWIDGET_H
#define PVAXISPROPERTIESWIDGET_H

#include <QComboBox>
#include <QDialog>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QStringList>



namespace PVInspector {
class PVMainWindow;

/**
 * \class PVAxisPropertiesWidget
 */
class PVAxisPropertiesWidget : public QDialog
{
Q_OBJECT

private:
	PVMainWindow *main_window;

	QPushButton *apply_button;
	QComboBox   *axes_list;
	QStringList  axes_names_list;
	QLineEdit   *axis_name;
	QLabel      *axis_name_label;
	QPushButton *cancel_button;
	QPushButton *done_button;
	QGridLayout *main_layout;
	

public:
	/**
	 * Constructor
	 */
	PVAxisPropertiesWidget(PVMainWindow *mw);

	/**
	 * Destructor
	 */
	~PVAxisPropertiesWidget();

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
