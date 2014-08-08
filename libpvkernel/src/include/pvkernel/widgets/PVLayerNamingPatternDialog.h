/**
 * \file PVLayerNamingPatternDialog.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVWIDGETS_PVLAYERNAMINGPATTERNDIALOG_H
#define PVWIDGETS_PVLAYERNAMINGPATTERNDIALOG_H

#include <QDialog>

class QLineEdit;
class QComboBox;

namespace PVWidgets
{

/**
 * This dialog is used to get a name for a (or more) layer, given a format string
 *
 *
 *
 */
class PVLayerNamingPatternDialog : public QDialog
{
public:
	enum insert_mode {
		ON_TOP = 0,
		ABOVE_CURRENT,
		BELOW_CURRENT
	};

public:
	/**
	 * CTOR
	 *
	 * @param title the dialog's title
	 * @param text the beginning of the explanation text
	 * @param pattern the default pattern to use
	 * @param m the default mode
	 * @param parent the parent widget
	 */
	PVLayerNamingPatternDialog(const QString& title,
	                           const QString& text,
	                           const QString& pattern = "%v",
	                           insert_mode m = ON_TOP,
	                           QWidget* parent = nullptr);

	/**
	 * get the pattern to use for layers' name(s)
	 *
	 * @return the pattern to use
	 */
	QString get_name_pattern() const;

	/**
	 * get the insertion mode to use
	 *
	 * @return the insertion mode to use
	 */
	insert_mode get_insertion_mode() const;

private:
	QLineEdit* _line_edit;
	QComboBox* _combo_box;
};

}

#endif // PVWIDGETS_PVLAYERNAMINGPATTERNDIALOG_H
