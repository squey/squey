//! \file PVAxisIndexCheckBoxEditor.h
//! $Id: PVAxisIndexCheckBoxEditor.h 2498 2011-04-25 14:27:23Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXISINDEXCHECKBOXEDITOR_H
#define PVCORE_PVAXISINDEXBHECKBOXEDITOR_H

#include <QComboBox>
#include <QCheckBox>
#include <QString>
#include <QWidget>

#include <pvcore/general.h>
#include <pvcore/PVAxisIndexCheckBoxType.h>

#include <picviz/PVView.h>

/* #include <PVCheckableComboBox.h> */

namespace PVInspector {

/**
 * \class PVAxisIndexCheckBoxEditor
 */
/* class PVAxisIndexCheckBoxEditor : public PVCheckableComboBox */
/* class PVAxisIndexCheckBoxEditor : public QComboBox */
class PVAxisIndexCheckBoxEditor : public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVAxisIndexCheckBoxType _axis_index READ get_axis_index WRITE set_axis_index USER true)

private:
	QComboBox *combobox;
	QCheckBox *checkbox;
	bool _checked;
	int _current_index;

public:
	PVAxisIndexCheckBoxEditor(Picviz::PVView& view, QWidget *parent = 0);
	virtual ~PVAxisIndexCheckBoxEditor();

	PVCore::PVAxisIndexCheckBoxType get_axis_index() const;
	void set_axis_index(PVCore::PVAxisIndexCheckBoxType axis_index);

protected:
	Picviz::PVView& _view;
};

}

#endif // PVCORE_PVAXISINDEXCHECKBOXEDITOR_H
