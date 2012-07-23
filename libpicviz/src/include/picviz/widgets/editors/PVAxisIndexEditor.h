/**
 * \file PVAxisIndexEditor.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVAXISINDEXEDITOR_H
#define PVCORE_PVAXISINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <picviz/PVView.h>

namespace PVWidgets {

/**
 * \class PVAxisIndexEditor
 */
class PVAxisIndexEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVAxisIndexType _axis_index READ get_axis_index WRITE set_axis_index USER true)

public:
	PVAxisIndexEditor(Picviz::PVView const& view, QWidget *parent = 0);
	virtual ~PVAxisIndexEditor();

public:
	PVCore::PVAxisIndexType get_axis_index() const;
	void set_axis_index(PVCore::PVAxisIndexType axis_index);

protected:
	Picviz::PVView const& _view;
};

}

#endif // PVCORE_PVAXISINDEXEDITOR_H
