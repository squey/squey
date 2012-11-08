/**
 * \file PVAxisIndexEditor.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVORIGINALAXISINDEXEDITOR_H
#define PVCORE_PVORIGINALAXISINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <picviz/PVView.h>

namespace PVWidgets {

/**
 * \class PVOriginalAxisIndexEditor
 */
class PVOriginalAxisIndexEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVOriginalAxisIndexType _axis_index READ get_axis_index WRITE set_axis_index USER true)

public:
	PVOriginalAxisIndexEditor(Picviz::PVView const& view, QWidget *parent = 0);
	virtual ~PVOriginalAxisIndexEditor();

public:
	PVCore::PVOriginalAxisIndexType get_axis_index() const;
	void set_axis_index(PVCore::PVOriginalAxisIndexType axis_index);

protected:
	Picviz::PVView const& _view;
};

}

#endif // PVCORE_PVAXISINDEXEDITOR_H
