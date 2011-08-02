//! \file PVAxisIndexEditor.h
//! $Id: PVAxisIndexEditor.h 2498 2011-04-25 14:27:23Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXISINDEXEDITOR_H
#define PVCORE_PVAXISINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <picviz/PVView.h>

namespace PVInspector {

/**
 * \class PVAxisIndexEditor
 */
class PVAxisIndexEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVAxisIndexType _axis_index READ get_axis_index WRITE set_axis_index USER true)

public:
	PVAxisIndexEditor(Picviz::PVView& view, QWidget *parent = 0);
	virtual ~PVAxisIndexEditor();

public:
	PVCore::PVAxisIndexType get_axis_index() const;
	void set_axis_index(PVCore::PVAxisIndexType axis_index);

protected:
	Picviz::PVView& _view;
};

}

#endif // PVCORE_PVAXISINDEXEDITOR_H
