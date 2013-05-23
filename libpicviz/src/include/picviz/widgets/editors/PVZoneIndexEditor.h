/**
 * \file PVZoneIndexEditor.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVZONEINDEXEDITOR_H
#define PVCORE_PVZONEINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVZoneIndexType.h>

#include <picviz/PVView.h>

namespace PVWidgets {

/**
 * \class PVZoneIndexEditor
 */
class PVZoneIndexEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVZoneIndexType _zone_index READ get_zone_index WRITE set_zone_index USER true)

public:
	PVZoneIndexEditor(Picviz::PVView const& view, QWidget *parent = 0);
	virtual ~PVZoneIndexEditor();

public:
	PVCore::PVZoneIndexType get_zone_index() const;
	void set_zone_index(PVCore::PVZoneIndexType zone_index);

protected:
	Picviz::PVView const& _view;
};

}

#endif // PVCORE_PVZONEINDEXEDITOR_H
