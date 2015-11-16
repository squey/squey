/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVORIGINALAXISINDEXEDITOR_H
#define PVCORE_PVORIGINALAXISINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <inendi/PVView.h>

namespace PVWidgets {

/**
 * \class PVOriginalAxisIndexEditor
 */
class PVOriginalAxisIndexEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVOriginalAxisIndexType _axis_index READ get_axis_index WRITE set_axis_index USER true)

public:
	PVOriginalAxisIndexEditor(Inendi::PVView const& view, QWidget *parent = 0);
	virtual ~PVOriginalAxisIndexEditor();

public:
	PVCore::PVOriginalAxisIndexType get_axis_index() const;
	void set_axis_index(PVCore::PVOriginalAxisIndexType axis_index);

protected:
	Inendi::PVView const& _view;
};

}

#endif // PVCORE_PVAXISINDEXEDITOR_H
