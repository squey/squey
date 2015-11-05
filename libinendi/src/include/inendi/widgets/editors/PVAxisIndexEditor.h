/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVAXISINDEXEDITOR_H
#define PVCORE_PVAXISINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <inendi/PVView.h>

namespace PVWidgets {

/**
 * \class PVAxisIndexEditor
 */
class PVAxisIndexEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVAxisIndexType _axis_index READ get_axis_index WRITE set_axis_index USER true)

public:
	PVAxisIndexEditor(Inendi::PVView const& view, QWidget *parent = 0);
	virtual ~PVAxisIndexEditor();

public:
	PVCore::PVAxisIndexType get_axis_index() const;
	void set_axis_index(PVCore::PVAxisIndexType axis_index);

protected:
	Inendi::PVView const& _view;
};

}

#endif // PVCORE_PVAXISINDEXEDITOR_H
