/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVAXISINDEXFILTEREDEDITOR_H
#define PVCORE_PVAXISINDEXFILTEREDEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/PVAxisIndexType.h>

#include <inendi/PVView.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVWidgets
{

/**
 * \class PVAxisIndexFilteredEditor
 */
class PVAxisIndexFilteredEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(
	    PVCore::PVAxisIndexType _axis_index READ get_axis_index WRITE set_axis_index USER true)

  public:
	explicit PVAxisIndexFilteredEditor(Inendi::PVView const& view,
	                                   PVDisplays::PVDisplayViewAxisIf const& display_if,
	                                   QWidget* parent = 0);

  public:
	PVCore::PVAxisIndexType get_axis_index() const;
	void set_axis_index(PVCore::PVAxisIndexType axis_index);

  protected:
	Inendi::PVView const& _view;
	PVDisplays::PVDisplayViewAxisIf const& _display_if;
};
} // namespace PVWidgets

#endif // PVCORE_PVAXISINDEXFILTEREDEDITOR_H
