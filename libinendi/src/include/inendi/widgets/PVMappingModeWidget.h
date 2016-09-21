/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef WIDGETS_PVMAPPINGMODEWIDGET_H
#define WIDGETS_PVMAPPINGMODEWIDGET_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/widgets/PVComboBox.h>

#include <QWidget>

namespace Inendi
{
class PVMapped;
} // namespace Inendi

namespace PVWidgets
{

class PVMappingModeWidget : public QWidget
{
  public:
	explicit PVMappingModeWidget(QWidget* parent = nullptr);
	PVMappingModeWidget(PVCol axis_id, Inendi::PVMapped& mapping, QWidget* parent = nullptr);

  public:
	void populate_from_type(QString const& type);
	void populate_from_mapping(PVCol axis_id, Inendi::PVMapped& mapping);
	inline void select_default() { set_mode("default"); }
	inline void clear() { _combo->clear(); }

	QSize sizeHint() const override;

  public:
	bool set_mode(QString const& mode);
	inline QString get_mode() const { return _combo->get_sel_userdata().toString(); }

  public:
	PVComboBox* get_combo_box() { return _combo; }
	PVCore::PVArgumentList const& get_cur_filter_params() const { return _cur_filter_params; }

  private:
	PVComboBox* _combo;
	PVCore::PVArgumentList _cur_filter_params;
};
} // namespace PVWidgets

#endif
