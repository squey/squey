/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef WIDGETS_PVPLOTTINGMODEWIDGET_H
#define WIDGETS_PVPLOTTINGMODEWIDGET_H

#include <pvbase/types.h>
#include <pvkernel/widgets/PVComboBox.h>

#include <QPushButton>

namespace Inendi
{
class PVPlotting;
}

namespace PVWidgets
{

class PVPlottingModeWidget : public QWidget
{
  public:
	PVPlottingModeWidget(QWidget* parent = nullptr);
	PVPlottingModeWidget(PVCol axis_id, Inendi::PVPlotting& plotting, QWidget* parent = nullptr);

  public:
	void populate_from_type(QString const& type, QString const& mapped);
	void populate_from_plotting(PVCol axis_id, Inendi::PVPlotting& plotting);
	inline void select_default() { set_mode("default"); }
	inline void clear() { _combo->clear(); }

	virtual QSize sizeHint() const;

  public:
	bool set_mode(QString const& mode) { return _combo->select_userdata(mode); }
	inline QString get_mode() const { return _combo->get_sel_userdata().toString(); }

  public:
	PVComboBox* get_combo_box() { return _combo; }

  private:
	PVComboBox* _combo;
};
}

#endif
