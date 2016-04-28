/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef WIDGETS_PVPLOTTINGMODEWIDGET_H
#define WIDGETS_PVPLOTTINGMODEWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/widgets/PVComboBox.h>

#include <inendi/PVView_types.h>

#include <QPushButton>

namespace Inendi
{
class PVPlotting;
class PVPlottingProperties;
}

namespace PVWidgets
{

class PVPlottingModeWidget : public QWidget
{
	Q_OBJECT
  public:
	PVPlottingModeWidget(QWidget* parent = NULL) : QWidget(parent) { init(false); }
	PVPlottingModeWidget(QString const& type, QWidget* parent = NULL);
	PVPlottingModeWidget(PVCol axis_id, Inendi::PVPlotting& plotting, bool params_btn = false,
	                     QWidget* parent = NULL);
	PVPlottingModeWidget(PVCol axis_id, Inendi::PVView& view, bool params_btn = false,
	                     QWidget* parent = NULL);

  public:
	void populate_from_type(QString const& type);
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
	void init(bool params_btn);

  private slots:
	void change_params();

  private:
	PVComboBox* _combo;
	QPushButton* _params_btn;
	Inendi::PVPlottingProperties* _props;
};
}

#endif
