#ifndef WIDGETS_PVPLOTTINGMODEWIDGET_H
#define WIDGETS_PVPLOTTINGMODEWIDGET_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVView_types.h>
#include <PVComboBox.h>

namespace PVInspector {

namespace PVWidgetsHelpers {

class PVPlottingModeWidget: public PVComboBox
{
public:
	PVPlottingModeWidget(QWidget* parent = NULL):
		PVComboBox(parent)
	{ }
	PVPlottingModeWidget(QString const& type, QWidget* parent = NULL);
	PVPlottingModeWidget(PVCol axis_id, Picviz::PVPlotting const& plotting, QWidget* parent = NULL);
	PVPlottingModeWidget(PVCol axis_id, Picviz::PVView const& view, QWidget* parent = NULL);

public:
	void populate_from_type(QString const& type);
	void populate_from_plotting(PVCol axis_id, Picviz::PVPlotting const& plotting);
	inline void select_default() { select("default"); }

public:
	bool set_mode(QString const& mode) { return select_userdata(mode); }
	inline QString get_mode() const { return get_sel_userdata().toString(); }

};

}

}

#endif
