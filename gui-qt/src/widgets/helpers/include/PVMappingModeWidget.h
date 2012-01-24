#ifndef WIDGETS_PVMAPPINGMODEWIDGET_H
#define WIDGETS_PVMAPPINGMODEWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMappingProperties.h>
#include <picviz/PVView_types.h>
#include <PVComboBox.h>

#include <QPushButton>
#include <QWidget>

namespace PVInspector {

namespace PVWidgetsHelpers {

class PVMappingModeWidget: public QWidget
{
	Q_OBJECT
public:
	PVMappingModeWidget(QWidget* parent = NULL):
		QWidget(parent)
	{ init(false); }
	PVMappingModeWidget(QString const& type, QWidget* parent = NULL);
	PVMappingModeWidget(PVCol axis_id, Picviz::PVMapping& mapping, bool params_btn = false, QWidget* parent = NULL);
	PVMappingModeWidget(PVCol axis_id, Picviz::PVView& view, bool params_btn = false, QWidget* parent = NULL);

public:
	void populate_from_type(QString const& type);
	void populate_from_mapping(PVCol axis_id, Picviz::PVMapping& mapping);
	inline void select_default() { set_mode("default"); }
	inline void clear() { _combo->clear(); }

	virtual QSize sizeHint() const;

public:
	bool set_mode(QString const& mode) { return _combo->select_userdata(mode); }
	inline QString get_mode() const { return _combo->get_sel_userdata().toString(); }

private:
	void init(bool params_btn);

private slots:
	void change_params();

private:
	PVComboBox* _combo;
	QPushButton* _params_btn;
	Picviz::PVMappingProperties* _props;
};

}

}


#endif
