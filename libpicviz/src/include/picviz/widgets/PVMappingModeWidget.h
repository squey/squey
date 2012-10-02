/**
 * \file PVMappingModeWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef WIDGETS_PVMAPPINGMODEWIDGET_H
#define WIDGETS_PVMAPPINGMODEWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/widgets/PVComboBox.h>

#include <picviz/PVMapping.h>
#include <picviz/PVView_types.h>

#include <QPushButton>
#include <QWidget>

namespace Picviz {
class PVMappingProperties;
}

namespace PVWidgets {

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
	bool set_mode(QString const& mode);
	inline QString get_mode() const { return _combo->get_sel_userdata().toString(); }

private:
	void init(bool params_btn);
	void set_filter_params_from_type_mode(QString const& type, QString const& mode);

private slots:
	void change_params();

private:
	PVComboBox* _combo;
	QPushButton* _params_btn;
	Picviz::PVMappingProperties* _props;
	QHash<QString, QHash<QString, PVCore::PVArgumentList> > _filter_params;
	PVCore::PVArgumentList _cur_filter_params;
	QString _cur_type;
};

}

#endif
