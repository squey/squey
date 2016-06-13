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

#include <inendi/PVMapping.h>
#include <inendi/PVView_types.h>

#include <QWidget>

namespace Inendi
{
class PVMappingProperties;
}

namespace PVWidgets
{

class PVMappingModeWidget : public QWidget
{
	Q_OBJECT
  public:
	PVMappingModeWidget(QWidget* parent = nullptr);
	PVMappingModeWidget(PVCol axis_id, Inendi::PVMapping& mapping, QWidget* parent = nullptr);

  public:
	void populate_from_type(QString const& type);
	void populate_from_mapping(PVCol axis_id, Inendi::PVMapping& mapping);
	inline void select_default() { set_mode("default"); }
	inline void clear() { _combo->clear(); }

	virtual QSize sizeHint() const;

  public:
	bool set_mode(QString const& mode);
	inline QString get_mode() const { return _combo->get_sel_userdata().toString(); }

  public:
	PVComboBox* get_combo_box() { return _combo; }
	PVCore::PVArgumentList const& get_cur_filter_params() const { return _cur_filter_params; }

  private:
	void set_filter_params_from_type_mode(QString const& type, QString const& mode);

  private Q_SLOTS:
	void change_params();

  private:
	PVComboBox* _combo;
	Inendi::PVMappingProperties* _props;
	QHash<QString, QHash<QString, PVCore::PVArgumentList>> _filter_params;
	PVCore::PVArgumentList _cur_filter_params;
	QString _cur_type;
};
}

#endif
