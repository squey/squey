/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H
#define PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QAction;
class QWidget;
class QListWidget;
class QLineEdit;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter
{

class PVFieldSplitterKeyValueParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT

  public:
	PVFieldSplitterKeyValueParamWidget();

  public:
	QAction* get_action_menu(QWidget* parent) override;
	QWidget* get_param_widget() override;

  private Q_SLOTS:
	void update_params();
	void add_new_key();
	void del_keys();
	void move_key_down();
	void move_key_up();
	void update_children_count();
	void copy_keys();
	void paste_keys();
	void selection_has_changed();

  private:
	void add_new_keys(QStringList& keys);

  private:
	QWidget* _param_widget;

	PVWidgets::QKeySequenceWidget* _quote_char;
	QLineEdit* _separator_char_lineedit;
	QLineEdit* _affectation_operator_lineedit;
	QPushButton* _del_button;
	QPushButton* _up_button;
	QPushButton* _down_button;
	QPushButton* _copy_button;

	QListWidget* _keys_list;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterKeyValueParamWidget)
};
}

#endif // PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H
