/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVARGUMENTLISTWIDGET_H
#define PVARGUMENTLISTWIDGET_H

#include <QDialog>
#include <QEvent>

#include <QDataWidgetMapper>
#include <QItemEditorFactory>
#include <QTableView>
#include <QListWidget>
#include <QVariant>
#include <QHBoxLayout>

#include <pvkernel/core/PVArgument.h>

namespace PVWidgets
{

class PVArgumentListModel;

class PVArgumentListWidget : public QWidget
{
	Q_OBJECT

  public:
	PVArgumentListWidget(QWidget* parent = nullptr);
	PVArgumentListWidget(QItemEditorFactory* args_widget_factory, QWidget* parent = nullptr);
	PVArgumentListWidget(QItemEditorFactory* args_widget_factory,
	                     PVCore::PVArgumentList& args,
	                     QWidget* parent = nullptr);
	~PVArgumentListWidget() override;
	// bool eventFilter(QObject *obj, QEvent *event);
	void set_args(PVCore::PVArgumentList& args);
	void set_args_values(PVCore::PVArgumentList const& args);
	void set_widget_factory(QItemEditorFactory* factory);
	inline bool args_changed() { return _args_has_changed; }
	inline void clear_args_state() { _args_has_changed = false; }
	PVCore::PVArgumentList* get_args() { return _args; }

  public Q_SLOTS:
	inline void force_submit() { _mapper->submit(); }

  public:
	static QDialog* create_dialog_for_arguments(QItemEditorFactory* widget_factory,
	                                            PVCore::PVArgumentList& args,
	                                            QWidget* parent = nullptr);
	static bool modify_arguments_dlg(QItemEditorFactory* widget_factory,
	                                 PVCore::PVArgumentList& args,
	                                 QWidget* parent = nullptr);

  private:
	void init_widgets();

  private Q_SLOTS:
	void args_changed_Slot(const QModelIndex& a = QModelIndex(),
	                       const QModelIndex& b = QModelIndex());

  Q_SIGNALS:
	void args_changed_Signal();

	/* public Q_SLOTS: */
	/* 	void widget_clicked_Slot(); */

  protected:
	QItemEditorFactory* _args_widget_factory;
	PVCore::PVArgumentList* _args;
	QDataWidgetMapper* _mapper;

	QGridLayout* _args_layout;
	PVArgumentListModel* _args_model;

	// Standard buttons
	QPushButton* _apply_btn;
	QPushButton* _cancel_btn;
	QHBoxLayout* _btn_layout;

  private:
	bool _args_has_changed;
};
} // namespace PVWidgets

#endif
