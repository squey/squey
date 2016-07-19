/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVLAYERFILTERPROCESSWIDGET_H
#define PVGUIQT_PVLAYERFILTERPROCESSWIDGET_H

#include <QSplitter>

#include <inendi/PVLayerFilter.h>
#include <pvkernel/widgets/PVPresetsWidget.h>

#include <condition_variable>
#include <mutex>

namespace PVWidgets
{
class PVArgumentListWidget;
}

namespace PVGuiQt
{

class PVLayerFilterProcessWidget : public QDialog
{
	Q_OBJECT

  public:
	PVLayerFilterProcessWidget(Inendi::PVView* view,
	                           PVCore::PVArgumentList& args,
	                           Inendi::PVLayerFilter_p filter_p,
	                           QWidget* parent = nullptr);
	virtual ~PVLayerFilterProcessWidget();

	void change_args(PVCore::PVArgumentList const& args);

  public Q_SLOTS:
	void save_Slot();
	void preview_Slot();
	void reset_Slot();
	void load_preset_Slot(const QString& preset);
	void add_preset_Slot(const QString& preset);
	void save_preset_Slot(const QString& preset);
	void remove_preset_Slot(const QString& preset);
	void rename_preset_Slot(const QString& old_preset, const QString& new_preset);

  protected:
	void reject() override;

	void create_btns();
	void set_btns_layout();
	void connect_btns();

	/**
	 * Apply filter computation on post_filter_layer and refresh view.
	 */
	bool process();

  private:
	static void process_layer_filter(Inendi::PVLayerFilter* filter,
	                                 Inendi::PVLayer const* layer,
	                                 Inendi::PVLayer* out_layer);

  Q_SIGNALS:
	void layer_filter_error(const Inendi::PVLayerFilter_p& filter);

  private Q_SLOTS:
	void show_layer_filter_error(const Inendi::PVLayerFilter_p& filter);

  protected:
	Inendi::PVView* _view;
	Inendi::PVLayerFilter_p _filter_p;
	PVWidgets::PVPresetsWidget* _presets_widget;
	QSplitter* _splitter;
	QHBoxLayout* _presets_layout;
	QComboBox* _presets_combo;
	QPushButton* _cancel_btn;
	QPushButton* _reset_btn;
	QPushButton* _help_btn;
	QPushButton* _preview_btn;
	QPushButton* _apply_btn;
	PVCore::PVArgumentList _args_org;
	PVWidgets::PVArgumentListWidget* _args_widget;
	QHBoxLayout* _btn_layout;

  private:
	bool _has_apply;
	std::mutex _blocking_msg; //!< Mutex to have blocking message during thread execution.
	std::condition_variable
	    _cv; //!< Condition variable to sync thread and message during thread execution.
};
}

#endif
