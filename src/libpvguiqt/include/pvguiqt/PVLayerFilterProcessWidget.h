/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVGUIQT_PVLAYERFILTERPROCESSWIDGET_H
#define PVGUIQT_PVLAYERFILTERPROCESSWIDGET_H

#include <QSplitter>

#include <squey/PVLayerFilter.h>
#include <pvkernel/widgets/PVPresetsWidget.h>

#include <condition_variable>
#include <mutex>

namespace PVWidgets
{
class PVArgumentListWidget;
} // namespace PVWidgets

namespace PVGuiQt
{

class PVLayerFilterProcessWidget : public QDialog
{
	Q_OBJECT

  public:
	PVLayerFilterProcessWidget(Squey::PVView* view,
	                           PVCore::PVArgumentList& args,
	                           Squey::PVLayerFilter_p filter_p,
	                           QWidget* parent = nullptr);
	~PVLayerFilterProcessWidget() override;

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
	static void process_layer_filter(Squey::PVLayerFilter* filter,
	                                 Squey::PVLayer const* layer,
	                                 Squey::PVLayer* out_layer);

  Q_SIGNALS:
	void layer_filter_error(const Squey::PVLayerFilter_p& filter);

  private Q_SLOTS:
	void show_layer_filter_error(const Squey::PVLayerFilter_p& filter);

  protected:
	Squey::PVView* _view;
	Squey::PVLayerFilter_p _filter_p;
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
} // namespace PVGuiQt

#endif
