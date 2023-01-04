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

#ifndef __PVSTATSLISTINGWIDGET_H__
#define __PVSTATSLISTINGWIDGET_H__

#include <sigc++/sigc++.h>

#include <thread>
#include <unordered_map>

#include <QApplication>
#include <QClipboard>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMovie>
#include <QTableWidget>
#include <QWidget>
#include <QLabel>
#include <QMouseEvent>
class QEvent;
class QMenu;
class QPixmap;
class QPushButton;
class QTableWidgetItem;
class QDialog;

#include <pvguiqt/PVListingView.h>

namespace PVGuiQt
{

namespace __impl
{
class PVCellWidgetBase;
class PVUniqueValuesCellWidget;
class PVSumCellWidget;
} // namespace __impl

class PVStatsListingWidget : public QWidget, public sigc::trackable
{
	Q_OBJECT
	friend class __impl::PVCellWidgetBase;

  public:
	struct PVParams {
		QString cached_value;
		bool auto_refresh;
	};

  public:
	typedef std::unordered_map<uint32_t, std::unordered_map<uint32_t, PVParams>> param_t;

  public:
	explicit PVStatsListingWidget(PVListingView* listing_view);

  private:
	param_t& get_params() { return _params; }
	void set_refresh_buttons_enabled(bool loading);

  private Q_SLOTS:
	void plugin_visibility_toggled(bool checked);
	void resize_listing_column_if_needed(int col);

  private:
	void init_plugins();

	template <typename T>
	int init_plugin(QString header_text, bool visible = false)
	{
		int row = _stats_panel->rowCount();
		_stats_panel->insertRow(row);
		for (PVCombCol col(0); col < _listing_view->horizontalHeader()->count(); col++) {
			create_item<T>(row, col);
		}

		QStringList vertical_headers;
		_stats_panel->setVerticalHeaderItem(row, new QTableWidgetItem(header_text));
		if (!visible) {
			_stats_panel->hideRow(row);
		}

		//_stats_panel->verticalHeaderItem(row)->setToolTip("Refresh all");
		return row;
	}

	template <typename T>
	void create_item(int row, int col)
	{
		QTableWidgetItem* item = new QTableWidgetItem();
		_stats_panel->setItem(row, col, item);
		T* widget = new T(_stats_panel, _listing_view->lib_view(), item);
		connect(widget, SIGNAL(cell_refreshed(int)), this,
		        SLOT(resize_listing_column_if_needed(int)));
		_stats_panel->setCellWidget(row, col, widget);
	}

	void create_vhead_ctxt_menu();

  public:
	void sync_vertical_headers();

  private Q_SLOTS:
	void toggle_stats_panel_visibility();
	void update_header_width(int column, int old_width, int new_width);
	void update_scrollbar_position();
	void refresh();
	void resize_panel();
	void selection_changed();
	void axes_comb_changed();
	void vertical_header_section_clicked(const QPoint&);

  public:
	static const QColor INVALID_COLOR;

  private:
	PVListingView* _listing_view;
	QTableWidget* _stats_panel;

	param_t _params;

	int _old_maximum_width;
	bool _maxed = false;
	QMenu* _vhead_ctxt_menu;

	int _row_distinct;
	int _row_sum;
	int _row_min;
	int _row_max;
	int _row_avg;
};

namespace __impl
{

class PVVerticalHeaderView : public QHeaderView
{
	Q_OBJECT

  public:
	explicit PVVerticalHeaderView(PVStatsListingWidget* parent);
};

class PVLoadingLabel : public QLabel
{
	Q_OBJECT

  public:
	explicit PVLoadingLabel(QWidget* parent) : QLabel(parent) {}

  protected:
	void mousePressEvent(QMouseEvent* ev) override
	{
		if (ev->button() == Qt::LeftButton) {
			Q_EMIT clicked();
		}
	}

  Q_SIGNALS:
	void clicked();
};

class PVCellWidgetBase : public QWidget
{
	Q_OBJECT;

  public:
	PVCellWidgetBase(QTableWidget* table, Inendi::PVView& view, QTableWidgetItem* item);
	~PVCellWidgetBase() override {}

  public:
	inline int get_widget_cell_row() { return _table->row(_item); }
	inline PVCombCol get_widget_cell_col() { return PVCombCol(_table->column(_item)); }

	inline int get_real_axis_row() { return _table->row(_item); }
	inline PVCol get_real_axis_col() { return _view.get_nraw_axis_index(get_widget_cell_col()); }

	static QMovie* get_movie(); // Singleton to share the animation among all the widgets in order
	                            // to keep them synchronized
	virtual void set_loading(bool loading);
	void set_refresh_button_enabled(bool loading);
	inline int minimum_size()
	{
		return _main_layout->minimumSize().width() -
		       QApplication::style()->pixelMetric(QStyle::PM_ScrollBarExtent);
	}

  public Q_SLOTS:
	void refresh(bool use_cache = false);
	void auto_refresh();
	static void cancel_thread();
	virtual void update_type_capabilities(){};

  protected Q_SLOTS:
	void refreshed(QString value);
	void context_menu_requested(const QPoint&);

  private Q_SLOTS:
	virtual void vertical_header_clicked(int index);
	void toggle_auto_refresh();
	void copy_to_clipboard();

  Q_SIGNALS:
	void refresh_impl_finished(QString value);
	void cell_refreshed(int col);

  protected:
	virtual void refresh_impl() = 0;
	typename PVStatsListingWidget::PVParams& get_params();
	PVGuiQt::PVStatsListingWidget* get_panel();
	void set_valid(const QString& value, bool autorefresh);
	void set_invalid();

  protected:
	QTableWidget* _table;
	Inendi::PVView& _view;
	QTableWidgetItem* _item;

	bool _valid = false;

	QHBoxLayout* _main_layout;
	QHBoxLayout* _customizable_layout;
	QPushButton* _refresh_icon;
	QPushButton* _autorefresh_icon;
	PVLoadingLabel* _loading_label;
	static QMovie* _loading_movie;
	const QPixmap _refresh_pixmap;
	const QPixmap _autorefresh_on_pixmap;
	const QPixmap _autorefresh_off_pixmap;

	QLabel* _text;
	QMenu* _ctxt_menu;

	static std::thread _thread;
	static tbb::task_group_context* _ctxt;
	static bool _thread_running;

	bool _is_summable = false;
};

/**
 * Widget for cell at the bottom of the listing to display unique values.
 */
class PVUniqueValuesCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

  public:
	PVUniqueValuesCellWidget(QTableWidget* table, Inendi::PVView& view, QTableWidgetItem* item);

  public Q_SLOTS:
	void refresh_impl() override;

  private Q_SLOTS:
	void show_unique_values_dlg();
	void unique_values_dlg_closed();

  private:
	QDialog* _dialog = nullptr;
};

class PVSumCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

  public:
	PVSumCellWidget(QTableWidget* table, Inendi::PVView& view, QTableWidgetItem* item)
	    : PVCellWidgetBase(table, view, item)
	{
		update_type_capabilities();
	}

  public Q_SLOTS:
	void refresh_impl() override;
	void update_type_capabilities() override;
};

class PVMinCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

  public:
	PVMinCellWidget(QTableWidget* table, Inendi::PVView& view, QTableWidgetItem* item)
	    : PVCellWidgetBase(table, view, item)
	{
	}

  public Q_SLOTS:
	void refresh_impl() override;
};

class PVMaxCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

  public:
	PVMaxCellWidget(QTableWidget* table, Inendi::PVView& view, QTableWidgetItem* item)
	    : PVCellWidgetBase(table, view, item)
	{
	}

  public Q_SLOTS:
	void refresh_impl() override;
};

class PVAverageCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

  public:
	PVAverageCellWidget(QTableWidget* table, Inendi::PVView& view, QTableWidgetItem* item)
	    : PVCellWidgetBase(table, view, item)
	{
		update_type_capabilities();
	}

  public Q_SLOTS:
	void refresh_impl() override;
	void update_type_capabilities() override;
};
} // namespace __impl
} // namespace PVGuiQt

#endif // __PVSTATSLISTINGWIDGET_H__
