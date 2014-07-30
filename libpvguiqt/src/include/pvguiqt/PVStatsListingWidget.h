/**
 * \file PVStatsListingWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVSTATSLISTINGWIDGET_H__
#define __PVSTATSLISTINGWIDGET_H__

#include <thread>

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

#include <pvguiqt/PVListingView.h>

namespace PVGuiQt
{

namespace __impl
{
class PVCellWidgetBase;
class PVUniqueValuesCellWidget;
class PVSumCellWidget;
}

class PVStatsListingWidget : public QWidget
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
	PVStatsListingWidget(PVListingView* listing_view);

private:
	param_t& get_params()
	{
		return _params;
	}
	void set_refresh_buttons_enabled(bool loading);

protected:
	bool eventFilter(QObject *obj, QEvent *event);

private slots:
	void plugin_visibility_toggled(bool checked);
	void resize_listing_column_if_needed(int col);

private:
	void init_plugins();

	template <typename T>
	void init_plugin(QString header_text, bool visible = false)
	{
		int row = _stats_panel->rowCount();
		_stats_panel->insertRow(row);
		for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
			create_item<T>(row, col);
		}

		QStringList vertical_headers;
		_stats_panel->setVerticalHeaderItem(row, new QTableWidgetItem(header_text));
		if (!visible) {
			_stats_panel->hideRow(row);
		}

		//_stats_panel->verticalHeaderItem(row)->setToolTip("Refresh all");
	}

	template <typename T>
	void create_item(int row, int col)
	{
		QTableWidgetItem* item = new QTableWidgetItem();
		_stats_panel->setItem(row, col, item);
		T* widget = new T(_stats_panel, _listing_view->lib_view(), item);
		connect(widget, SIGNAL(cell_refreshed(int)), this, SLOT(resize_listing_column_if_needed(int)));
		_stats_panel->setCellWidget(row, col, widget);
	}

	void create_vhead_ctxt_menu();

private slots:
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
};

namespace __impl
{

class PVVerticalHeaderView : public QHeaderView
{
	Q_OBJECT

public:
	PVVerticalHeaderView(PVStatsListingWidget* parent);
};

class PVLoadingLabel : public QLabel
{
	Q_OBJECT

public:
	PVLoadingLabel(QWidget* parent) : QLabel(parent) {}

protected:
	virtual void mousePressEvent(QMouseEvent * ev) override { if (ev->button()==Qt::LeftButton) { emit clicked(); } }

signals:
	void clicked();
};

class PVCellWidgetBase : public QWidget
{
	Q_OBJECT;

public:
	PVCellWidgetBase(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item);
	virtual ~PVCellWidgetBase() {}

public:
	inline int get_widget_cell_row() { return _table->row(_item); }
	inline int get_widget_cell_col() { return _table->column(_item); }

	inline int get_real_axis_row() { return _table->row(_item); }
	inline int get_real_axis_col() { return _view.get_real_axis_index(_table->column(_item)); }

	static QMovie* get_movie(); // Singleton to share the animation among all the widgets in order to keep them synchronized
	virtual void set_loading(bool loading);
	void set_refresh_button_enabled(bool loading);
	inline int minimum_size() { return _main_layout->minimumSize().width() - QApplication::style()->pixelMetric(QStyle::PM_ScrollBarExtent); }

public slots:
	void refresh(bool use_cache = false);
	void auto_refresh();
	static void cancel_thread();

protected slots:
	void refreshed(QString value, bool valid);
	void context_menu_requested(const QPoint&);

private slots:
	virtual void vertical_header_clicked(int index);
	void toggle_auto_refresh();
	void copy_to_clipboard();

signals:
	void refresh_impl_finished(QString value, bool valid);
	void cell_refreshed(int col);

protected:
	virtual void refresh_impl() = 0;
	typename PVStatsListingWidget::PVParams& get_params();
	PVGuiQt::PVStatsListingWidget* get_panel();
	void set_valid(const QString& value, bool autorefresh);
	void set_invalid();

protected:
	QTableWidget* _table;
	Picviz::PVView const& _view;
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
};

class PVUniqueValuesCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

public:
	PVUniqueValuesCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item);

public slots:
	virtual void refresh_impl() override;

private slots:
	void show_unique_values_dlg();

private:
	uint32_t _unique_values_number;
	QPushButton* _unique_values_dlg_icon;
	QPixmap _unique_values_pixmap;
};

class PVSumCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

public:
	PVSumCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) : PVCellWidgetBase(table, view, item) {}

public slots:
	virtual void refresh_impl() override;
};

class PVMinCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

public:
	PVMinCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) : PVCellWidgetBase(table, view, item) {}

public slots:
	virtual void refresh_impl() override;
};

class PVMaxCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

public:
	PVMaxCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) : PVCellWidgetBase(table, view, item) {}

public slots:
	virtual void refresh_impl() override;
};

class PVAverageCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

public:
	PVAverageCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) : PVCellWidgetBase(table, view, item) {}

public slots:
	virtual void refresh_impl() override;
};

}

}

#endif // __PVSTATSLISTINGWIDGET_H__
