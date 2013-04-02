/**
 * \file PVStatsListingWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVSTATSLISTINGWIDGET_H__
#define __PVSTATSLISTINGWIDGET_H__

#include <thread>

#include <QHBoxLayout>
#include <QHeaderView>
#include <QMovie>
#include <QTableWidget>
#include <QWidget>
class QEvent;
class QLabel;
class QPixmap;
class QPushButton;
class QTableWidgetItem;

#include <pvguiqt/PVListingView.h>

namespace PVGuiQt
{

namespace __impl
{
class PVUniqueValuesCellWidget;
}

struct PVParams {
	uint32_t cached_value;
	bool auto_refresh;
};

class PVStatsListingWidget : public QWidget
{
	Q_OBJECT

public:
	typedef std::unordered_map<uint32_t, PVParams> param_t;

public:
	PVStatsListingWidget(PVListingView* listing_view);

public:
	param_t& get_params() { return _params; }
	void set_refresh_buttons_enabled(bool loading);

protected:
	bool eventFilter(QObject *obj, QEvent *event);

private:
	void init_plugins();

	template <typename T>
	void init_plugin(QString header_text)
	{
		int row = _stats_panel->rowCount();
		_stats_panel->insertRow(row);
		for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
			create_item<T>(row, col);
		}

		QStringList vertical_headers;
		_stats_panel->setVerticalHeaderItem(row, new QTableWidgetItem(header_text));
		//_stats_panel->verticalHeaderItem(row)->setToolTip("Refresh all");
	}

	template <typename T>
	void create_item(int row, int col)
	{
		_stats_panel->insertColumn(col);
		QTableWidgetItem* item = new QTableWidgetItem();
		_stats_panel->setItem(row, col, item);
		_stats_panel->setCellWidget(row, col, new T(_stats_panel, _listing_view->lib_view(), item));
	}

private slots:
	void toggle_stats_panel_visibility();
	void update_header_width(int column, int old_width, int new_width);
	void update_scrollbar_position();
	void refresh();
	void resize_panel();
	void selection_changed();
	void axes_comb_changed();

public:
	static const QColor INVALID_COLOR;

private:
	PVListingView* _listing_view;
	QTableWidget* _stats_panel;

	param_t _params;

	int _old_maximum_width;
	bool _maxed = false;
};

namespace __impl
{

class PVCellWidgetBase : public QWidget
{
	Q_OBJECT;

public:
	PVCellWidgetBase(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item);
	virtual ~PVCellWidgetBase() {}

public:
	static QMovie* get_movie(); // Singleton to share the animation among all the widgets in order to keep them synchronized
	virtual void set_loading(bool loading);
	void set_refresh_button_enabled(bool loading);
	static void cancel_thread();

public slots:
	void refresh(bool use_cache = false);
	void auto_refresh();

protected slots:
	void refreshed(int value, bool valid);

private slots:
	virtual void vertical_header_clicked(int index);
	void toggle_auto_refresh();

signals:
	void refresh_impl_finished(int value, bool valid);

protected:
	virtual void refresh_impl() = 0;
	typename PVStatsListingWidget::param_t& get_params();
	PVGuiQt::PVStatsListingWidget* get_panel();
	void set_valid(uint32_t value, bool autorefresh);
	void set_invalid();

protected:
	QTableWidget* _table;
	Picviz::PVView const& _view;
	QTableWidgetItem* _item;

	bool _valid = false;

	QHBoxLayout* _main_layout;
	QPushButton* _refresh_icon;
	QPushButton* _autorefresh_icon;
	QLabel* _loading_label;
	static QMovie* _loading_movie;
	const QPixmap _refresh_pixmap;
	const QPixmap _autorefresh_on_pixmap;
	const QPixmap _autorefresh_off_pixmap;

	QLabel* _text;

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
	const QPixmap _unique_values_pixmap;
};

}

}

#endif // __PVSTATSLISTINGWIDGET_H__
