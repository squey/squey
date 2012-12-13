/**
 * \file PVStatsListingWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVSTATSLISTINGWIDGET_H__
#define __PVSTATSLISTINGWIDGET_H__

#include <QWidget>
#include <QTableWidget>
#include <QHeaderView>
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

class PVStatsListingWidget : public QWidget
{
	Q_OBJECT
	friend class __impl::PVUniqueValuesCellWidget;

public:
	PVStatsListingWidget(PVListingView* listing_view);

protected:
	bool eventFilter(QObject *obj, QEvent *event);

private:
	template <typename T>
	void init_plugin(QString header_text)
	{
		int row = _stats_panel->rowCount();
		_stats_panel->insertRow(row);
		for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
			_stats_panel->insertColumn(col);
			QTableWidgetItem* item = new QTableWidgetItem();
			_stats_panel->setItem(row, col, item);
			_stats_panel->setCellWidget(row, col, new T(_stats_panel, _listing_view->lib_view(), item));
		}

		QStringList vertical_headers;
		_stats_panel->setVerticalHeaderItem(row, new QTableWidgetItem(header_text));
		_stats_panel->verticalHeaderItem(row)->setToolTip("Refresh all");
	}

private slots:
	void toggle_stats_panel_visibility();
	void update_header_width(int column, int old_width, int new_width);
	void refresh();
	void resize_panel();
	void axes_comb_changed();

private:
	PVListingView* _listing_view;
	QTableWidget* _stats_panel;
};

namespace __impl
{

class PVCellWidgetBase : public QWidget
{
	Q_OBJECT;

public:
	PVCellWidgetBase(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) : _table(table), _view(view), _item(item)
	{
		connect(table->verticalHeader(), SIGNAL(sectionClicked(int)), this, SLOT(vertical_header_clicked(int)));
	}

public:
	virtual void refresh() = 0;

public slots:
	virtual void auto_refresh() = 0;
	virtual void vertical_header_clicked(int index) = 0;

protected:
	QTableWidget* _table;
	Picviz::PVView const& _view;
	QTableWidgetItem* _item;

	QColor _invalid_color = QColor(0xe7, 0xa3, 0xa3);
};

class PVUniqueValuesCellWidget : public PVCellWidgetBase
{
	Q_OBJECT

public:
	PVUniqueValuesCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item);

public:
	void auto_refresh() override;
	void set_auto_refresh(bool auto_refresh);

public slots:
	void refresh() override;
	void vertical_header_clicked(int index) override;

private slots:
	void toggle_auto_refresh();
	void show_unique_values_dlg();

private:
	QLabel* _text;

	bool _refreshed = false;

	QPushButton* _refresh_icon;
	QPushButton* _autorefresh_icon;
	QPushButton* _unique_values_dlg_icon;

	const QPixmap _refresh_pixmap;
	const QPixmap _autorefresh_pixmap;
	const QPixmap _no_autorefresh_pixmap;
	const QPixmap _unique_values_pixmap;

	static std::unordered_map<uint32_t, bool> _auto_refresh;
};

}

}

#endif // __PVSTATSLISTINGWIDGET_H__
