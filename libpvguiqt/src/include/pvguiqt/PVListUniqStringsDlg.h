/**
 * \file PVListColNrawDlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVLISTCOLNRAWDLG_H
#define PVGUIQT_PVLISTCOLNRAWDLG_H

#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVView_types.h>

#include <pvguiqt/PVListDisplayDlg.h>

#include <QAbstractListModel>
#include <QDialog>
#include <QStyledItemDelegate>

#include <QResizeEvent>

namespace PVGuiQt {

class PVListUniqStringsDlg: public PVListDisplayDlg
{
	Q_OBJECT

public:
	PVListUniqStringsDlg(Picviz::PVView_sp& view, PVCol c, PVRush::PVNraw::unique_values_t& values, size_t selection_count, QWidget* parent = NULL);
	virtual ~PVListUniqStringsDlg();

public:
	inline size_t get_selection_count() { return _selection_count; }
	inline bool use_logorithmic_scale() { return _use_logorithmic_scale; }

protected:
	void showEvent(QShowEvent * event) override;
	void sort_by_column(int col) override;
	void process_context_menu(QAction* act) override;
	void process_hhead_context_menu(QAction* act) override;

private slots:
	void view_resized();
	void section_resized(int logicalIndex, int oldSize, int newSize);

private:
	Picviz::PVView& lib_view() { return *_obs.get_object(); }
	void multiple_search(QAction* act);
	void resize_section();

private:
	PVCol _col;
	PVHive::PVObserverSignal<Picviz::PVView> _obs;
	PVHive::PVActor<Picviz::PVView> _actor;
	bool _store_last_section_width = true;
	int _last_section_width = 125;
	size_t _selection_count;

	bool _use_logorithmic_scale = true;
	QAction* _act_toggle_linear;
	QAction* _act_toggle_log;
};

namespace __impl {

class PVListUniqStringsModel: public QAbstractListModel
{
	Q_OBJECT

public:
	PVListUniqStringsModel(PVRush::PVNraw::unique_values_t& values, QWidget* parent = NULL);

public:
	int rowCount(QModelIndex const& parent = QModelIndex()) const;
	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;
	int columnCount(const QModelIndex& /*index*/) const override;

private:
	typedef std::pair<std::string_tbb, size_t> pair_t;
	std::vector<pair_t> _values;
};

class PVListUniqStringsDelegate: public QStyledItemDelegate
{
	Q_OBJECT

public:
	PVListUniqStringsDelegate(PVListUniqStringsDlg* parent) : QStyledItemDelegate(parent) {}

protected:
	void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

private:
	PVGuiQt::PVListUniqStringsDlg* get_dialog() const;
};

/**
 * \class PVTableViewResizeEventFilter
 *
 * \note This class is intended to be notified of the resize of the table view
 *       to resize its last section according to the user preference.
 *       i.e: the last section can only be changed by user interaction
 *       on the section, not on the dialog size.
 *
 *       Note: I couldn't subclass the QTableView to achieve this goal because
 *             the UI was created using Qt Creator, but it would also have
 *             been a bit overkill anyway...
 */
class PVTableViewResizeEventFilter : public QObject
{
	Q_OBJECT

signals:
	void resized();

protected:
	bool eventFilter(QObject *obj, QEvent *event) override
	{
		 if (event->type() == QEvent::Resize) {
			 emit resized();
		 }
		 return QObject::eventFilter(obj, event);
	}
};

}

}

#endif
