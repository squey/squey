/**
 * \file PVAbstractListStatsDlg.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVGUIQT_PVABSTRACTLISTSTATSDLG_H__
#define __PVGUIQT_PVABSTRACTLISTSTATSDLG_H__

#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVView_types.h>

#include <pvguiqt/PVListDisplayDlg.h>

#include <QAbstractListModel>
#include <QDialog>
#include <QStyledItemDelegate>

#include <QResizeEvent>

namespace PVGuiQt {

namespace __impl {
class PVListStringsDelegate;

template <typename T>
class PVListUniqStringsModel;
}

class PVAbstractListStatsDlg: public PVListDisplayDlg
{
	// TODO: Better members visibility
	Q_OBJECT

	friend class __impl::PVListStringsDelegate;

public:
	PVAbstractListStatsDlg(Picviz::PVView_sp& view, PVCol c, QAbstractListModel* model, size_t selection_count, QWidget* parent = NULL) :
		PVListDisplayDlg(model, parent), _col(c), _selection_count(selection_count)
	{
		init(view);
	}

	void init(Picviz::PVView_sp& view);
	virtual ~PVAbstractListStatsDlg();

public:
	inline size_t get_selection_count() { return _selection_count; }
	inline bool use_logorithmic_scale() { return _use_logarithmic_scale; }

protected:
	void showEvent(QShowEvent * event) override;
	void sort_by_column(int col) override;
	void process_context_menu(QAction* act) override;
	void process_hhead_context_menu(QAction* act) override;

protected slots:
	void view_resized();
	void section_resized(int logicalIndex, int oldSize, int newSize);
	void scale_changed(QAction* act);

protected:
	Picviz::PVView& lib_view() { return *_obs.get_object(); }
	void multiple_search(QAction* act);
	void resize_section();

protected:
	PVCol _col;
	PVHive::PVObserverSignal<Picviz::PVView> _obs;
	PVHive::PVActor<Picviz::PVView> _actor;
	bool _store_last_section_width = true;
	int _last_section_width = 175;

	size_t _selection_count;

	bool _use_logarithmic_scale = true;
	QAction* _act_toggle_linear;
	QAction* _act_toggle_log;

	QAction* _act_show_percentage;
	QAction* _act_show_count;
	QAction* _act_show_scientific_notation;

	size_t _max_e;
};

namespace __impl {

class PVAbstractListStatsModel: public QAbstractListModel
{

public:
	PVAbstractListStatsModel(QWidget* parent = NULL) : QAbstractListModel(parent) {}

public:
	QVariant headerData(int section, Qt::Orientation orientation, int role) const
	{
		QHash<size_t, QString> h;
		h[0] = "Value";
		h[1] = "Frequency";

		if (role == Qt::DisplayRole) {
			if (orientation == Qt::Horizontal) {
				return h[section];
			}
			return QVariant(QString().setNum(section));
		}
		else if (role == Qt::TextAlignmentRole) {
			if (orientation == Qt::Horizontal) {
				return (Qt::AlignLeft + Qt::AlignVCenter);
			}
			else {
				return (Qt::AlignRight + Qt::AlignVCenter);
			}
		}

		return QVariant();
	}

	int columnCount(const QModelIndex& /*index*/) const
	{
		return 2;
	}
};

class PVListStringsDelegate: public QStyledItemDelegate
{
	Q_OBJECT

public:
	PVListStringsDelegate(PVAbstractListStatsDlg* parent) : QStyledItemDelegate(parent) {}

protected:
	void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

private:
	inline QString format_occurence(size_t occurence_count) const { return QString("%L1").arg(occurence_count); };
	inline QString format_percentage(double ratio) const { return QString::number(ratio * 100, 'f', 1) + "%"; };
	inline QString format_scientific_notation(double ratio) const  { return QString::number(ratio, 'e', 1); };

	PVGuiQt::PVAbstractListStatsDlg* d() const;
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


#endif // __PVGUIQT_PVABSTRACTLISTSTATSDLG_H__
