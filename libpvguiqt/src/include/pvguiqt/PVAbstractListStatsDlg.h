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
#include <QStyledItemDelegate>

#include <QResizeEvent>

class QComboBox;

namespace PVGuiQt {

namespace __impl {
class PVListStringsDelegate;
class PVAbstractListStatsModel;

class PVAbstractListStatsRangePicker;
}

class PVStringSortProxyModel;

class PVAbstractListStatsDlg: public PVListDisplayDlg
{
	// TODO: Better members visibility
	Q_OBJECT

	friend class __impl::PVListStringsDelegate;

public:
	PVAbstractListStatsDlg(
		Picviz::PVView_sp& view,
		PVCol c,
		__impl::PVAbstractListStatsModel* model,
		size_t absolute_max_count,
		size_t relative_min_count,
		size_t relative_max_count,
		QWidget* parent = nullptr) :
		PVListDisplayDlg((QAbstractListModel*) model, parent),
		_col(c),
		_absolute_max_count(absolute_max_count),
		_relative_min_count(relative_min_count),
		_relative_max_count(relative_max_count)
	{
		init(view);
	}

	void init(Picviz::PVView_sp& view);
	virtual ~PVAbstractListStatsDlg();

public:
	inline size_t absolute_max_count() const { return _absolute_max_count; }
	inline size_t relative_min_count() const { return _relative_min_count; }
	inline size_t relative_max_count() const { return _relative_max_count; }
	inline size_t max_count() const { return _use_absolute_max_count ? _absolute_max_count : _relative_max_count; }

	inline bool use_logarithmic_scale() { return _use_logarithmic_scale; }

protected:
	void showEvent(QShowEvent * event) override;
	void sort_by_column(int col) override;
	bool process_context_menu(QAction* act) override;
	void process_hhead_context_menu(QAction* act) override;
	void ask_for_copying_count() override;
	QString export_line(
		PVGuiQt::PVStringSortProxyModel* model,
		std::function<void (PVGuiQt::PVStringSortProxyModel*, int, QModelIndex&)> f,
		int i
	) override;

	/**
	 * create a new layer using the selected values.
	 */
	void create_layer_with_selected_values();

	/**
	 * create a set of new layers, one layer for each selected value.
	 */
	void create_layers_for_selected_values();

protected slots:
	void view_resized();
	void section_resized(int logicalIndex, int oldSize, int newSize);
	void scale_changed(QAction* act);
	void max_changed(QAction* act);

protected slots:
	void select_set_mode_count(bool checked);
	void select_set_mode_frequency(bool checked);
	void select_refresh(bool checked);

protected:
	Picviz::PVView* lib_view() { return _obs.get_object(); }
	void multiple_search(QAction* act, const QStringList &sl);
	void resize_section();

protected:
	PVCol _col;
	PVHive::PVObserverSignal<Picviz::PVView> _obs;
	PVHive::PVActor<Picviz::PVView> _actor;
	bool _store_last_section_width = true;
	int _last_section_width = 250;

	size_t _absolute_max_count;
	size_t _relative_min_count;
	size_t _relative_max_count;
	bool _use_absolute_max_count = true;

	bool _use_logarithmic_scale = true;
	QAction* _act_toggle_linear;
	QAction* _act_toggle_log;
	QAction* _act_toggle_absolute;
	QAction* _act_toggle_relative;

	QAction* _act_show_percentage;
	QAction* _act_show_count;
	QAction* _act_show_scientific_notation;

	__impl::PVAbstractListStatsRangePicker* _select_picker;
	bool                                    _select_is_count;

	QMenu* _copy_values_menu;
	QAction* _copy_values_without_count_act;
	QAction* _copy_values_with_count_act;

	QAction* _create_layer_with_values_act;
	QAction* _create_layers_for_values_act;

	bool _copy_count;

private:
	/*RH this QAction list is a big workaround^Whack to:
	 * - use a PVLayerFilter but not the right way
	 * - keep track of "multiple search"'s dynamically created QAction as all _act_* do
	 * - to use ::multiple_search() to change the current selection to affect it to
	 *   newly created layer
	 * theorically, the wanted action is numbered 1
	 */
	QList<QAction*> _msearch_actions;
};

namespace __impl {

class PVAbstractListStatsModel: public QAbstractListModel
{

public:
	PVAbstractListStatsModel(QWidget* parent = NULL) : QAbstractListModel(parent) {}

public:
	QVariant headerData(int section, Qt::Orientation orientation, int role) const
	{
		static QString h[] = { "Value", "Count " };

		switch (role) {
			case(Qt::DisplayRole) :
			{
				if (orientation == Qt::Horizontal) {
					if (section == 1) {
						return count_header();
					}
					return h[section];
				}
				return QVariant(QString().setNum(section));
			}
			break;
			case (Qt::TextAlignmentRole) :
				if (orientation == Qt::Horizontal) {
					return (Qt::AlignLeft + Qt::AlignVCenter);
				}
				else {
					return (Qt::AlignRight + Qt::AlignVCenter);
				}
			break;
			default:
				return QVariant();
			break;
		}

		return QVariant();
	}

	int columnCount(const QModelIndex& /*index*/) const
	{
		return 2;
	}

public:
	void use_logarithmic_scale(bool log_scale)
	{
		_use_logarithmic_scale = log_scale;
	}

	void use_absolute_max_count(bool abs_max)
	{
		_use_absolute_max_count = abs_max;
	}

	static inline QString format_occurence(uint64_t occurence_count) { return QLocale().toString((qulonglong)occurence_count); };
	static inline QString format_percentage(double ratio) { return QLocale().toString(ratio * 100, 'f', 1) + "%"; };
	static inline QString format_scientific_notation(double ratio) { return QLocale().toString(ratio, 'e', 1); };

private:
	QString count_header() const
	{
		return QString("Count ") + " (" + ( _use_logarithmic_scale ? "Log" : "Lin") + "/" + (_use_absolute_max_count ? "Abs" : "Rel") + ")";
	}

private:
	bool _use_logarithmic_scale;
	bool _use_absolute_max_count;

};

class PVListStringsDelegate: public QStyledItemDelegate
{
	Q_OBJECT

public:
	PVListStringsDelegate(PVAbstractListStatsDlg* parent) : QStyledItemDelegate(parent) {}

protected:
	void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

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
