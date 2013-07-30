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

namespace PVGuiQt {

class PVListUniqStringsDlg: public PVListDisplayDlg
{
	Q_OBJECT

public:
	PVListUniqStringsDlg(Picviz::PVView_sp& view, PVCol c, PVRush::PVNraw::unique_values_t& values, size_t selection_count, QWidget* parent = NULL);
	virtual ~PVListUniqStringsDlg();

public:
	inline size_t get_selection_count() { return _selection_count; }

protected:
	void resizeEvent(QResizeEvent * event) override;
	void showEvent(QShowEvent * event) override;
	void sort_by_column(int col) override;

private slots:
	void section_resized(int logicalIndex, int oldSize, int newSize);

private:
	Picviz::PVView& lib_view() { return *_obs.get_object(); }
	void process_context_menu(QAction* act);
	void multiple_search(QAction* act);
	void resize_section();

private:
	PVCol _col;
	PVHive::PVObserverSignal<Picviz::PVView> _obs;
	PVHive::PVActor<Picviz::PVView> _actor;
	bool _resize = false;
	int _last_section_size = 125;
	size_t _selection_count;
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

}

}

#endif
