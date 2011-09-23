//! \file PVAxesIndexEditor.cpp
//! $Id: PVAxesIndexEditor.cpp 2699 2011-05-12 03:58:48Z cdash $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxesIndexType.h>

#include <picviz/PVView.h>

#include <PVAxesIndexEditor.h>

#include <QList>
#include <QAbstractItemView>
#include <QSizePolicy>

/******************************************************************************
 *
 * PVCore::PVAxesIndexEditor::PVAxesIndexEditor
 *
 *****************************************************************************/
PVInspector::PVAxesIndexEditor::PVAxesIndexEditor(Picviz::PVView& view, QWidget *parent):
	QListWidget(parent),
	_view(view)
{
	setSelectionMode(QAbstractItemView::ExtendedSelection);

	QSizePolicy sp(QSizePolicy::Expanding, QSizePolicy::Minimum);
	sp.setHeightForWidth(sizePolicy().hasHeightForWidth());
	setSizePolicy(sp);
	setMinimumHeight(70);
}

/******************************************************************************
 *
 * PVInspector::PVAxesIndexEditor::~PVAxesIndexEditor
 *
 *****************************************************************************/
PVInspector::PVAxesIndexEditor::~PVAxesIndexEditor()
{
}

/******************************************************************************
 *
 * PVInspector::PVAxesIndexEditor::set_axes_index
 *
 *****************************************************************************/
void PVInspector::PVAxesIndexEditor::set_axes_index(PVCore::PVAxesIndexType axes_index)
{
	clear();
			
	QStringList const& axes = _view.get_axes_names_list();
	QListWidgetItem* item;

	for (int i = 0; i < axes.count(); i++) {
		item = new QListWidgetItem(axes[i]);
		addItem(item);
		if (std::find(axes_index.begin(), axes_index.end(), i) != axes_index.end()) {
			item->setSelected(true);
		}
	}
}

PVCore::PVAxesIndexType PVInspector::PVAxesIndexEditor::get_axes_index() const
{
	QModelIndexList selitems = selectionModel()->selectedIndexes();
	QModelIndexList::iterator it;
	PVCore::PVAxesIndexType ret;
	for (it = selitems.begin(); it != selitems.end(); it++) {
		ret.push_back((*it).row());
	}

	return ret;
}
