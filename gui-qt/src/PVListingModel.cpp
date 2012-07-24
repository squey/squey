/**
 * \file PVListingModel.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QtCore>
#include <QtGui>

#include <pvkernel/core/general.h>

#include <picviz/PVView.h>
#include <pvkernel/core/PVColor.h>
#include <picviz/PVStateMachine.h>

#include <PVCustomQtRoles.h>
#include <PVListingModel.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <picviz/PVSortQVectorQStringList.h>

#include <tbb/parallel_sort.h>

#include <omp.h>

#include <map>

/******************************************************************************
 *
 * PVInspector::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVInspector::PVListingModel::PVListingModel(PVMainWindow *mw, PVTabSplitter *parent) : QAbstractTableModel(parent) {
	PVLOG_DEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);

	PVLOG_INFO("%s : Creating object\n", __FUNCTION__);

	main_window = mw;
	parent_widget = parent;

	assert(parent_widget);

	test_fontdatabase = QFontDatabase();
	//test_fontdatabase.addApplicationFont(QString("/donnees/HORS_SVN/TESTS_PHIL/GOOGLE_WEBFONTS/Convergence/Convergence-Regular.ttf"));
	//test_fontdatabase.addApplicationFont(QString("/donnees/HORS_SVN/TESTS_PHIL/GOOGLE_WEBFONTS/Metrophobic/Metrophobic.ttf"));
	
	test_fontdatabase.addApplicationFont(QString(":/Convergence-Regular.ttf"));
	
	row_header_font = QFont("Convergence-Regular", 6);
	

	select_brush = QBrush(QColor(255, 240, 200));
	unselect_brush = QBrush(QColor(180, 180, 180));
	select_font = QFont();
	select_font.setBold(true);
	unselect_font = QFont();
	not_zombie_font_brush = QBrush(QColor(0, 0, 0));
	zombie_font_brush = QBrush(QColor(200, 200, 200));
	colSorted = -1;

	reset_lib_view();
}



/******************************************************************************
 *
 * PVInspector::PVListingModel::columnCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::columnCount(const QModelIndex &) const 
{
	PVLOG_HEAVYDEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);

	return lib_view->get_axes_count();
}



/******************************************************************************
 *
 * PVInspector::PVListingModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::data(const QModelIndex &index, int role) const {
	PVLOG_HEAVYDEBUG("PVInspector::PVListingModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

	PVCore::PVColor color;
	int i;
	int real_row_index;

	unsigned char r;
	unsigned char g;
	unsigned char b;

	real_row_index = index.row();

	switch (role) {
		case (Qt::DisplayRole):
			return lib_view->get_data(real_row_index, index.column());

		case (Qt::TextAlignmentRole):
			return (Qt::AlignLeft + Qt::AlignVCenter);

		case (Qt::BackgroundRole):
			/* We get the current selected axis index */
			i = lib_view->active_axis;

			if ((state_machine->is_axes_mode()) && (i == index.column())) {
				/* We must provide an evidence of the active_axis ! */
				return QBrush(QColor(130, 100, 25));
			} else {
				if (lib_view->get_line_state_in_output_layer(real_row_index)) {
					color = lib_view->get_color_in_output_layer(real_row_index);
					r = color.r();
					g = color.g();
					b = color.b();

					return QBrush(QColor(r, g, b));
				} else {
					return unselect_brush;
				}
			}

		case (Qt::ForegroundRole):
			if (lib_view->layer_stack_output_layer.get_selection().get_line(real_row_index)) {
				/* The line is NOT a ZOMBIE */
				return not_zombie_font_brush;
			} else {
				/* The line is a ZOMBIE */
				return zombie_font_brush;
			}

		case (PVCustomQtRoles::Sort):
		{
			QVariant ret;
			ret.setValue<void*>((void*) &lib_view->get_data_unistr(real_row_index, index.column()));
			return ret;
		}
	}
	return QVariant();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVInspector::PVListingModel::flags(const QModelIndex &/*index*/) const {
	return (Qt::ItemIsEnabled | Qt::ItemIsSelectable);
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::headerData(int section, Qt::Orientation orientation, int role) const {
	switch (role) {
		case (Qt::DisplayRole):
			if (orientation == Qt::Horizontal) {
				if (section < 0 || section >= lib_view->get_axes_count()) {
					// That should never happen !
					return QVariant();
				}
				QString axis_name = lib_view->get_axis_name(section);
				return QVariant(axis_name);
			} else {
				if (section < 0) {
					// That should never happen !
					return QVariant();
				}
				return section;
			}
			break;
// 		case (Qt::FontRole):
// 			if ((orientation == Qt::Vertical) && (lib_view->real_output_selection.get_line(getRealRowIndex(section)))) {
// 				return row_header_font;
// // 				return select_font;
// 			} else {
// 				return unselect_font;
// 			}
// 			break;
		case (Qt::TextAlignmentRole):
			if (orientation == Qt::Horizontal) {
				return (Qt::AlignLeft + Qt::AlignVCenter);
			} else {
				return (Qt::AlignRight + Qt::AlignVCenter);
			}
			break;

		default:
			return QVariant();
	}

	return QVariant();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::reset_lib_view
 *
 *****************************************************************************/
void PVInspector::PVListingModel::reset_lib_view()
{
	PVLOG_DEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);

	beginResetModel();
	
	lib_view = parent_widget->get_lib_view();
	assert(lib_view);
	state_machine = lib_view->state_machine;

	endResetModel();
}

/******************************************************************************
 *
 * PVInspector::PVListingModel::rowCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::rowCount(const QModelIndex &/*index*/) const 
{
	return lib_view->get_row_count();
}
