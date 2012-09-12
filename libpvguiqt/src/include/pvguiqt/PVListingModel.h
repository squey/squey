/**
 * \file PVListingModel.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVGUIQT_PVLISTINGMODEL_H
#define PVGUIQT_PVLISTINGMODEL_H

#include <vector>
#include <utility>

#include <QAbstractTableModel>
#include <QBrush>
#include <QFont>
#include <QFontDatabase>

#include <pvkernel/core/general.h>
#include <picviz/PVView_types.h>

#include <pvhive/PVObserverSignal.h>

#include <QAbstractTableModel>

#include <tbb/tbb_allocator.h>
#include <tbb/cache_aligned_allocator.h>

#include <pvhive/PVActor.h>

namespace PVGuiQt {

/**
 * \class PVListingModel
 */

class PVListingModel : public QAbstractTableModel
{
	Q_OBJECT

public:
    enum TypeOfSort {
        NoOrder, AscendingOrder, DescendingOrder
    };

private:
	QBrush not_zombie_font_brush; //!<
	QBrush zombie_font_brush; //!<

protected:
	QFontDatabase test_fontdatabase;
	QFont  row_header_font;
	QBrush select_brush;            //!<
	QFont  select_font;             //!<
	QBrush unselect_brush;          //!<
	QFont  unselect_font;           //!<

public:
    /**
     * Constructor.
     *
     * @param mw
     * @param parent
     */
    PVListingModel(Picviz::PVView_sp& view, QObject* parent = NULL);

    /**
     * return data requested by the View
     * @param index
     * @param role
     * @return
     */
    QVariant data(const QModelIndex &index, int role) const;

    /**
     * return header requested by the View
     * @param section
     * @param orientation
     * @param role
     * @return 
     */
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;

    /**
     * 
     * @param index
     * @return the number of log line.
     */
    int rowCount(const QModelIndex &index) const;

    /**
     *
     * @param index
     *
     * @return
     */
    int columnCount(const QModelIndex &index) const;

    /**
     *
     * @param index
     *
     * @return
     */
    Qt::ItemFlags flags(const QModelIndex &index) const;

private slots:
	void view_about_to_be_deleted(PVHive::PVObserverBase* o);

private:
	inline Picviz::PVView const& lib_view() const { return *_obs.get_object(); }

private:
	PVHive::PVActor<Picviz::PVView> _actor;
	PVHive::PVObserverSignal<Picviz::PVView> _obs;
	bool _view_valid;
};

}

#endif
