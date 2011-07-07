//! \file PVListingModel.h
//! $Id: PVListingModel.h 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGMODEL_H
#define PVLISTINGMODEL_H

#include <QtGui>
#include <QtCore>

#include <PVListingModelBase.h>
#include <picviz/state-machine.h>
#include <PVProgressBox.h>

namespace PVInspector {
    class PVMainWindow;
    class PVTabSplitter;

    /**
     * \class PVListingModel
     */
    class PVListingModel : public PVListingModelBase {
        Q_OBJECT

        QBrush not_zombie_font_brush; //!<
        QBrush zombie_font_brush; //!<
        //QVector<QStringList> widgetCpyOfData;
        //corresponding table between widgetCpyOfData and nrow



    public:

        enum TypeOfSort {
            NoOrder, AscendingOrder, DescendingOrder
        };

        /**
         * Constructor.
         *
         * @param mw
         * @param parent
         */
        PVListingModel(PVMainWindow *mw, PVTabSplitter *parent, Picviz::StateMachine_ListingMode_t state = Picviz::LISTING_BAD_LISTING_MODE);

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
         * Order to PVView to sort table
         * @param idColumn the id of the column to sort the table.
         */
        void sortByColumn(int idColumn);

        /**
         * not implemented
         * @param line
         * @return 
         */
        int getCorrespondance(int line);
        
        /**
         * initialize the matching table for sort.
         */
        void initCorrespondance();
        
        /**
         * reset the model
         * @param initMatchTable
         */
        virtual void reset_model(bool initMatchTable = true);
        
        /**
         * @brief set listing mode
         * @param mode
         */
        void setState(Picviz::StateMachine_ListingMode_t mode);
        
    private:
        //sorting data
        QVector<int> matchingTable; //!<the table sort, modify this array to order the values
        TypeOfSort sortOrder;//!<save the current sorting state (NoOrder, AscendingOrder, DescendingOrder)
        int colSorted;//!<save the last column whiche was used to sort
        Picviz::StateMachine_ListingMode_t state_listing; //!<this state indicate the mode of listing

    };
}

#endif
