/**
 * @file
 *
 * 
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#ifndef PVGUIQT_PVTABLEVIEW_HPP
#define PVGUIQT_PVTABLEVIEW_HPP

#include <QTableView>

namespace PVGuiQt {
/**
 * It is a QTableView with filtering on Tooltip event to display tooltip only
 * when the content can't be display in the corresponding cell.
 */
class PVTableView: public QTableView
{
    public:
        PVTableView(QWidget* parent): QTableView(parent)
        {}

    protected:
        /**
         * Check for ToolTip event to disable tooltip when cell is big enough
         * to show the full cell content
         */
        bool viewportEvent(QEvent *event) override;
};

}

#endif
