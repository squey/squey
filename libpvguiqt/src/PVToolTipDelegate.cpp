/**
 * \file PVToolTipDelegate.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QAbstractItemView>
#include <QHelpEvent>
#include <QModelIndex>
#include <QString>
#include <QStyleOptionViewItem>
#include <QTextDocument>
#include <QToolTip>

#include <pvguiqt/PVToolTipDelegate.h>
#include <pvkernel/widgets/PVUtils.h>

#define TOOLTIP_MAX_WIDTH 800

bool PVGuiQt::PVToolTipDelegate::helpEvent(QHelpEvent* e, QAbstractItemView* view, const QStyleOptionViewItem& option, const QModelIndex& index)
{
    if (!e || !view) {
    	return false;
    }

    if (e->type() == QEvent::ToolTip) {
        QRect rect = view->visualRect(index);
        QSize size = sizeHint(option, index);
        if (rect.width() < size.width()) {
            QVariant tooltip = index.data(Qt::DisplayRole);
            if (tooltip.canConvert<QString>()) {
            	QString tooltip_text = Qt::escape(tooltip.toString());
            	int tooltip_width = QFontMetrics(view->font()).width(tooltip.toString());
            	if (tooltip_width > TOOLTIP_MAX_WIDTH) {
            		PVWidgets::PVUtils::html_word_wrap_text(tooltip_text, QToolTip::font(), TOOLTIP_MAX_WIDTH);
            	}
                QToolTip::showText(e->globalPos(), QString("<div>%1</div>").arg(tooltip_text), view);
                return true;
            }
        }
        if (!QStyledItemDelegate::helpEvent(e, view, option, index)) {
            QToolTip::hideText();
        }
        return true;
    }

    return QStyledItemDelegate::helpEvent(e, view, option, index);
}