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
#include <QTextLayout>
#include <QApplication>
#include <QStyle>

#include <pvguiqt/PVToolTipDelegate.h>
#include <pvkernel/widgets/PVUtils.h>

#include <pvkernel/core/PVLogger.h>


bool PVGuiQt::PVToolTipDelegate::helpEvent(QHelpEvent* e, QAbstractItemView* view, const QStyleOptionViewItem& option, const QModelIndex& index)
{
    if (!e || !view) {
    	return false;
    }

    if (e->type() == QEvent::ToolTip) {
        QRect rect = view->visualRect(index);
        QSize size = sizeHint(option, index);

        // Recompute word-wrap text elision
        const int textMargin = QApplication::style()->pixelMetric(QStyle::PM_FocusFrameHMargin, 0, view) + 1;
        int width = rect.width() - textMargin*2;
        QString text = index.data(Qt::DisplayRole).toString();
		QTextLayout textLayout(text);
		QTextOption text_option;
		text_option.setAlignment(QStyle::visualAlignment(option.direction, option.displayAlignment));
		text_option.setWrapMode(QTextOption::WordWrap);
		textLayout.setTextOption(text_option);
		textLayout.setFont(option.font);
		textLayout.beginLayout();
		QTextLine line1 = textLayout.createLine();
		line1.setLineWidth(width);
		QTextLine line2 = textLayout.createLine();
		if (line2.isValid()) {
			line2.setLineWidth(width);
		}
		textLayout.endLayout();

		QString last_line = text.right(text.length()-line1.textLength());
		QString elided_last_line = QFontMetrics(option.font).elidedText(last_line, Qt::ElideRight, width);

        if (last_line != elided_last_line) {
            QVariant tooltip = index.data(Qt::DisplayRole);
            if (tooltip.canConvert<QString>()) {
            	QString tooltip_text = Qt::escape(tooltip.toString());
				const int32_t tooltip_max_width = PVWidgets::PVUtils::tooltip_max_width(view);
            	int tooltip_width = QFontMetrics(view->font()).width(tooltip.toString());
            	if (tooltip_width > tooltip_max_width) {
            		PVWidgets::PVUtils::html_word_wrap_text(tooltip_text, QToolTip::font(), tooltip_max_width);
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
