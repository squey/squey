#include <pvguiqt/PVImportWorkflowTabBar.h>

#include <QPainterPath>
#include <QStylePainter>
#include <QStyleOptionTab>


QSize PVGuiQt::PVImportWorkflowTabBar::tabSizeHint(int index) const
{
    QSize size = QTabBar::tabSizeHint(index);
    bool first_or_last = (index == 0 or index == (count() -1));
    return QSize(size.width() + (ARROW_WIDTH * (2 - first_or_last)), size.height());
}

QSize PVGuiQt::PVImportWorkflowTabBar::minimumTabSizeHint(int index) const
{
    return tabSizeHint(index);
}

void PVGuiQt::PVImportWorkflowTabBar::paintEvent(QPaintEvent * /*event*/)
{
    QStylePainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    QStyleOptionTab opt;
    
    QPen pen = painter.pen();
    for(int i = 0; i < count(); i++)
    {
        initStyleOption(&opt, i);

        QPainterPath path;

        int start_x = opt.rect.x() + (PEN_WIDTH/2);
        int stop_x = opt.rect.x() + opt.rect.width() - (PEN_WIDTH / 2);
        if (i < (count()-1)) {
            stop_x += (ARROW_WIDTH / 3);
        }
        int start_y = opt.rect.y() + (PEN_WIDTH/2);
        int stop_y = opt.rect.y() + opt.rect.height() - (PEN_WIDTH / 2);
        path.moveTo(start_x, start_y);
        if (i == (count()-1)) { // last tab
            path.lineTo(stop_x, start_y);
            path.lineTo(stop_x, stop_y);
        }
        else { // other tabs
            path.lineTo(stop_x - ARROW_WIDTH, start_y);
            path.lineTo(stop_x, start_y + (opt.rect.height() / 2));
            path.lineTo(stop_x - ARROW_WIDTH, stop_y);
        }
        path.lineTo(start_x, stop_y);
        if (i == 0) { // first tab
            path.lineTo(start_x, start_y);
        }
        else { // other tabs
            path.lineTo(start_x + ARROW_WIDTH, start_y + (opt.rect.height() / 2));
            path.lineTo(start_x, start_y);
        }

        if (isTabEnabled(i)) {
            painter.setPen(QPen(QBrush(QColor(COLOR)), PEN_WIDTH));
        }
        else {
            QColor disabled_text_color = palette().color(QPalette::Disabled, QPalette::WindowText);
            painter.setPen(QPen(QBrush(disabled_text_color), PEN_WIDTH));
        }

        painter.drawPath(path);
        if (i == currentIndex()) {
            painter.fillPath(path, QBrush(QColor(COLOR)));
        }

        if (isTabEnabled(i)) {
            if (currentIndex() == i) {
                painter.setPen(QPen(QBrush(Qt::white), PEN_WIDTH));
            }
            else {
                painter.setPen(pen);
            }
        }

        QRect text_rect = opt.rect.adjusted(5, 0, 0, 0);;
        if (i != 0) { // not first tab
            text_rect.adjust(ARROW_WIDTH, 0, 0, 0);
        }

        painter.drawItemText(text_rect, Qt::AlignLeft|Qt::AlignVCenter, palette(), true, opt.text);
    }
}