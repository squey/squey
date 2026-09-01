#include <pvkernel/widgets/PVModdedIcon.h>
#include <QSizeF>
#include <QtCore/qobjectdefs.h>
#include <qguiapplication.h>
#include <qnamespace.h>
#include <QScreen>
#include <QWindow>

#include "pvkernel/core/PVTheme.h"

class QPainter;
class QRect;

PVModdedIconEngine::PVModdedIconEngine(QString icon_name) : QIconEngine()
{
    _icons.append(create_icon(icon_name, "light"));
    _icons.append(create_icon(icon_name, "dark"));
}

QPixmap PVModdedIconEngine::pixmap(const QSize &size, QIcon::Mode mode, QIcon::State state)
{
    return _icons[(int)PVCore::PVTheme::is_color_scheme_dark()].pixmap(size, mode, state);
}

QPixmap PVModdedIconEngine::scaledPixmap(const QSize &size, QIcon::Mode mode, QIcon::State state,
                                         qreal scale)
{
    /* The size is in device independent pixels and the scale is what the screen
     * this is bound for wants, so the pixmap has to carry size * scale real
     * pixels and say that it does. Qt then draws it at `size`, out of enough
     * pixels to stay sharp.
     *
     * Working the ratio out here instead, off whichever window held the focus,
     * is what this used to do: it answered for a window that was not the one
     * being drawn -- while a menu is open the focus is on the menu's own popup --
     * so an icon on a button carrying a menu changed size as that menu came and
     * went, and on a second screen it was built for the wrong one. Too big for
     * the room it was given, and a widget aligns a pixmap in that room rather
     * than scaling it down, so the edges fell outside and were clipped.
     */
    return _icons[(int)PVCore::PVTheme::is_color_scheme_dark()].pixmap(size, scale, mode, state);
}

void PVModdedIconEngine::paint(QPainter *painter, const QRect &rect, QIcon::Mode mode, QIcon::State state)
{
    _icons[(int)PVCore::PVTheme::is_color_scheme_dark()].paint(painter, rect, Qt::AlignCenter, mode, state);
}

QIconEngine* PVModdedIconEngine::clone() const
{
    return new PVModdedIconEngine(*this);
}

PVModdedIcon::PVModdedIcon(QString icon_name) : QIcon(new PVModdedIconEngine(icon_name))
{
}

PVModdedIcon::PVModdedIcon() : QIcon()
{
}

QIcon PVModdedIconEngine::create_icon(QString icon_name, QString color_scheme) {
    QIcon icon;
    QString icon_rc = QString(":/qss_icons/" + color_scheme + "/rc." + color_scheme + "/%1");
    icon.addFile(icon_rc.arg(icon_name + "@2x.png"), QSize(), QIcon::Normal);
    //addFile(icon_rc.arg(icon + "_focus@2x.png"), QSize(), QIcon::Active);
    icon.addFile(icon_rc.arg(icon_name + "_pressed@2x.png"), QSize(), QIcon::Selected);
    icon.addFile(icon_rc.arg(icon_name + "_disabled@2x.png"), QSize(), QIcon::Disabled);
    return icon;
}

PVModdedIconLabel::PVModdedIconLabel(QString name, QSize size) : 
    _icon(PVModdedIcon(name)),
    _name(name),
    _size(size)
{
    connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this, &PVModdedIconLabel::set_pixmap);
    set_pixmap();
}

void PVModdedIconLabel::set_pixmap()
{
    setPixmap(_icon.pixmap(_size, QIcon::Normal, QIcon::On));
};