#include <QApplication>
#include <QIcon>
#include <QString>
#include <QStyleHints>

#include <pvkernel/core/PVTheme.h>

#ifndef __PVMODDEDICON_H__
#define __PVMODDEDICON_H__

#include <QIconEngine>
#include <QLabel>

class PVModdedIconEngine : public QIconEngine
{
public:
    PVModdedIconEngine(QString icon_name) : QIconEngine()
    {
        _icons.append(create_icon(icon_name, "light"));
        _icons.append(create_icon(icon_name, "dark"));
    }

protected:
    QPixmap pixmap(const QSize &size, QIcon::Mode mode, QIcon::State state) override
    {
        return _icons[(int)PVTheme::is_color_scheme_dark()].pixmap(size, mode, state);
    }

    void paint(QPainter *painter, const QRect &rect, QIcon::Mode mode, QIcon::State state) override
    {
        _icons[(int)PVTheme::is_color_scheme_dark()].paint(painter, rect, Qt::AlignCenter, mode, state);
    }

    QIconEngine* clone() const override
    {
        return new PVModdedIconEngine(*this);
    }

private:
    QIcon create_icon(QString icon_name, QString color_scheme) {
        QIcon icon;
        QString icon_rc = QString(":/qss_icons/" + color_scheme + "/rc." + color_scheme + "/%1");
        icon.addFile(icon_rc.arg(icon_name + "@2x.png"), QSize(), QIcon::Normal);
        //addFile(icon_rc.arg(icon + "_focus@2x.png"), QSize(), QIcon::Active);
        icon.addFile(icon_rc.arg(icon_name + "_pressed@2x.png"), QSize(), QIcon::Selected);
        icon.addFile(icon_rc.arg(icon_name + "_disabled@2x.png"), QSize(), QIcon::Disabled);
        return icon;
    }

private:
    QVector<QIcon> _icons;

};

class PVModdedIcon : public QIcon
{
public:
    PVModdedIcon(QString icon) {
        const QString& color_scheme = PVTheme::color_scheme_name();

        QString icon_rc = QString(":/qss_icons/" + color_scheme + "/rc." + color_scheme + "/%1");
        addFile(icon_rc.arg(icon + "@2x.png"), QSize(), QIcon::Normal);
        //addFile(icon_rc.arg(icon + "_focus@2x.png"), QSize(), QIcon::Active);
        addFile(icon_rc.arg(icon + "_pressed@2x.png"), QSize(), QIcon::Selected);
        addFile(icon_rc.arg(icon + "_disabled@2x.png"), QSize(), QIcon::Disabled);
    }
};

class PVModdedIconLabel : public QLabel
{
    Q_OBJECT

public:
    PVModdedIconLabel(QString icon_name, QSize size)
    {
        auto set_pixmap_f = [this, icon_name, size](){
            setPixmap(PVModdedIcon(icon_name).pixmap(size.width(), size.height()));
        };
        QObject::connect(&PVTheme::get(), &PVTheme::color_scheme_changed, [set_pixmap_f](){
            set_pixmap_f();
        });
        set_pixmap_f();
    }
};

#endif // __PVMODDEDICON_H__