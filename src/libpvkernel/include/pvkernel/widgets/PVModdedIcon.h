#include <pvkernel/core/PVTheme.h>
#include <qcontainerfwd.h>
#include <qpixmap.h>
#include <qsize.h>
#include <qtmetamacros.h>
#include <QApplication>
#include <QIcon>
#include <QString>
#include <QStyleHints>

class QPainter;
class QRect;

#ifndef __PVMODDEDICON_H__
#define __PVMODDEDICON_H__

#include <QIconEngine>
#include <QLabel>

class PVModdedIconEngine : public QIconEngine
{
public:
    PVModdedIconEngine(QString icon_name);

protected:
    QPixmap pixmap(const QSize &size, QIcon::Mode mode, QIcon::State state) override;

    void paint(QPainter *painter, const QRect &rect, QIcon::Mode mode, QIcon::State state) override;

    QIconEngine* clone() const override;

private:
    QIcon create_icon(QString icon_name, QString color_scheme);

private:
    QVector<QIcon> _icons;

};

class PVModdedIcon : public QIcon
{
public:
    PVModdedIcon(QString icon_name);
    PVModdedIcon();
};

class PVModdedIconLabel : public QLabel
{
    Q_OBJECT

public:
    PVModdedIconLabel(QString icon_name, QSize size);

private Q_SLOTS:
    void set_pixmap();

private:
    PVModdedIcon _icon;
    QString _name;
    QSize _size;
};

#endif // __PVMODDEDICON_H__