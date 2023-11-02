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