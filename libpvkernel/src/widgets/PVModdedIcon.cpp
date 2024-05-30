#include <pvkernel/widgets/PVModdedIcon.h>

#include <QScreen>
#include <QWindow>

PVModdedIconEngine::PVModdedIconEngine(QString icon_name) : QIconEngine()
{
    _icons.append(create_icon(icon_name, "light"));
    _icons.append(create_icon(icon_name, "dark"));
}

QPixmap PVModdedIconEngine::pixmap(const QSize &size, QIcon::Mode mode, QIcon::State state)
{
    QWindow* window = QGuiApplication::focusWindow();
    double divide_by = 1;
    if (window) {
        double primary_screen_device_pixel_ratio = QGuiApplication::primaryScreen()->devicePixelRatio();
        double current_screen_device_pixel_ratio = window->screen()->devicePixelRatio();
        if (current_screen_device_pixel_ratio != 0) {
            divide_by = primary_screen_device_pixel_ratio / current_screen_device_pixel_ratio;
        }
    }

    return _icons[(int)PVCore::PVTheme::is_color_scheme_dark()].pixmap(size / divide_by, mode, state);
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