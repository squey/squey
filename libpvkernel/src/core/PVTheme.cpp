/* MIT License
 *
 * © Squey, 2024
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <pvkernel/core/PVTheme.h>

#include <QApplication>
#include <QDBusConnection>
#include <QDBusInterface>
#include <QDBusReply>
#include <QGuiApplication>
#include <QStyleHints>
#include <QFile>

static constexpr const char* DBUS_NAME = "org.freedesktop.portal.Desktop";
static constexpr const char* DBUS_PATH = "/org/freedesktop/portal/desktop";
static constexpr const char* DBUS_INTERFACE = "org.freedesktop.portal.Settings";
static constexpr const char* DBUS_NAMESPACE = "org.freedesktop.appearance";
static constexpr const char* DBUS_METHOD = "ReadOne";
static constexpr const char* DBUS_SIGNAL = "SettingChanged";
static constexpr const char* DBUS_KEY = "color-scheme";

PVTheme::PVTheme()
{
    QString dark_theme = QString(std::getenv("DARK_THEME"));
    if (dark_theme != QString()) {
        _color_scheme = dark_theme == "true" ? EColorScheme::DARK : EColorScheme::LIGHT;
    }
    else {
        // Query DBUS
        QDBusInterface ifc(
            DBUS_NAME,
            DBUS_PATH,
            DBUS_INTERFACE,
            QDBusConnection::sessionBus()
        );
        if (ifc.isValid()) {
            QDBusReply<QVariant> reply = ifc.call(DBUS_METHOD, DBUS_NAMESPACE, DBUS_KEY);
            if (reply.isValid()) {
                _color_scheme = reply.value().value<uint>() == 1 ? EColorScheme::DARK : EColorScheme::LIGHT;
            }
        }
    }

    // Monitor system theme scheme changes
    QDBusConnection::sessionBus().connect(
        DBUS_NAME,
        DBUS_PATH,
        DBUS_INTERFACE,
        DBUS_SIGNAL,
        this,
        SLOT(setting_changed(QString, QString, QDBusVariant))
    );
    // See https://docs.flatpak.org/en/latest/portal-api-reference.html#gdbus-signal-org-freedesktop-portal-Settings.SettingChanged
}

PVTheme& PVTheme::get()
{
    static PVTheme instance;
    return instance;
}

PVTheme::EColorScheme PVTheme::color_scheme()
{
    return PVTheme::get()._color_scheme;
}

bool PVTheme::is_color_scheme_light() {
    return PVTheme::color_scheme() == PVTheme::EColorScheme::LIGHT;
}

bool PVTheme::is_color_scheme_dark() {
    return PVTheme::color_scheme() == PVTheme::EColorScheme::DARK;
}

void PVTheme::set_color_scheme(bool dark_theme)
{
    PVTheme::get()._color_scheme = dark_theme ? PVTheme::EColorScheme::DARK : PVTheme::EColorScheme::LIGHT ;
    PVTheme::apply_style(dark_theme);
    Q_EMIT PVTheme::get().color_scheme_changed(get()._color_scheme);
}

void PVTheme::set_color_scheme(PVTheme::EColorScheme color_scheme)
{
    PVTheme::get()._color_scheme = color_scheme;
    PVTheme::set_color_scheme(color_scheme == PVTheme::EColorScheme::DARK);
}

const char* PVTheme::color_scheme_name()
{
    return PVTheme::color_scheme() == PVTheme::EColorScheme::LIGHT ? "light" : "dark";
}

void PVTheme::follow_system_scheme(bool follow)
{
    PVTheme::get()._follow_system_scheme = follow;
}

void PVTheme::apply_style(bool dark_theme)
{
    QString css_theme(":/theme-light.qss");
    if (dark_theme) {
        css_theme = ":/theme-dark.qss";
    }
#ifdef SQUEY_DEVELOPER_MODE
    css_theme = SQUEY_SOURCE_DIRECTORY "/gui-qt/src/resources/" + css_theme.mid(2);
#endif
    QFile css_file(css_theme);
    css_file.open(QFile::ReadOnly);
    QTextStream css_stream(&css_file);
    QString css_string(css_stream.readAll());
    css_file.close();
    qApp->setStyleSheet(css_string);
}

void PVTheme::PVTheme::setting_changed(const QString& ns, const QString& key, const QDBusVariant& value)
{
    if (_follow_system_scheme and ns == DBUS_NAMESPACE and key == DBUS_KEY) {
        EColorScheme new_color_scheme = value.variant().value<uint32_t>() != 0 ? EColorScheme::DARK : EColorScheme::LIGHT;
        if (new_color_scheme != color_scheme()) {
            set_color_scheme(new_color_scheme);
        }
    }
}