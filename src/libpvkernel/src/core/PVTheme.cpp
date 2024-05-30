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

#include <pvkernel/core/PVConfig.h>

#include <QApplication>
#include <QDBusConnection>
#include <QDBusInterface>
#include <QDBusReply>
#include <QGuiApplication>
#include <QStyleHints>
#include <QFile>
#include <QSettings>

static constexpr const char* DBUS_NAME = "org.freedesktop.portal.Desktop";
static constexpr const char* DBUS_PATH = "/org/freedesktop/portal/desktop";
static constexpr const char* DBUS_INTERFACE = "org.freedesktop.portal.Settings";
static constexpr const char* DBUS_NAMESPACE = "org.freedesktop.appearance";
static constexpr const char* DBUS_METHOD = "ReadOne";
static constexpr const char* DBUS_SIGNAL = "SettingChanged";
static constexpr const char* DBUS_KEY = "color-scheme";

static constexpr const char* COLOR_SCHEME_SETTINGS_KEY = "gui/theme_scheme";

const QColor PVCore::PVTheme::link_colors[] = { 0x1a72bb, 0x1b98ff };

PVCore::PVTheme::PVTheme()
{
    QString dark_theme = QString(std::getenv("DARK_THEME"));
    if (dark_theme != QString()) {
        _color_scheme = dark_theme == "true" ? EColorScheme::DARK : EColorScheme::LIGHT;
    }
    else {
        _color_scheme = system_color_scheme();
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

PVCore::PVTheme::EColorScheme PVCore::PVTheme::system_color_scheme()
{
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
            return reply.value().value<uint>() == 1 ? EColorScheme::DARK : EColorScheme::LIGHT;
        }
    }

    return  EColorScheme::UNKNOWN;
}


PVCore::PVTheme& PVCore::PVTheme::get()
{
    static PVTheme instance;
    return instance;
}

void PVCore::PVTheme::init()
{
	const QString& settings_color_scheme = PVCore::PVTheme::settings_color_scheme();
	if (settings_color_scheme == "light") {
		PVCore::PVTheme::set_color_scheme(PVCore::PVTheme::EColorScheme::LIGHT);
	}
	else if (settings_color_scheme == "system") {
		PVCore::PVTheme::follow_system_scheme(true);
        PVCore::PVTheme::set_color_scheme(system_color_scheme());
	}
	else {
		PVCore::PVTheme::set_color_scheme(PVCore::PVTheme::EColorScheme::DARK);
	}
}

PVCore::PVTheme::EColorScheme PVCore::PVTheme::color_scheme()
{
    return PVCore::PVTheme::get()._color_scheme;
}

bool PVCore::PVTheme::is_color_scheme_light() {
    return PVCore::PVTheme::color_scheme() == PVCore::PVTheme::EColorScheme::LIGHT;
}

bool PVCore::PVTheme::is_color_scheme_dark() {
    return PVCore::PVTheme::color_scheme() == PVCore::PVTheme::EColorScheme::DARK;
}

void PVCore::PVTheme::set_color_scheme(bool dark_theme)
{
    PVCore::PVTheme::get()._color_scheme = dark_theme ? PVCore::PVTheme::EColorScheme::DARK : PVCore::PVTheme::EColorScheme::LIGHT ;
    PVCore::PVConfig::set_value(COLOR_SCHEME_SETTINGS_KEY, PVCore::PVTheme::get()._follow_system_scheme ? "system" : color_scheme_name());
    PVCore::PVTheme::apply_style(dark_theme);
    Q_EMIT PVCore::PVTheme::get().color_scheme_changed(get()._color_scheme);
}

QString PVCore::PVTheme::settings_color_scheme()
{
    QString settings_scheme_color = PVCore::PVConfig::value(COLOR_SCHEME_SETTINGS_KEY).toString();
    if (settings_scheme_color.isEmpty()) {
        settings_scheme_color = "dark"; // default
    }
    return settings_scheme_color;
}

void PVCore::PVTheme::set_color_scheme(PVCore::PVTheme::EColorScheme color_scheme)
{
    PVCore::PVTheme::get()._color_scheme = color_scheme;
    PVCore::PVTheme::set_color_scheme(color_scheme == PVCore::PVTheme::EColorScheme::DARK);
}

const char* PVCore::PVTheme::color_scheme_name()
{
    return PVCore::PVTheme::color_scheme() == PVCore::PVTheme::EColorScheme::LIGHT ? "light" : "dark";
}

void PVCore::PVTheme::follow_system_scheme(bool follow)
{
    PVCore::PVTheme::get()._follow_system_scheme = follow;
}

void PVCore::PVTheme::apply_style(bool dark_theme)
{
    QString css_theme(":/theme-light.qss");
    if (dark_theme) {
        css_theme = ":/theme-dark.qss";
    }
//#ifdef SQUEY_DEVELOPER_MODE
//    css_theme = SQUEY_SOURCE_DIRECTORY "/gui-qt/src/resources/" + css_theme.mid(2);
//#endif
    QFile css_file(css_theme);
    css_file.open(QFile::ReadOnly);
    QTextStream css_stream(&css_file);
    QString css_string(css_stream.readAll());
    css_file.close();
    qApp->setStyleSheet(css_string);
}

void PVCore::PVTheme::setting_changed(const QString& ns, const QString& key, const QDBusVariant& value)
{
    if (_follow_system_scheme and ns == DBUS_NAMESPACE and key == DBUS_KEY) {
        EColorScheme new_color_scheme = value.variant().value<uint32_t>() != 0 ? EColorScheme::DARK : EColorScheme::LIGHT;
        if (new_color_scheme != color_scheme()) {
            set_color_scheme(new_color_scheme);
        }
    }
}