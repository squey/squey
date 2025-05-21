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
#include <qdbusextratypes.h>
#include <qobjectdefs.h>
#include <qtextstream.h>
#include <qvariant.h>
#include <stdint.h>
#include <sys/types.h>
#include <QApplication>
#include <QDBusConnection>
#include <QDBusInterface>
#include <QDBusReply>
#include <QFile>
#include <QPalette>
#include <QWidget>
#include <QStyle>

#include <cstdlib>

#ifdef __linux__
static constexpr const char* DBUS_NAME = "org.freedesktop.portal.Desktop";
static constexpr const char* DBUS_PATH = "/org/freedesktop/portal/desktop";
static constexpr const char* DBUS_INTERFACE = "org.freedesktop.portal.Settings";
static constexpr const char* DBUS_NAMESPACE = "org.freedesktop.appearance";
static constexpr const char* DBUS_METHOD = "ReadOne";
static constexpr const char* DBUS_SIGNAL = "SettingChanged";
static constexpr const char* DBUS_KEY = "color-scheme";
#elifdef _WIN32
#include <windows.h>
#include <thread>
static constexpr const wchar_t* THEME_REG_KEY = L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize";
#endif

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
#ifdef __linux__
    QDBusConnection::sessionBus().connect(
        DBUS_NAME,
        DBUS_PATH,
        DBUS_INTERFACE,
        DBUS_SIGNAL,
        this,
        SLOT(setting_changed(QString, QString, QDBusVariant))
    );
    // See https://docs.flatpak.org/en/latest/portal-api-reference.html#gdbus-signal-org-freedesktop-portal-Settings.SettingChanged
#elifdef _WIN32
    std::thread theme_change_monitoring_thread([this](){
        HKEY hKey;
        HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        if (RegOpenKeyExW(HKEY_CURRENT_USER, THEME_REG_KEY, 0, KEY_NOTIFY, &hKey) != ERROR_SUCCESS) {
            pvlogger::error() << "Failed to monitor system theme changes" << std::endl;
            CloseHandle(hEvent);
            return;
        }

        while (true) {
            if (RegNotifyChangeKeyValue(hKey, FALSE, REG_NOTIFY_CHANGE_LAST_SET, hEvent, TRUE) == ERROR_SUCCESS) {
                if (WaitForSingleObject(hEvent, INFINITE) == WAIT_OBJECT_0) {
                    QMetaObject::invokeMethod(this, &PVTheme::setting_changed, Qt::QueuedConnection);
                }
            } else {
                pvlogger::error() << "Failed to read THEME_REG_KEY" << std::endl;
                break;
            }
        }

        RegCloseKey(hKey);
        CloseHandle(hEvent);
    });
    theme_change_monitoring_thread.detach();   
#endif
}

PVCore::PVTheme::EColorScheme PVCore::PVTheme::system_color_scheme()
{
#ifdef __linux__
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
#elifdef _WIN32
    HKEY hKey;
    DWORD value = 0;
    DWORD size = sizeof(value);
    
    if (RegOpenKeyExW(HKEY_CURRENT_USER, THEME_REG_KEY, 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        if (RegQueryValueExW(hKey, L"AppsUseLightTheme", nullptr, nullptr, reinterpret_cast<LPBYTE>(&value), &size) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return value == 0 ? EColorScheme::DARK : EColorScheme::LIGHT;  // 0 = Dark Mode, 1 = Light Mode
        }
        RegCloseKey(hKey);
    }
#endif

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
    PVCore::PVTheme::get().apply_style(dark_theme);
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

QColor light_alternate_base_color();

void PVCore::PVTheme::apply_style(bool dark_theme)
{
    auto load_stylesheet = [](const QString& css_theme){
        QFile css_file(css_theme);
        css_file.open(QFile::ReadOnly);
        QTextStream css_stream(&css_file);
        QString css_string(css_stream.readAll());
        css_file.close();
        return css_string;
    };
    QString css_theme(":/theme-light.qss");

    if (_init) {
#ifdef __APPLE__
        PVCore::PVTheme::get().light_base_color = QPalette().color(QPalette::AlternateBase).lighter(150);
        PVCore::PVTheme::get().light_alternate_base_color = QPalette().color(QPalette::BrightText).lighter(500);
#else
        QString css_string = load_stylesheet(css_theme);
        QWidget dummy_widget;
        dummy_widget.hide();
        dummy_widget.setStyleSheet(css_string);
        PVCore::PVTheme::get().light_base_color = dummy_widget.style()->standardPalette().color(QPalette::Base);
        PVCore::PVTheme::get().light_alternate_base_color = dummy_widget.style()->standardPalette().color(QPalette::AlternateBase);
#endif
        _init = false;
    }

    if (dark_theme) {
        css_theme = ":/theme-dark.qss";
    }
    
//#ifdef SQUEY_DEVELOPER_MODE
//    css_theme = SQUEY_SOURCE_DIRECTORY "/gui-qt/src/resources/" + css_theme.mid(2);
//#endif
    QString css_string = load_stylesheet(css_theme);
    qApp->setStyleSheet(css_string);
}

#ifdef __linux__
void PVCore::PVTheme::setting_changed(const QString& ns, const QString& key, const QDBusVariant& value)
{
    if (_follow_system_scheme and ns == DBUS_NAMESPACE and key == DBUS_KEY) {
        EColorScheme new_color_scheme = value.variant().value<uint32_t>() != 0 ? EColorScheme::DARK : EColorScheme::LIGHT;
        if (new_color_scheme != color_scheme()) {
            set_color_scheme(new_color_scheme);
        }
    }
}
#elifdef _WIN32
void PVCore::PVTheme::setting_changed()
{
    set_color_scheme(system_color_scheme());
}
#endif