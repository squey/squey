/* * MIT License
 *
 * © Squey, 2026
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

// That the scheme follows the one the settings portal signals, while it follows the system.
// QtDBus delivers the signal to the slot it finds in the meta-object by its signature, as
// the calls below do.

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVTheme.h>
#include <pvkernel/core/squey_assert.h>

#include <QApplication>
#include <QDBusVariant>
#include <QDir>
#include <QFile>
#include <QMetaObject>
#include <QMetaType>
#include <QString>
#include <QVariant>

#include <cstdint>

static bool portal_signals(uint32_t scheme)
{
	return QMetaObject::invokeMethod(
	    &PVCore::PVTheme::get(), "setting_changed", Qt::DirectConnection,
	    Q_ARG(QString, "org.freedesktop.appearance"), Q_ARG(QString, "color-scheme"),
	    Q_ARG(QDBusVariant, QDBusVariant(QVariant::fromValue(scheme))));
}

int main(int argc, char** argv)
{
	QApplication app(argc, argv);
	// Without a configuration of the user, PVConfig copies the one Squey ships, which a test
	// finds nowhere: the test gives it, in the directory of its own CTest sets.
	PV_VALID(QDir().mkpath(PVCore::PVConfig::user_dir()), true);
	PV_VALID(QFile::exists(PVCore::PVConfig::user_path()) or
	             QFile::copy(QString::fromLocal8Bit(argv[1]), PVCore::PVConfig::user_path()),
	         true);
	// As QtDBus does once connected to a bus, which the test may not have.
	qRegisterMetaType<QDBusVariant>();

	PVCore::PVTheme::follow_system_scheme(true);
	PVCore::PVTheme::set_color_scheme(PVCore::PVTheme::EColorScheme::DARK);
	// 1 prefers dark, 2 light.
	PV_ASSERT_VALID(portal_signals(2));
	PV_ASSERT_VALID(PVCore::PVTheme::is_color_scheme_light());
	PV_ASSERT_VALID(portal_signals(1));
	PV_ASSERT_VALID(PVCore::PVTheme::is_color_scheme_dark());

	// The scheme the user chose stays.
	PVCore::PVTheme::follow_system_scheme(false);
	PV_ASSERT_VALID(portal_signals(2));
	PV_ASSERT_VALID(PVCore::PVTheme::is_color_scheme_dark());

	return 0;
}
