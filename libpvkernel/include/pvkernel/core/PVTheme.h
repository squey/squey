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
#ifndef __PVTHEME_H__
#define __PVTHEME_H__

#include <pvbase/general.h>

#include <QObject>

class QDBusVariant;


class PVTheme : public QObject
{
    Q_OBJECT

public:
    enum class EColorScheme {
        LIGHT = 0,
        DARK = 1
    };

private:
    PVTheme();

public:
    static PVTheme& get();

    static EColorScheme color_scheme();
    static bool is_color_scheme_light();
    static bool is_color_scheme_dark();
    static void set_color_scheme(bool dark_theme);
    static void set_color_scheme(EColorScheme color_scheme);
    static const char* color_scheme_name();
    static void follow_system_scheme(bool follow);

private:
    static void apply_style(bool dark_theme);

Q_SIGNALS:
    void color_scheme_changed(EColorScheme color_scheme);


public Q_SLOTS:
    void setting_changed(const QString& ns, const QString& key, const QDBusVariant& value);

private:
    PVTheme::EColorScheme _color_scheme;
    bool _follow_system_scheme = true;
};

#endif // __PVTHEME_H__