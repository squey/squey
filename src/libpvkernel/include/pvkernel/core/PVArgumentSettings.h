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

#ifndef PVCORE_PVARGUMENTSETTINGS_H
#define PVCORE_PVARGUMENTSETTINGS_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/export.h>

#include <QString>

/**
 * Settings generated from the arguments of a plugin, for the user to choose
 * their defaults.
 */
namespace PVCore
{

/**
 * Declares a setting for each argument of \p keys that a setting can hold: a
 * boolean, an integer, a number, a string, a character, or a PVEnumType, whose
 * labels the setting holds. The setting is "<section>.<key>", its title the
 * description of the key, its default the value in \p args.
 *
 * \return The keys of the arguments that have a setting.
 */
PVKERNEL_EXPORT PVArgumentKeyList declare_argument_settings(QString const& section,
                                                            PVArgumentList const& args,
                                                            PVArgumentKeyList const& keys,
                                                            QString const& description = {});

/** Gives the arguments of \p args that have a setting in \p section its value. */
PVKERNEL_EXPORT void apply_argument_settings(QString const& section, PVArgumentList& args);

} // namespace PVCore

#endif // PVCORE_PVARGUMENTSETTINGS_H
