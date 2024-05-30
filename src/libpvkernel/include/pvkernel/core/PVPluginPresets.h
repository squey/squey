/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef PVCORE_PVPLUGIN_PRESETS_H
#define PVCORE_PVPLUGIN_PRESETS_H

#include <pvkernel/core/PVFunctionArgs.h>

namespace PVCore
{

namespace __impl
{

class PVPluginPresets
{
  public:
	PVPluginPresets(PVFunctionArgsBase* fargs, QString const& registered_name, QString const& path);

	QStringList list_presets() const;
	void del_preset(const QString& name) const;

	void add_preset(const QString& name) const;
	void load_preset(const QString& name);
	void modify_preset(const QString& name) const;
	void rename_preset(QString const& old_name, QString const& new_name) const;

	PVArgumentList get_args_for_preset() const;
	bool can_have_presets() const;

  private:
	PVFunctionArgsBase* _fargs;
	QString _abs_reg_name;
};
} // namespace __impl

template <class T>
class PVPluginPresets : public __impl::PVPluginPresets
{
  public:
	PVPluginPresets(T& o, QString const& path)
	    : __impl::PVPluginPresets(dynamic_cast<PVFunctionArgsBase*>(&o), o.registered_name(), path)
	{
	}
};
} // namespace PVCore

#endif // PVCORE_PVPLUGIN_PRESETS_H
