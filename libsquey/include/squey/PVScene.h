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

#ifndef SQUEY_PVSCENE_H
#define SQUEY_PVSCENE_H

#include <squey/PVSource.h> // for PVSource

#include <pvkernel/core/PVDataTreeObject.h> // for PVDataTreeParent, etc

#include <QString>

#include <sigc++/sigc++.h>

#include <algorithm> // for forward
#include <string>    // for string

namespace Squey
{
class PVRoot;
} // namespace Squey
namespace Squey
{
class PVView;
} // namespace Squey
namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore

namespace Squey
{
/**
 * \class PVScene
 */
class PVScene : public PVCore::PVDataTreeParent<PVSource, PVScene>,
                public PVCore::PVDataTreeChild<PVRoot, PVScene>
{
  public:
	PVScene(PVRoot& root, std::string const& scene_name);
	virtual ~PVScene();

  public:
	template <class... T>
	PVSource& emplace_add_child(T&&... t)
	{
		auto& src =
		    PVCore::PVDataTreeParent<PVSource, PVScene>::emplace_add_child(std::forward<T>(t)...);
		_project_updated.emit();
		return src;
	}

	void set_name(std::string name)
	{
		_name = name;
		_project_updated.emit();
	}
	const std::string& get_name() const { return _name; }

	PVSource* current_source();
	PVSource const* current_source() const;

	PVView* current_view();
	PVView const* current_view() const;

	inline PVSource* last_active_source() { return _last_active_src; }
	inline PVSource const* last_active_source() const { return _last_active_src; }

  public:
	inline bool is_empty() const { return size() == 0; }

	std::string get_serialize_description() const override { return get_name(); }

  public:
	inline void set_last_active_source(PVSource* src) { _last_active_src = src; }

  public:
	// Serialization
	static Squey::PVScene& serialize_read(PVCore::PVSerializeObject& so, Squey::PVRoot& parent);
	void serialize_write(PVCore::PVSerializeObject& so) const;

  public:
	sigc::signal<void()> _project_updated;

  private:
	Squey::PVSource* _last_active_src;

	std::string _name;
};
} // namespace Squey

#endif /* SQUEY_PVSCENE_H */
