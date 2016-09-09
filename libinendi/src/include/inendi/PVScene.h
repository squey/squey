/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSCENE_H
#define INENDI_PVSCENE_H

#include <QString>

#include <sigc++/sigc++.h>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <inendi/PVSource.h>

namespace Inendi
{

class PVRoot;

/**
 * \class PVScene
 */
class PVScene : public PVCore::PVDataTreeParent<PVSource, PVScene>,
                public PVCore::PVDataTreeChild<PVRoot, PVScene>
{
  public:
	PVScene(PVRoot& root, std::string const& scene_name);
	~PVScene();

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

	virtual std::string get_serialize_description() const { return get_name(); }

  public:
	inline void set_last_active_source(PVSource* src) { _last_active_src = src; }

  public:
	// Serialization
	static Inendi::PVScene& serialize_read(PVCore::PVSerializeObject& so, Inendi::PVRoot& parent);
	void serialize_write(PVCore::PVSerializeObject& so);

  public:
	sigc::signal<void> _project_updated;

  private:
	Inendi::PVSource* _last_active_src;

	std::string _name;
};
}

#endif /* INENDI_PVSCENE_H */
