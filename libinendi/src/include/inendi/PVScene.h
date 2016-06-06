/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSCENE_H
#define INENDI_PVSCENE_H

#include <QString>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <inendi/PVPtrObjects.h>
#include <inendi/PVSource_types.h>
#include <inendi/PVSource.h>
#include <inendi/PVView_types.h>
#include <inendi/PVScene_types.h>

#define INENDI_SCENE_ARCHIVE_EXT "pv"
#define INENDI_SCENE_ARCHIVE_FILTER "INENDI project files (*." INENDI_SCENE_ARCHIVE_EXT ")"

namespace Inendi
{

class PVSource;

/**
 * \class PVScene
 */
class PVScene : public PVCore::PVDataTreeParent<PVSource, PVScene>,
                public PVCore::PVDataTreeChild<PVRoot, PVScene>,
                public PVCore::PVEnableSharedFromThis<PVScene>
{
	friend class PVCore::PVSerializeObject;
	friend class PVRoot;
	friend class PVSource;
	friend class PVView;

  public:
	typedef QList<PVSource const*> list_sources_t;

  public:
	PVScene(PVRoot* root, QString scene_name);
	~PVScene();

  public:
	void set_name(QString name) { _name = name; }
	const QString& get_name() const { return _name; }

	PVSource* current_source();
	PVSource const* current_source() const;

	PVView* current_view();
	PVView const* current_view() const;

	inline PVSource* last_active_source() { return _last_active_src; }
	inline PVSource const* last_active_source() const { return _last_active_src; }

  public:
	list_sources_t get_sources(PVRush::PVInputType const& type) const;
	PVRush::PVInputType::list_inputs_desc get_inputs_desc(PVRush::PVInputType const& type) const;

	inline bool is_empty() const { return get_children().size() == 0; }

	virtual QString get_serialize_description() const { return get_name(); }

  protected:
	virtual QString get_children_description() const { return "Source(s)"; }
	virtual QString get_children_serialize_name() const { return "sources"; }

	QList<PVRush::PVInputType_p> get_all_input_types() const;

	inline void set_last_active_source(PVSource* src) { _last_active_src = src; }

  protected:
	// Serialization
	void serialize_read(PVCore::PVSerializeObject& so);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

  private:
	Inendi::PVSource* _last_active_src;

	QString _name;
};

using PVScene_p = PVCore::PVSharedPtr<PVScene>;
}

#endif /* INENDI_PVSCENE_H */
