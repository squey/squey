#include <pvkernel/core/PVSerializeArchiveOptions.h>


PVCore::PVSerializeObject_p PVCore::PVSerializeArchiveOptions::create_object(QString const& name, PVSerializeObject_p parent)
{
	PVSerializeObject_p ret(allocate_object(name, parent));
	return ret;
}


bool PVCore::PVSerializeArchiveOptions::must_write(PVSerializeObject const& parent, QString const& child)
{
	QString path = get_object_logical_path(parent);
	assert(_objects.contains(path));

	PVSerializeObject_p ours(_objects.value(path));
	PVSerializeObject_p our_child = ours->get_child_by_name(child);
	assert(our_child);
	assert(our_child->is_optional());
	return our_child->must_write();
}
