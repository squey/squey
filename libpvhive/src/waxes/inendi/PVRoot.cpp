/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVHive.h>
#include <pvhive/waxes/inendi/PVRoot.h>

#include <inendi/PVRoot.h>
#include <inendi/PVView.h>

#include <iostream>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Inendi::PVRoot::select_view, root, args)
{
	about_to_refresh_observers(root->get_current_view_hive_property());
	about_to_refresh_observers(root->get_current_source_hive_property());
	about_to_refresh_observers(root->get_current_scene_hive_property());
	call_object_default<Inendi::PVRoot, FUNC(Inendi::PVRoot::select_view)>(root, args);
	refresh_observers(root->get_current_view_hive_property());
	refresh_observers(root->get_current_source_hive_property());
	refresh_observers(root->get_current_scene_hive_property());
}

IMPL_WAX(Inendi::PVRoot::select_source, root, args)
{
	about_to_refresh_observers(root->get_current_view_hive_property());
	about_to_refresh_observers(root->get_current_source_hive_property());
	about_to_refresh_observers(root->get_current_scene_hive_property());
	call_object_default<Inendi::PVRoot, FUNC(Inendi::PVRoot::select_source)>(root, args);
	refresh_observers(root->get_current_view_hive_property());
	refresh_observers(root->get_current_source_hive_property());
	refresh_observers(root->get_current_scene_hive_property());
}

IMPL_WAX(Inendi::PVRoot::select_scene, root, args)
{
	about_to_refresh_observers(root->get_current_view_hive_property());
	about_to_refresh_observers(root->get_current_source_hive_property());
	about_to_refresh_observers(root->get_current_scene_hive_property());
	call_object_default<Inendi::PVRoot, FUNC(Inendi::PVRoot::select_scene)>(root, args);
	refresh_observers(root->get_current_view_hive_property());
	refresh_observers(root->get_current_source_hive_property());
	refresh_observers(root->get_current_scene_hive_property());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
