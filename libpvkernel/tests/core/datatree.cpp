/**
 * \file datatree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVDataTreeObject.h>

/******************************************************************************
 *
 * Tree classes declaration (PVScene -> PVSource -> PVMapped -> PVPlotted)
 *
 *****************************************************************************/
// forward declarations
class PVPlotted;
class PVMapped;
class PVSource;
class PVScene;

typedef typename PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<PVScene>, PVSource> data_tree_scene_t;
class PVScene : public data_tree_scene_t
{
public:
	PVScene(PVCore::PVDataTreeNoParent<PVScene>* parent = NULL) : data_tree_scene_t(parent) {};
};

typedef typename PVCore::PVDataTreeObject<PVScene, PVMapped> data_tree_source_t;
class PVSource : public data_tree_source_t
{
public:
	PVSource(PVScene* parent = NULL) : data_tree_source_t(parent) {};
};

typedef typename PVCore::PVDataTreeObject<PVSource, PVPlotted> data_tree_mapped_t;
class PVMapped : public data_tree_mapped_t
{
public:
	PVMapped(PVSource* parent = NULL) : data_tree_mapped_t(parent) {};
};

typedef typename PVCore::PVDataTreeObject<PVMapped, PVCore::PVDataTreeNoChildren<PVPlotted>> data_tree_plotted_t;
class PVPlotted : public data_tree_plotted_t
{
public:
	PVPlotted(PVMapped* parent = NULL) : data_tree_plotted_t(parent) {};
};


class Test{};
/******************************************************************************
 *
 * Test case
 *
 *****************************************************************************/
int main()
{
	// Construct data tree
	PVScene* scene1 = new PVScene();
	PVScene* scene2 = new PVScene();
	PVSource* source = new PVSource(scene1);
	PVSource* source2 = new PVSource(scene1);
	PVMapped* mapped = new PVMapped(source);
	PVPlotted* plotted = new PVPlotted(mapped);

	// Dump scene
	std::cout << "=SCENE DUMP=" << std::endl;
	std::cout << "scene1:" << std::endl;
	scene1->dump();
	std::cout << "scene2:" << std::endl;
	scene2->dump();
	std::cout << std::endl;

	// Parent access
	std::cout << "=PARENT ACCESS=" << std::endl;
	auto scene_parent = scene1->get_parent();
	auto source_parent = source->get_parent<PVScene>();
	auto mapped_parent = mapped->get_parent();
	auto mapped_scene_parent = mapped->get_parent<PVScene>();
	auto plotted_parent = plotted->get_parent();
	auto plotted_parent2 = plotted->get_parent<PVMapped>();
	auto plotted_parent_parent = plotted->get_parent<PVSource>();
	auto plotted_parent_parent_parent = plotted->get_parent<PVScene>();
	std::cout << "scene=" << std::hex << scene1 << std::endl;
	std::cout << "scene->get_parent()=" << std::hex << scene_parent << std::endl;
	std::cout << "source="  << std::hex << source << std::endl;
	std::cout << "source->get_parent()=" << std::hex << source_parent << std::endl;
	std::cout << "mapped="  << std::hex << mapped << std::endl;
	std::cout << "mapped.get_parent()=" << std::hex << mapped_parent << std::endl;
	std::cout << "mapped.get_parent<PVScene>()=" << std::hex << mapped_scene_parent << std::endl;
	std::cout << "plotted="  << std::hex << plotted << std::endl;
	std::cout << "plotted->get_parent()=" << std::hex << plotted_parent << std::endl;
	std::cout << "plotted->get_parent<PVMapped>()=" << std::hex << plotted_parent2 << std::endl;
	std::cout << "plotted->get_parent<PVSource>()=" << std::hex << plotted_parent_parent << std::endl;
	std::cout << "plotted->get_parent<PVScene>()=" << std::hex << plotted_parent_parent_parent << std::endl;

	// Children access
	std::cout << std::endl << "=CHILDREN ACCESS=" << std::endl;
	std::cout << "scene1->get_children()=";
	scene1->dump_children();
	std::cout << "scene1->get_children<PVSource>()=";
	scene1->dump_children<PVSource>();
	std::cout << "scene1->get_children<PVMapped>()=";
	scene1->dump_children<PVMapped>();
	std::cout << "scene1->get_children<PVPlotted>()=";
	scene1->dump_children<PVPlotted>();

	// Reparenting
	std::cout << std::endl << "=REPARENTING=" << std::endl;
	source->set_parent(scene2);
	std::cout << "scene1:" << std::endl;
	scene1->dump();
	std::cout << "scene2:" << std::endl;
	scene2->dump();

	// Changing child
	std::cout << std::endl << "=CHANGING CHILD=" << std::endl;
	source2->set_parent(scene1);
	source2->add_child(mapped);
	std::cout << "scene1:" << std::endl;
	scene1->dump();
	std::cout << "scene2:" << std::endl;
	scene2->dump();

	// Delete the whole hierarchy
	std::cout << std::endl << "=HIERARCHY DESTRUCTION=" << std::endl;
	delete source; // << we can safely delete intermediary part of the hierarchy
	delete scene1;
	delete scene2;
}
