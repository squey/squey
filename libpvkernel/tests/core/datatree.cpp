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

class PVScene : public PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<PVScene>, PVSource>
{
public:
	PVScene(PVCore::PVDataTreeNoParent<PVScene>* parent = NULL) : PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<PVScene>, PVSource>(parent) {};
};

class PVSource : public PVCore::PVDataTreeObject<PVScene, PVMapped>
{
public:
	PVSource(PVScene* parent = NULL) : PVCore::PVDataTreeObject<PVScene, PVMapped>(parent) {};
};

class PVMapped : public PVCore::PVDataTreeObject<PVSource, PVPlotted>
{
public:
	PVMapped(PVSource* parent = NULL) : PVCore::PVDataTreeObject<PVSource, PVPlotted>(parent) {};
};

class PVPlotted : public PVCore::PVDataTreeObject<PVMapped, PVCore::PVDataTreeNoChildren<PVPlotted> >
{
public:
	PVPlotted(PVMapped* parent = NULL) : PVCore::PVDataTreeObject<PVMapped, PVCore::PVDataTreeNoChildren<PVPlotted>>(parent) {};
};


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
	PVMapped* mapped = new PVMapped(source);
	PVPlotted* plotted = new PVPlotted(mapped);

	// Test parent access
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

	// Dump scene
	std::cout << std::endl << "=SCENE DUMP=" << std::endl;
	std::cout << "scene1:" << std::endl;
	scene1->dump();
	std::cout << "scene2:" << std::endl;
	scene2->dump();

	// Reparenting
	std::cout << std::endl << "=REPARENTING=" << std::endl;

	source->set_parent(scene2);
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
