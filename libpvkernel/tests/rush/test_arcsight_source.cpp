#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include "../../plugins/common/arcsight/PVArcsightQuery.h"

#include "helpers.h"
#include "test-env.h"

#include <iostream>
#include <QCoreApplication>

int main(int argc, char** argv)
{
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " IP port login password [query]" << std::endl;
		return 1;
	}

	QCoreApplication app(argc, argv);
	init_env();

	PVRush::PVPluginsLoad::load_all_plugins();

	const char* host = argv[1];
	uint16_t port = atoi(argv[2]);
	const char* user = argv[3];
	const char* pwd = argv[4];

	const char* query = "";
	if (argc >= 6) {
		query = argv[5];
	}

	PVRush::PVArcsightInfos infos;
	infos.set_host(host);
	infos.set_port(port);
	infos.set_username(user);
	infos.set_password(pwd);
	
	PVRush::PVArcsightQuery* as_query = new PVRush::PVArcsightQuery(infos);
	PVRush::PVArcsightQuery::fields_indexes_t& fields_kept = as_query->get_fields_kept();
	fields_kept.insert(0);
	fields_kept.insert(1);
	as_query->set_query(query);

	PVRush::PVInputDescription_p ind(as_query);

	PVRush::PVSourceCreator_p sc = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("arcsight");
	PVRush::PVFormat format;
	PVRush::PVSourceCreator::source_p src = sc->create_source_from_input(ind, format);

	PVCore::PVChunk* chunk;
	while ((chunk = src->operator()())) {
		dump_chunk_csv(*chunk);
		chunk->free();
	}
	
	return 0;
}
