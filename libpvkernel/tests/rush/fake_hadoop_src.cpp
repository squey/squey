#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>

#include <QCoreApplication>
#include <QString>

#include "test-env.h"
#include "helpers.h"

#define MAX_LINE 1024*1024

void usage(char* path)
{
	std::cerr << "Usage: " << path << " host port task_nb final file format" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 5) {
		usage(argv[0]);
		return 1;
	}
	char* host = argv[1];
	uint16_t port = atoi(argv[2]);
	char* task_id = argv[3];
	bool final = (argv[4][0] == '1');
	if (!final && argc < 7) {
		usage(argv[0]);
		return 1;
	}
	
	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();
	PVRush::PVSourceCreator::source_p src;
	PVFilter::PVChunkFilter_f chk_flt;
	PVRush::PVFormat format;
	if (!final) {
		// Input file
		PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(argv[5]));
		QString path_format(argv[6]);
		format = PVRush::PVFormat("format", path_format);
		if (!format.populate(true)) {
			std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
			return 1;
		}
		chk_flt = format.create_tbb_filters();

		PVRush::PVSourceCreator_p sc_file;
		if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
			return 1;
		}

		// Process that file with the found source creator thanks to the extractor
		src = sc_file->create_source_from_input(file, format);
		if (!src) {
			std::cerr << "Unable to create PVRush source from file " << argv[1] << std::endl;
			return 1;
		}
	}

	struct hostent* hp;
	hp = gethostbyname(host);
	if (hp == NULL) {
		std::cerr << "Unable to resolve " << host << std::endl;
		return 1;
	}

	struct sockaddr_in sin;
	memset(&sin, 0, sizeof(sin));
	memcpy(&sin.sin_addr, hp->h_addr, hp->h_length);
	sin.sin_family = hp->h_addrtype;
	sin.sin_port = htons(port);

	int s = socket(AF_INET, SOCK_STREAM, 0);
	if (s <= 0) {
		std::cerr << "Unable to create a TCP socket !" << std::endl;
		return 1;
	}

	int r = connect(s, (struct sockaddr*) &sin, sizeof(struct sockaddr_in));
	if (r < 0) {
		std::cerr << "Unable to connect to " << host << ":" << port << std::endl;
		return 1;
	}

	// Write the task number
	write(s, task_id, strlen(task_id));
	write(s, "\n", 1);

	// Is that the final task ?
	write(s, final ? "1":"0", 1);
	write(s, "\n", 1);

	uint64_t off = 0;
	if (!final) {
		QString str_tmp;
		// Get chunks, filter them and write them down.
		PVCore::PVChunk* chunk;
		while ((chunk = (*src)()) != NULL) {
			chk_flt(chunk);
			PVCore::list_elts& elts = chunk->elements();
			PVCore::list_elts::iterator it;
			for (it = elts.begin(); it != elts.end(); it++) {
				// Write offset (0 for now)
				write(s, &off, sizeof(uint64_t));
				PVCore::list_fields& fields = (*it)->fields();
				// Compute the whole element size
				PVCore::list_fields::iterator it_f;
				uint32_t selt = 0;
				for (it_f = fields.begin(); it_f != fields.end(); it_f++) {
					selt += 4 + it_f->get_qstr(str_tmp).toUtf8().size();
				}
				write(s, &selt, sizeof(uint32_t));
				uint32_t nfields = fields.size();
				write(s, &nfields, sizeof(uint32_t));
				for (it_f = fields.begin(); it_f != fields.end(); it_f++) {
					QByteArray fba = it_f->get_qstr(str_tmp).toUtf8();
					//std::cout << fba.constData() << std::endl;
					uint32_t sfield = (uint32_t) fba.size();
					write(s, &sfield, sizeof(uint32_t));
					write(s, fba.constData(), sfield);
				}
			}
			chunk->free();
		}
	}

	close(s);

	return 0;
}
