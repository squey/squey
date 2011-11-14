#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <tbb/tick_count.h>

#define PICVIZ_DISCOVERY_NCHUNKS 1

PVRush::list_creators PVRush::PVSourceCreatorFactory::get_by_input_type(PVInputType_p in_t)
{
	QString itype = in_t->name();
	LIB_CLASS(PVRush::PVSourceCreator) &src_creators = LIB_CLASS(PVRush::PVSourceCreator)::get();
	LIB_CLASS(PVRush::PVSourceCreator)::list_classes const& list_creators = src_creators.get_list();
	LIB_CLASS(PVRush::PVSourceCreator)::list_classes::const_iterator itc;

	PVRush::list_creators lcreators_type;

	for (itc = list_creators.begin(); itc != list_creators.end(); itc++) {
		PVRush::PVSourceCreator_p sc = itc.value();
		if (sc->supported_type().compare(itype) != 0) {
			continue;
		}
		PVRush::PVSourceCreator_p sc_clone = sc->clone<PVRush::PVSourceCreator>();
		PVLOG_INFO("Found source for input type %s\n", qPrintable(in_t->human_name()));
		lcreators_type.push_back(sc_clone);
	}

	return lcreators_type;
}


PVRush::hash_format_creator PVRush::PVSourceCreatorFactory::get_supported_formats(list_creators const& lcr)
{
	list_creators::const_iterator it;
	hash_format_creator ret;
	for (it = lcr.begin(); it != lcr.end(); it++) {
		PVSourceCreator_p sc = *it;
		hash_formats const& src_formats = sc->get_supported_formats();
		PVLOG_INFO("Found %d formats.\n", src_formats.size());
		hash_formats::const_iterator it_sf;
		for (it_sf = src_formats.begin(); it_sf != src_formats.end(); it_sf++) {
			if (ret.contains(it_sf.key())) {
				PVLOG_WARN("Two source creators plugins using the same input type define the same format !\n");
				continue;
			}
			PVLOG_INFO("Found format %s\n", qPrintable(it_sf.key()));
			ret.insert(it_sf.key(), pair_format_creator(it_sf.value(), sc));
		}
	}

	return ret;
}

float PVRush::PVSourceCreatorFactory::discover_input(pair_format_creator format_, input_type input)
{
	PVFormat format = format_.first;
	tbb::tick_count start,end;
	start = tbb::tick_count::now();
	if (!format.populate()) {
		throw PVRush::PVFormatInvalid();
	}
	end = tbb::tick_count::now();
	PVLOG_INFO("(PVSourceCreatorFactory::discover_input) format population took %0.4f.\n", (end-start).seconds());
	PVSourceCreator_p sc = format_.second;

	PVFilter::PVChunkFilter_f chk_flt = format.create_tbb_filters();
	PVSourceCreator::source_p src = sc->create_discovery_source_from_input(input, format);

	size_t nelts = 0;
	size_t nelts_valid = 0;

	static size_t nelts_max = pvconfig.value("pvkernel/auto_discovery_number_elts", 500).toInt();

	for (int i = 0; i < PICVIZ_DISCOVERY_NCHUNKS; i++) {
		// Create a chunk
		PVCore::PVChunk* chunk = (*src)();
		if (chunk == NULL) { // No more chunks !
			break;
		}

		// Limit the number of elements filtered
		if (chunk->c_elements().size() + nelts > nelts_max) {
			PVCore::list_elts& l = chunk->elements();
			size_t new_size = nelts_max-nelts;
			PVLOG_INFO("(PVSourceCreatorFactory::discover_input) new chunk size %d.\n", new_size);
			l.resize(new_size);
		}

		// Apply the filter
		chk_flt(chunk);
		if (chunk->c_elements().size() == 0) {
			continue;
		}

		// Check the number of fields of the first element, and compare to the one
		// of the given format
		PVCol chunk_nfields = (*(chunk->c_elements().begin()))->c_fields().size();
		PVCol format_nfields = format.get_axes().size();
		if (chunk_nfields != format_nfields) {
		   PVLOG_INFO("For format %s, the number of fields after the normalization is %d, different of the number of axes of the format (%d).\n",	qPrintable(format.get_format_name()), chunk_nfields, format_nfields);
		   chunk->free();
		   return 0;
		}

		// Count the number of valid elts
		size_t chunk_nelts;
		size_t chunk_nelts_valid;
		chunk->get_elts_stat(chunk_nelts, chunk_nelts_valid);
		nelts += chunk_nelts;
		nelts_valid += chunk_nelts_valid;
		chunk->free();
		if (nelts >= nelts_max) {
			break;
		}
	}

	if (nelts == 0) {
		return 0;
	}

	end = tbb::tick_count::now();
	PVLOG_INFO("(PVSourceCreatorFactory::discover_input) format discovery took %0.4f.\n", (end-start).seconds());

	PVLOG_INFO("Number of elts valid/total: %d/%d\n", nelts_valid, nelts);

	return (float)nelts_valid/(float)nelts;
}
