/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVConfig.h>

#include <tbb/tick_count.h>

static constexpr int INENDI_DISCOVERY_NCHUNKS = 1;

PVRush::PVSourceCreator_p PVRush::PVSourceCreatorFactory::get_by_input_type(PVInputType_p in_t)
{
	QString itype = in_t->name();
	LIB_CLASS(PVRush::PVSourceCreator)& src_creators = LIB_CLASS(PVRush::PVSourceCreator)::get();
	LIB_CLASS(PVRush::PVSourceCreator)::list_classes const& list_creators = src_creators.get_list();
	LIB_CLASS(PVRush::PVSourceCreator)::list_classes::const_iterator itc;

	for (itc = list_creators.begin(); itc != list_creators.end(); itc++) {
		PVRush::PVSourceCreator_p sc = itc->value();
		if (sc->supported_type().compare(itype) != 0) {
			continue;
		}
		PVRush::PVSourceCreator_p sc_clone = sc->clone<PVRush::PVSourceCreator>();
		PVLOG_INFO("Found source for input type %s\n", qPrintable(in_t->human_name()));
		return sc_clone;
	}

	return PVRush::PVSourceCreator_p();
}

float PVRush::PVSourceCreatorFactory::discover_input(pair_format_creator format_,
                                                     PVInputDescription_p input,
                                                     bool* cancellation)
{
	PVFormat format = format_.first;
	PVSourceCreator_p sc = format_.second;

	tbb::tick_count start, end;
	start = tbb::tick_count::now();

	try {
		PVFilter::PVChunkFilterByEltCancellable chk_flt =
		    format.create_tbb_filters_autodetect(1.0, cancellation);
		PVSourceCreator::source_p src = sc->create_source_from_input(input);

		if (src == nullptr) {
			return 0.f;
		}

		src->set_number_cols_to_reserve(PVCol(format.get_axes().size()));

		size_t nelts = 0;
		size_t nelts_valid = 0;

		QSettings& pvconfig = PVCore::PVConfig::get().config();

		static size_t nelts_max =
		    pvconfig.value("pvkernel/auto_discovery_number_elts", 500).toInt();

		for (int i = 0; i < INENDI_DISCOVERY_NCHUNKS; i++) {
			// Create a chunk
			PVCore::PVTextChunk* chunk = dynamic_cast<PVCore::PVTextChunk*>((*src)());
			if (chunk == nullptr) { // No more chunks !
				break;
			}

			// Limit the number of elements filtered
			if (chunk->c_elements().size() + nelts > nelts_max) {
				PVCore::list_elts& l = chunk->elements();
				size_t new_size = nelts_max - nelts;
				PVLOG_DEBUG("(PVSourceCreatorFactory::discover_input) new chunk size %d.\n",
				            new_size);
				// Free the elements that we are going to remove
				auto it_elt = l.begin();
				std::advance(it_elt, new_size);
				for (; it_elt != l.end(); it_elt++) {
					PVCore::PVElement::free(*it_elt);
				}
				it_elt = l.begin();
				std::advance(it_elt, new_size);
				l.erase(it_elt, l.end());
			}

			// Apply the filter
			chk_flt(chunk);
			if (chunk->c_elements().size() == 0) {
				chunk->free();
				continue;
			}

			// Check the number of fields of the first element, and compare to the one
			// of the given format
			PVCol chunk_nfields((*(chunk->c_elements().begin()))->c_fields().size());
			PVCol format_nfields(format.get_axes().size());
			if (chunk_nfields != format_nfields) {
				PVLOG_DEBUG("For format %s, the number of fields after the normalization is %d, "
				            "different of the number of axes of the format (%d).\n",
				            qPrintable(format.get_format_name()), chunk_nfields, format_nfields);
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
		PVLOG_INFO("Discovery with format %s took %0.4f, %d/%d elements are valid.\n",
		           qPrintable(format.get_format_name()), (end - start).seconds(), nelts_valid,
		           nelts);

		return (float)nelts_valid / (float)nelts;
	} catch (PVFilter::PVFieldsFilterInvalidArguments const&) {

		// Formats with filters containing invalid arguments are not candidates to auto discovery
		return 0.0;
	}
}
