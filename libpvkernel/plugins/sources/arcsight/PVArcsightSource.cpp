#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

#include "PVArcsightSource.h"
#include "../../common/arcsight/PVArcsightQuery.h"

#include <tbb/atomic.h>

#include <string>

template <class V, class I, class F>
void visit_vector_with_indexes(V const& v, I const& idxes, F const& f)
{
	if (idxes.size() == 0) {
		for (auto const& i: v) {
			f(i);
		}
	}
	else {
		for (auto i: idxes) {
			f(v[i]);
		}
	}
}

PVRush::PVArcsightSource::PVArcsightSource(PVInputDescription_p input, size_t min_chunk_size, PVFilter::PVChunkFilter_f src_filter):
	PVRawSourceBase(src_filter),
	_min_chunk_size(min_chunk_size),
	_next_index(0)
{
	PVArcsightQuery* query = dynamic_cast<PVArcsightQuery*>(input.get());
	assert(query);

	_query = *query;

	PVArcsightInfos const& infos = query->get_infos();
	PVLOG_INFO("Create arcsight source with: host=%s, port=%d, username=%s, password=%s, query=%s, start=%li, end=%li\n",
		qPrintable(infos.get_host()), infos.get_port(), qPrintable(infos.get_username()), qPrintable(infos.get_password()),
		qPrintable(query->get_query()), query->get_start_ms(), query->get_end_ms());
	try {
		_arcsight.login(qPrintable(infos.get_host()), infos.get_port(),
				infos.get_username().toStdString(), infos.get_password().toStdString());
		_arcsight_search = _arcsight.search(qPrintable(query->get_query()), query->get_start_ms(), query->get_end_ms());
	}
	catch (arcsight::Exception const& e) {
		throw PVArcsightInputException(e.what());
	}
}

PVRush::PVArcsightSource::~PVArcsightSource()
{
}

QString PVRush::PVArcsightSource::human_name()
{
	return QString("python:toto.pl");
}

void PVRush::PVArcsightSource::seek_begin()
{
	// TODO: implement this!
}

void PVRush::PVArcsightSource::prepare_for_nelts(chunk_index /*nelts*/)
{
}

PVCore::PVChunk* PVRush::PVArcsightSource::operator()()
{
	PVCore::PVChunk* chunk;
	try {
		if (!_arcsight_search.has_more_tuples()) {
			// That's the end
			return NULL;
		}

		// TODO: optimise this so that one allocation is done for the chunk, and
		// data is "memcpy"'ed in the chunk. UTF8 conversion should also be done
		// thanks to ICU (faster and save allocations).
		chunk = PVCore::PVChunkMem<>::allocate(0, this);
		chunk->set_index(_next_index);

		_arcsight_search.get_next_tuples(_min_chunk_size, 1500);
		_arcsight_search.visit_last_tuples([&](std::vector<std::string> const& elt_strs) {
			PVCore::PVElement* elt = chunk->add_element();
			elt->fields().clear();

			if (elt_strs.size() == 0) {
				return;
			}
			visit_vector_with_indexes(elt_strs, _query.get_fields_kept(), [&elt](std::string const& s) {
				QString value(QString::fromUtf8(s.c_str(), s.size()));
				PVCore::PVField f(*elt);
				size_t size_buf = value.size() * sizeof(QChar);
				f.allocate_new(size_buf);
				memcpy(f.begin(), value.constData(), size_buf);
				elt->fields().push_back(f);
			});
		});
	}
	catch (arcsight::Exception const& e)
	{
		PVLOG_ERROR("Error while reading data from Arcsight:\n----\n%s\n----\n", e.what());
		return NULL;
	}

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index-1>_last_elt_index) {
		_last_elt_index = _next_index-1;
	}

	return chunk;
}
