/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVERFSOURCE_FILE_H__
#define __PVERFSOURCE_FILE_H__

#include <iterator>
#include <fcntl.h>
#include <memory>

#include <QString>

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include "../../common/erf/PVERFDescription.h"
#include "PVERFSource.h"

#include <pvkernel/core/PVBinaryChunk.h>

#include <pvkernel/core/PVBinaryChunk.h>
#include "inendi_erfio/inendi_erfio.h"

namespace PVRush
{

class PVERFSource : public PVRawSourceBaseType<PVCore::PVBinaryChunk>
{
  public:
	PVERFSource(PVInput_p input, PVERFDescription* input_desc, size_t chunk_size)
	    : _input_desc(input_desc)
	{
	}
	virtual ~PVERFSource() {}

	QString human_name() override { return "ERF"; }
	void seek_begin() override {}
	void prepare_for_nelts(chunk_index nelts) override{}; // FIXME
	size_t get_size() const override { return 0; };       // FIXME
	PVCore::PVBinaryChunk* operator()() override
	{
		size_t column_count = 743;
		PVCore::PVBinaryChunk* bc = new PVCore::PVBinaryChunk(column_count, 1000, 0);

		PVCore::PVBinaryChunk* chunk = nullptr;
		if (_first_time) {
			auto start = std::chrono::system_clock::now();

			inendi_erf::inendi_erfio perfio(_input_desc->path().toStdString().c_str());

			std::vector<std::string> stage_names;
			perfio.get_stage_names(stage_names);

			_cols.resize(column_count);

			for (PVCol col(0); col < column_count; col++) {
				pvlogger::info() << "col=" << col << std::endl;
				std::vector<ERF_INT> iDs;
				EString node("NODE");
				EString varGroup("THERMAL");
				// EString varGroup("Temperature");
				perfio.contour_results(col + 1, node, varGroup, iDs, _cols[col]);
				bc->set_column_chunk(col, _cols[col]);
			}

			bc->set_rows_count(_cols[0].size());

			chunk = bc;

			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> diff = end - start;
			pvlogger::error() << "set_column_chunk : " << diff.count() << std::endl;
		};

		_first_time = false;
		return chunk;

	} // FIXME

  private:
	PVERFDescription* _input_desc;
	bool _first_time = true;
	std::vector<std::vector<ERF_FLOAT>> _cols; // FIXME : should take ownership
};

} // namespace PVRush

#endif // __PVERFSOURCE_FILE_H__
