#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/rush/PVRawSourceBase.h>


PVRush::PVAggregator::PVAggregator(list_inputs const& inputs)
{
	_inputs = inputs;
	_src_offsets[0] = _inputs.front();
	init();
}

PVRush::PVAggregator::PVAggregator()
{
	init();
}

void PVRush::PVAggregator::init()
{
	_eoi = false;
	_nstart = 0;
	_nlast = 0;
	_nend = pvconfig.value("pvkernel/rush/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT).toInt();
	__stop_cond_false = false;
	_stop_cond = &__stop_cond_false;
	_last_elt_agg_index = 0;
	_cur_input = _inputs.begin();
	_cur_src_index = 0;
}


PVRush::PVAggregator::PVAggregator(const PVAggregator& org)
{
	// Special copy constructor !
	
	// Find original cur_input position
	list_inputs::const_iterator it,ite;
	it = org._inputs.begin();
	ite = org._inputs.end();
	int n = 0;
	while (it != ite && it != org._cur_input) {
		n++;
		it++;
	}

	__stop_cond_false = false;
	_inputs = org._inputs;
	_cur_input = _inputs.begin() + n;
	// If original condition has not been changed, then we take as a default our "false" boolean
	if (org._stop_cond != &(org.__stop_cond_false))
		_stop_cond = org._stop_cond;
	else
		_stop_cond = &(this->__stop_cond_false);
	_eoi = org._eoi;
	_nstart = org._nstart;
	_nend = org._nend;
	_nlast = org._nlast;
	_last_elt_agg_index = org._last_elt_agg_index;
	_cur_src_index = org._cur_src_index;
	debug();
}

void PVRush::PVAggregator::set_stop_condition(bool *cond)
{
	_stop_cond = cond;
}

void PVRush::PVAggregator::process_from_source(list_inputs::iterator input_start, PVCore::chunk_index nstart, PVCore::chunk_index nend)
{
	// FIXME: as process_indexes isn't efficient, this method could read *twice* the files...
	
	// Process from nstart to nend, starting by input_start
	
	// Find, compute offset for input_start
	if (!read_until_source(input_start)) {
		PVLOG_ERROR("(PVAggregator::process_from_source) unable to reach source %s. Using the last one...\n", qPrintable((*input_start)->human_name()));
		if (input_start != _inputs.end()-1) {
			process_from_source(_inputs.end()-1, nstart, nend);
		}
		else {
			PVLOG_ERROR("(PVAggregator::process_from_source) already searching for the last source ! Starting from the beggining...\n");
			process_indexes(_nstart, _nend);
			return;
		}
	}
	PVCore::chunk_index offset = _cur_src_index;

	// Then compute the new nstart and nend value
	_nstart = nstart + offset;
	_nend = nend + offset;

	// And use process_indexes
	process_indexes(_nstart, _nend);
}

void PVRush::PVAggregator::process_indexes(PVCore::chunk_index nstart, PVCore::chunk_index nend)
{
	_nstart = nstart;
	_nend = nend;
	_eoi = false;
	_nlast = 0;
	_last_elt_agg_index = 0;
	_cur_src_index = 0;
	_cur_input = _inputs.begin();

	list_inputs::iterator it;
	for (it = _inputs.begin(); it != _inputs.end(); it++) {
		// Reset all inputs position pointer
		PVLOG_DEBUG("PVExtractor::process_indexes seek begin on source %s\n", qPrintable((*it)->human_name()));
		(*it)->seek_begin();
	}
}

bool PVRush::PVAggregator::read_until_source(list_inputs::iterator input_start)
{
	PVCore::PVChunk* c;
	while (_cur_input != input_start) {
		if ((c = next_chunk()) == NULL)
			return false;
		c->free();
	}
	return true;
}

PVCore::PVChunk* PVRush::PVAggregator::read_until_index(PVCore::chunk_index idx) const
{
	PVCore::PVChunk* ret = NULL;
	while (_nlast < idx) {
		if (ret != NULL)
			ret->free();
		if ((ret = next_chunk()) == NULL)
			return NULL;
	}
	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::next_chunk() const
{
	if (*_stop_cond)
	{
		PVLOG_DEBUG("(PVAggregator::next_chunk) aggregator stop because of stop condition\n");
		return NULL;
	}

	// Get chunk from current input
	PVCore::PVChunk* ret = (*_cur_input)->operator()();

	while (ret == NULL) {
		// No more chunks available from current input. Go on !
		_cur_input++;
		if (_cur_input == _inputs.end()) {
			// No more data available. Return NULL
			return NULL;
		}
		_cur_src_index = _nlast;
		_src_offsets[_cur_src_index] = *_cur_input;
		ret = (*_cur_input)->operator()();
	}

	if (ret != NULL) {
		ret->_agg_index = _cur_src_index + ret->_index;
		_nlast += ret->c_elements().size();
		_last_elt_agg_index = _nlast + ret->_index-1;
	}

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()() const
{
	if (_nlast >= _nend) {
		// This is the end of this job
		return NULL;
	}

	PVCore::PVChunk* ret;
	if (_nlast < _nstart) {
		// We have to read until _nstart indexes
		ret = read_until_index(_nstart);
	}
	else {
		ret = next_chunk();
	}

	if (ret == NULL) {
		_eoi = true;
		PVLOG_DEBUG("Aggregator: end of inputs\n");
	}

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()(tbb::flow_control &fc) const
{
	PVCore::PVChunk* ret = this->operator()();
	if (ret == NULL) {
		PVLOG_DEBUG("(PVAggregator::next_chunk) aggregator stop because of no more input datas\n");
		fc.stop();
	}
	return ret;
}

void PVRush::PVAggregator::read_all_chunks_from_beggining()
{
	PVLOG_DEBUG("(PVAggregator) read all chunks\n");
	// Just reset everything
	process_indexes(0, 1000);

	// Save the old stop condition pointer (just in case)
	bool* old_stop_cond = _stop_cond;
	// Never ever ever stop !!
	_stop_cond = &__stop_cond_false;
	// And here we go !!
	PVCore::PVChunk* chunk;
	while ((chunk = next_chunk()) != NULL) {
		chunk->free();
	}
	// Restore it
	_stop_cond = old_stop_cond;
}

PVRush::PVAggregator::list_inputs const& PVRush::PVAggregator::get_inputs() const
{
	return _inputs;
}

// Helper function
PVRush::PVAggregator PVRush::PVAggregator::from_unique_source(PVRush::PVRawSourceBase_p source)
{
	list_inputs inputs;
	inputs.push_back(source);

	return PVAggregator(inputs);
}

void PVRush::PVAggregator::add_input(PVRush::PVRawSourceBase_p in)
{
	if (!_src_offsets[0])
		_src_offsets[0] = in;
	_inputs.push_back(in);
}

PVCore::chunk_index PVRush::PVAggregator::last_elt_agg_index()
{
	return _last_elt_agg_index;
}

PVRush::PVRawSourceBase_p PVRush::PVAggregator::agg_index_to_source(PVCore::chunk_index idx, size_t* offset)
{
	map_source_offsets::reverse_iterator it;
	for (it = _src_offsets.rbegin(); it != _src_offsets.rend(); it++) {
		if (idx >= (*it).first) {
			if (offset)
				*offset = (*it).first;
			return (*it).second;
		}
	}
	return PVRush::PVRawSourceBase_p();
}

void PVRush::PVAggregator::debug()
{
	// List sources
	PVLOG_DEBUG("PVAggregator::debug\n");
	list_inputs::iterator iti;
	PVLOG_DEBUG("PVAggregator::debug source\n");
	for (iti = _inputs.begin(); iti != _inputs.end(); iti++) {
		PVLOG_DEBUG("source %s\n", qPrintable((*iti)->human_name()));
	}

	PVLOG_DEBUG("PVAggregator::debug offset->source\n");
	map_source_offsets::iterator it;
	for (it = _src_offsets.begin(); it != _src_offsets.end(); it++)
		PVLOG_DEBUG("offset %d: source %s\n", (*it).first, qPrintable((*it).second->human_name()));
	PVLOG_DEBUG("PVAggregator::debug nstart=%d nlast=%d nend=%d\n", _nstart, _nlast, _nend);
}
