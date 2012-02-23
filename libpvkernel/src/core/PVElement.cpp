#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVNraw.h>

tbb::tbb_allocator<PVCore::PVElement> PVCore::PVElement::_alloc;
//std::allocator<PVCore::PVElement> PVCore::PVElement::_alloc;

PVCore::PVElement::PVElement(PVChunk* parent) :
	PVBufferSlice(_reallocated_buffers)
{
	init(parent);
}

PVCore::PVElement::PVElement(PVChunk* parent, char* begin, char* end) :
	PVBufferSlice(begin, end, _reallocated_buffers)
{
	init(parent);
}

PVCore::PVElement::PVElement(PVElement const& src) :
	PVBufferSlice(src)
{
	// No copy must occur !
	assert(false);
}

PVCore::PVElement::~PVElement()
{
	clear_saved_buf();

	static tbb::tbb_allocator<char> alloc;
	buf_list_t::const_iterator it;
	for (it = _reallocated_buffers.begin(); it != _reallocated_buffers.end(); it++) {
		alloc.deallocate(it->first, it->second);
	}
}

void PVCore::PVElement::init(PVChunk* parent)
{
	_valid = true;
	_parent = parent;
	// In the beggining, it only has a big field
	//PVField f(*this, begin(), end());
	//_fields.push_back(f);
	_org_buf = NULL;
	_org_buf_size = 0;
}

void PVCore::PVElement::init_fields(void* fields_buf, size_t size_buf)
{
	new (&_fields) list_fields(list_fields::allocator_type(fields_buf, size_buf));
	_fields.push_back(PVField(*this, begin(), end()));
}

bool PVCore::PVElement::valid() const
{
	return _valid;
}

void PVCore::PVElement::set_invalid()
{
	_valid = false;
}

PVCore::list_fields& PVCore::PVElement::fields()
{
	return _fields;
}

PVCore::list_fields const& PVCore::PVElement::c_fields() const
{
	return _fields;
}

void PVCore::PVElement::set_parent(PVChunk* parent)
{
	_parent = parent;
}

PVCore::PVChunk* PVCore::PVElement::chunk_parent()
{
	return _parent;
}

PVCore::buf_list_t& PVCore::PVElement::realloc_bufs()
{
	return _reallocated_buffers;
}

void PVCore::PVElement::save_elt_buffer()
{
	clear_saved_buf();
	static tbb::tbb_allocator<char> alloc;
	_org_buf = alloc.allocate(size());
	_org_buf_size = size();
	memcpy(_org_buf, begin(), size());
}

bool PVCore::PVElement::restore_elt_with_saved_buffer()
{
	if (!_org_buf) {
		return false;
	}
	assert(_org_buf_size <= physical_size());
	memcpy(begin(), _org_buf, _org_buf_size);
	clear_saved_buf();
	return true;
}

void PVCore::PVElement::clear_saved_buf()
{
	if (!_org_buf) {
		return;
	}

	static tbb::tbb_allocator<char> alloc;
	alloc.deallocate(_org_buf, _org_buf_size);
	_org_buf = NULL;
	_org_buf_size = 0;
}

char* PVCore::PVElement::get_saved_elt_buffer(size_t& n)
{
	n = _org_buf_size;
	return _org_buf;
}

chunk_index PVCore::PVElement::get_elt_index()
{
	PVChunk* parent = chunk_parent();
	return parent->get_index_of_element(*this);
}

chunk_index PVCore::PVElement::get_elt_agg_index()
{
	PVChunk* parent = chunk_parent();
	return parent->get_agg_index_of_element(*this);
}

void PVCore::PVElement::give_ownerhsip_realloc_buffers(PVRush::PVNraw& nraw)
{
	nraw.take_realloc_buffers(_reallocated_buffers);
}
