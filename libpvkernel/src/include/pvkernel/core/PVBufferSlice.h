/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVBUFFERSLICE_FILE_H
#define PVBUFFERSLICE_FILE_H

#include <cstddef>
#include <cstdint>
#include <list>
#include <utility>

namespace PVCore
{

typedef std::list<std::pair<char*, size_t>> buf_list_t;

/*! \brief Defines a "slice" of an existing buffer. Used by the normalisation process.
 *
 * This class originally defines a slice of an existing buffer. It is the base class of \ref
 *PVCore::PVElement
 * and \ref PVCore::PVField. A slice can be "detached" from the original buffer if a reallocation is
 *done. If such
 * reallocation occurs, the new pointer is add to a list of buffers, sothat further deallocation can
 *be done.
 * For now, a \ref PVCore::PVElement object contains a list of buffers reallocated by its fields.
 *
 * \remark
 * A buffer slice has two sizes: a logical and a physical one. The logical size is changed according
 *to \ref set_end, and
 * is the size of useful data inside this slice. The pysical size is the size to which a slice can
 *grow without the need
 * of any reallocations.
 */
class PVBufferSlice
{
  public:
	/*! \brief Construct a buffer slice from a start and end pointer.
	 *  \param[in] begin    Begin of the buffer slice
	 *  \param[in] end      End of the buffer slice, ie. the pointer after the last valid character
	 *  \param[in] buf_list Reference to an external buffer list used to keep track of reallocations
	 *
	 * Construct a buffer slice from a start and end pointer. Moreover, and buffer list must be
	 *provided in
	 * order to keep track of reallocations (if they occur).
	 *
	 * \note The size of this slice is thus end-begin .
	 */
	PVBufferSlice(char* begin, char* end, buf_list_t& buf_list);

	/*! \brief Construct an undefined slice.
	 *  \param[in] buf_list Reference to an external buffer list used to keep track of reallocations
	 */
	explicit PVBufferSlice(buf_list_t& buf_list);

	/*! \brief Copy-constructor
	 *  \param[in] src Original buffer slice
	 *
	 *  \sa copy_from
	 */
	PVBufferSlice(PVBufferSlice const& src) : _buf_list(src._buf_list) { copy_from(src); };

	// Destructor "inline" for performance reasons
	virtual ~PVBufferSlice() {}

  public:
	/*! \brief Copy operator
	 *  \param[in] src Original buffer slice
	 *
	 *  \sa copy_from
	 */
	inline PVBufferSlice& operator=(PVBufferSlice const& src)
	{
		copy_from(src);
		return *this;
	};

  public:
	/*! \brief Pointer to the beginning of this slice.
	 */
	char* begin() const;

	/*! \brief Pointer to the end of this slice (after the last valid character)
	 */
	char* end() const;

	/*! \brief Return the logical size of this slice.
	 *  \note It basically compute end() - start().
	 */
	size_t size() const;

	/*! \brief Return the physical size of this slice.
	 */
	inline size_t physical_size() const { return (uintptr_t)_physical_end - (uintptr_t)_begin; }

	/*! \brief Change the beginning of this slice.
	 *  \param[in] p Pointer to the beginning of the slice.
	 *
	 *  You must ensure that p <= end(). In debug mode, an assertion is made.
	 */
	void set_begin(char* p);

	/*! \brief Change the end of this slice
	 *  \param[in] p Pointer to the end of the slice.
	 *
	 *  You must ensure that p >= begin(). In debug mode, an assertion is made.
	 */
	void set_end(char* p);

	/*! \brief Change the physical end of this slice.
	 *  \param[in] p Pointer to the physical end of this slice.
	 *
	 *  You must ensure that p >= end(). In debug mode, an assertion is made.
	 */
	void set_physical_end(char* p); // In case more space can be taken by this slice if wanted

	/*! \brief Grow the logical size of this slice if enough physical space is available.
	 *  \param n Number of bytes to grow.
	 *  \return true if enough physical space was available, false otherwise.
	 *
	 *  This function basically set end() to end()+n, by checking if enough physical space is
	 *available. If this is not the case,
	 *  it will do nothing and returns false. Thus, a reallocation is needed, and this is the
	 *purpose of \ref grow_by_reallocate.
	 *
	 *  \sa grow_by_reallocate
	 */
	bool grow_by(size_t n);

	/*! \brief Grow the logical size of this slice, and reallocate it if necessary.
	 *  \param n Number of bytes to grow
	 *
	 *  This function basically set end() to end()+n. If there is not enough physical space for this
	 *operation, a reallocation
	 *  is performed.
	 */
	void grow_by_reallocate(size_t n);

	/*! \brief Discard the old content and reallocate a new buffer.
	 *  \param[in] n Size of the new buffer.
	 *
	 *  This function discard the old content and reallocate a new buffer. If a previous buffer had
	 *to
	 *  be reallocated (with \ref grow_by_reallocate for instance), then this buffer is deallocated.
	 */
	void allocate_new(size_t n);

  private:
	inline void copy_from(PVBufferSlice const& src)
	{
		_begin = src._begin;
		_end = src._end;
		_physical_end = src._physical_end;
		_realloc_buf = src._realloc_buf;
	}

  protected:
	char* _begin;
	char* _end;
	char* _physical_end;
	char* _realloc_buf;

	buf_list_t& _buf_list;
};
} // namespace PVCore

#endif
