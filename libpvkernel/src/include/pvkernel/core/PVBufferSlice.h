/*
 * $Id: PVBufferSlice.h 3221 2011-06-30 11:45:19Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVBUFFERSLICE_FILE_H
#define PVBUFFERSLICE_FILE_H

#include <pvkernel/core/general.h>
#include <QString>
#include <unicode/unistr.h>
#include <QStringList>
#include <unicode/regex.h>
#include <QRegExp>
#include <boost/shared_array.hpp>

#include <pvkernel/core/stdint.h>

namespace PVCore {

typedef std::list< std::pair<char*,size_t> > buf_list_t;

/*! \brief Defines a "slice" of an existing buffer. Used by the normalisation process.
 *
 * This class originally defines a slice of an existing buffer. It is the base class of \ref PVCore::PVElement
 * and \ref PVCore::PVField. A slice can be "detached" from the original buffer if a reallocation is done. If such
 * reallocation occurs, the new pointer is add to a list of buffers, sothat further deallocation can be done.
 * For now, a \ref PVCore::PVElement object contains a list of buffers reallocated by its fields.
 *
 * \remark
 * A buffer slice has two sizes: a logical and a physical one. The logical size is changed according to \ref set_end, and
 * is the size of useful data inside this slice. The pysical size is the size to which a slice can grow without the need
 * of any reallocations. 
 */
class LibKernelDecl PVBufferSlice {
public:
	/*! \brief Construct a buffer slice from a start and end pointer.
	 *  \param[in] begin    Begin of the buffer slice
	 *  \param[in] end      End of the buffer slice, ie. the pointer after the last valid character
	 *  \param[in] buf_list Reference to an external buffer list used to keep track of reallocations
	 *
	 * Construct a buffer slice from a start and end pointer. Moreover, and buffer list must be provided in
	 * order to keep track of reallocations (if they occur).
	 *
	 * \note The size of this slice is thus end-begin .
	 */
	PVBufferSlice(char* begin, char* end, buf_list_t &buf_list);

	/*! \brief Construct an undefined slice.
	 *  \param[in] buf_list Reference to an external buffer list used to keep track of reallocations
	 */
	PVBufferSlice(buf_list_t &buf_list);


	/*! \brief Copy-constructor
	 *  \param[in] src Original buffer slice
	 *
	 *  \sa copy_from
	 */
	PVBufferSlice(PVBufferSlice const& src): _buf_list(src._buf_list) { copy_from(src); };

	// Destructor "inline" for performance reasons
	virtual ~PVBufferSlice() {}
public:
	/*! \brief Copy operator
	 *  \param[in] src Original buffer slice
	 *
	 *  \sa copy_from
	 */
	inline PVBufferSlice& operator=(PVBufferSlice const& src) { copy_from(src); return *this; };
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
	inline size_t physical_size() const { return (uintptr_t)_physical_end - (uintptr_t)_begin; }

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
	 *  This function basically set end() to end()+n, by checking if enough physical space is available. If this is not the case,
	 *  it will do nothing and returns false. Thus, a reallocation is needed, and this is the purpose of \ref grow_by_reallocate.
	 *
	 *  \sa grow_by_reallocate
	 */
	bool grow_by(size_t n);

	/*! \brief Grow the logical size of this slice, and reallocate it if necessary.
	 *  \param n Number of bytes to grow
	 *
	 *  This function basically set end() to end()+n. If there is not enough physical space for this operation, a reallocation
	 *  is performed.
	 */
	void grow_by_reallocate(size_t n);


	/*! \brief Discard the old content and reallocate a new buffer.
	 *  \param[in] n Size of the new buffer.
	 *
	 *  This function discard the old content and reallocate a new buffer. If a previous buffer had to
	 *  be reallocated (with \ref grow_by_reallocate for instance), then this buffer is deallocated.
	 */
	void allocate_new(size_t n);

public:
	/*! \brief Split this buffer slice into a list of buffer slices, according to a given byte.
	 *  \tparam     L          Standard C++ compliant container type where the slices will be inserted. L::value_type must have buffer slice as parent.
	 *  \param[out] container  Container where the slices will be inserted.
	 *  \param[in]  c          Separation byte
	 *  \param[in]  it_ins     Iterator where the new slices will be inserted in container.
	 *  \return The number of slices inserted.
	 */
	template<class L> typename L::size_type split(L &container, char c, typename L::iterator it_ins);

	/*! \brief Split this buffer slice into a list of buffer slices, according to a UTF16 character.
	 *  \tparam     L          Standard C++ compliant container type where the slices will be inserted. L::value_type must have buffer slice as parent.
	 *  \param[out] container  Container where the slices will be inserted.
	 *  \param[in]  c          Separation byte
	 *  \param[in]  it_ins     Iterator where the new slices will be inserted in container.
	 *  \return The number of slices inserted.
	 */
	template<class L> typename L::size_type split_qchar(L &container, QChar c, typename L::iterator it_ins);

	/*! \brief Split this buffer slice into a list of buffer slices, according to a Qt regular expression object.
	 *  \tparam     L          Standard C++ compliant container type where the slices will be inserted. L::value_type must have buffer slice as parent.
	 *  \param[out] container  Container where the slices will be inserted.
	 *  \param[in]  c          Separation byte
	 *  \param[in]  it_ins     Iterator where the new slices will be inserted in container.
	 *  \return The number of slices inserted.
	 *
	 *  \sa QRegexp
	 */
	template<class L> typename L::size_type split_regexp(L &container, QRegExp& re, typename L::iterator it_ins, bool bFullLine);

	/*! \brief Split this buffer slice into a list of buffer slices, according to an ICU regular expression object.
	 *  \tparam     L          Standard C++ compliant container type where the slices will be inserted. L::value_type must have buffer slice as parent.
	 *  \param[out] container  Container where the slices will be inserted.
	 *  \param[in]  c          Separation byte
	 *  \param[in]  it_ins     Iterator where the new slices will be inserted in container.
	 *  \return The number of slices inserted.
	 *
	 *  \sa RegexMatcher
	 */
	template<class L> typename L::size_type split_regexp(L &container, RegexMatcher& re, typename L::iterator it_ins, bool bFullLine);
public:
	/*! \brief Bind a QString to this slice.
	 *  \param[in] ret QString object to be bound
	 *  \return Returns ret for conveniance.
	 *
	 * This method will call QString::setRawData on ret with the underlying buffer slice.
	 * The advantage to use an already defined QString is that the itnernal structure might have already been allocated once and for all.
	 * Creating a QString object each time this function is called can be (used to be, and is still for the mapping process) a performance bottleneck.
	 */
	inline QString& get_qstr(QString& ret) const
	{
		/*
		if (_qstr.isNull()) {
			size_t nc = (_end-_begin)/sizeof(QChar);
			_qstr.setRawData((QChar*) _begin, nc);
		}
		*/
		const size_t nc = (_end-_begin)/sizeof(QChar);
		ret.setRawData((QChar*) _begin, nc);
		return ret;
	}

	/*! \brief Create an ICU UnicodeString object bound to this slice.
	 *  \return A freshly created UnicodeString object bound to this slice.
	 *
	 * This function creates a new UnicodeString object and bound it to this buffer slice.
	 * It has performance issues since the internal structures of UnicodeString are allocated each time.
	 * See also \ref get_qstr that used to have these issues.
	 */
	inline UnicodeString get_icustr() const
	{
		size_t nc = (_end-_begin)/sizeof(QChar);
		// We must use the "setTo" method instead of using this UnicodeString constructor, as this method
		// it sets the UnicodeString as a "read-only" alias (and not the constructor). Cf. ICU's sources...
		UnicodeString ret;
		ret.setTo(false, (const UChar *)(_begin), nc);
		return ret;
	}

	/*! \brief Change the buffer list that save the allocated buffers.
	 *  \param[in] buf_list Reference to the buffer list.
	 *  \sa PVBufferSlice
	 */
	inline void set_buflist(buf_list_t& buf_list) { _buf_list = buf_list; }

protected:
	// Perform a deep copy of current data in a new buffer
	void _realloc_data();

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
	// AG: this cache is removed, so that 8 bytes (== sizeof(QString) are won, and get_qstr is generally only called once !
	//mutable QString _qstr; // QString "cache"

	// AG: historically, we had a shared_ptr to a malloc memory zone if necessary
	// A shared pointer take to much time to initialise and copy, and used to be
	// a performance bottleneck during the creation of PVField and PVElement objects !
	//boost::shared_array<char> _realloc_buf;
	char* _realloc_buf;

	buf_list_t &_buf_list;
};

template <class L>
typename L::size_type PVCore::PVBufferSlice::split(L& container, char c, typename L::iterator it_ins)
{
	char* start = begin();
	char* end;
	size_t len = size();
	size_t n = 0;
	while ((end = (char*) memchr(start, c, len)) != NULL) {
		// Create a new element according to the current type
		// and copy it
		typename L::value_type elt(*((typename L::value_type *)this));

		// Then, set the new slice
		elt._begin = start;
		elt._end = end;
		elt._physical_end = elt._end;

		// Add the new object to the container
		container.insert(it_ins, elt);

		// And go on
		n++;
		len -= (uintptr_t)end - (uintptr_t)start;
		if (len == 0)
			return n;
		len--;
		start = end+1;
	}
	if (len > 0) {
		typename L::value_type elt(*((typename L::value_type *)this));
		elt._begin = start;
		elt._end = this->end();
		container.insert(it_ins, elt);
		n++;
	}

	return n;
}

template <class L>
typename L::size_type PVCore::PVBufferSlice::split_qchar(L& container, QChar c, typename L::iterator it_ins)
{
	QString qs;
	get_qstr(qs);
	QChar* str_start = (QChar*) begin();
	int old_pos_c = 0;
	int pos_c;
	int n = 0;
	ssize_t sstr = qs.size();
	while ((pos_c = qs.indexOf(c, old_pos_c)) != -1) {
		typename L::value_type elt(*((typename L::value_type *)this));
		elt._begin = (char*) (str_start + old_pos_c);
		elt._end = (char*) (str_start + pos_c);
		elt._physical_end = elt._end;

		container.insert(it_ins, elt);

		n++;
		old_pos_c = pos_c+1;
	}

	if (old_pos_c < sstr) {
		typename L::value_type elt(*((typename L::value_type *)this));
		elt._begin = (char*) (str_start + old_pos_c);
		elt._end = this->end();
		container.insert(it_ins, elt);
		n++;
	}

	return n;
}

template <class L>
typename L::size_type PVCore::PVBufferSlice::split_regexp(L& container, RegexMatcher& re_, typename L::iterator it_ins, bool bFullLine)
{
	UErrorCode err = U_ZERO_ERROR;
	re_.reset(get_icustr());
	if (bFullLine) {
		if (!re_.matches(err)) {
			return 0;
		}
	}
	else {
		if (!re_.find()) {
			return 0;
		}
	}

	UChar* bstart = (UChar*) begin();
	size_t n = 0;
	for (int32_t i = 1; i <= re_.groupCount(); i++) {
		typename L::value_type elt(*((typename L::value_type *)this));
		int32_t start = re_.start(i, err);
		int32_t end = re_.end(i, err);
		if (U_FAILURE(err)) {
			return n;
		}

		elt._begin = (char*) (bstart+start);
		elt._end = (char*) (bstart+end);
		elt._physical_end = elt._end;
		container.insert(it_ins,elt);
		n++;
	}

	return n;
}

template <class L>
typename L::size_type PVCore::PVBufferSlice::split_regexp(L& container, QRegExp& re_, typename L::iterator it_ins, bool bFullLine)
{
	QString qs;
	get_qstr(qs);
	QStringList rx_fields;
	if (bFullLine) {
		if (re_.exactMatch(qs) == -1) {
			return 0;
		}
	}
	else {
		if (re_.indexIn(qs) == -1) {
			return 0;
		}
	}

	rx_fields = re_.capturedTexts();
	QStringList::iterator ite = rx_fields.end();
	QStringList::iterator it = rx_fields.begin();
	it++;
	size_t n = 0;
	char* bstart = begin();
	size_t len = size();
	while (it != ite) {
		typename L::value_type elt(*((typename L::value_type *)this));
		if (len < (*it).size()) {
			PVLOG_ERROR("(split_regexp): buffer slice size too small to hold regexp results !\n");
			return n;
		}
		size_t fsize = (*it).size() * sizeof(QChar);
		memcpy(bstart, (*it).constData(), fsize);
		len -= fsize;
		elt._begin = bstart;
		bstart += fsize + 1;
		elt._end = bstart-1;
		elt._physical_end = elt._end;
		container.insert(it_ins, elt);
		it++;
		n++;
	}

	return n;
}

}

#endif

