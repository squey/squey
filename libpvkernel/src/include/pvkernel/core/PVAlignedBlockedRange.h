#ifndef __PV_ALIGNED_BLOCKED_RANGE_H_
#define __PV_ALIGNED_BLOCKED_RANGE_H_

#include "tbb/tbb_stddef.h"

namespace PVCore {

/** \page range_req Requirements on range concept
    Class \c R implementing the concept of range must define:
    - \code R::R( const R& ); \endcode               Copy constructor
    - \code R::~R(); \endcode                        Destructor
    - \code bool R::is_divisible() const; \endcode   True if range can be partitioned into two subranges
    - \code bool R::empty() const; \endcode          True if range is empty
    - \code R::R( R& r, split ); \endcode            Split range \c r into two subranges.
**/

//! A range over which to iterate.
/** @ingroup algorithms */
template<typename Value, size_t Align = 1>
class PVAlignedBlockedRange {
public:
    //! Type of a value
    /** Called a const_iterator for sake of algorithms that need to treat a blocked_range
        as an STL container. */
    typedef Value const_iterator;

    //! Type for size of a range
    typedef std::size_t size_type;

    //! Construct range with default-constructed values for begin and end.
    /** Requires that Value have a default constructor. */
    PVAlignedBlockedRange() : my_end(), my_begin() {}

    //! Construct range over half-open interval [begin,end), with the given grainsize.
    PVAlignedBlockedRange( Value begin_, Value end_, size_type grainsize_=1 ) :
        my_end(end_), my_begin(begin_), my_grainsize(grainsize_)
    {
        __TBB_ASSERT( my_grainsize>0, "grainsize must be positive" );
    }

    //! Beginning of range.
    const_iterator begin() const {return my_begin;}

    //! One past last value in range.
    const_iterator end() const {return my_end;}

    //! Size of the range
    /** Unspecified if end()<begin(). */
    size_type size() const {
        __TBB_ASSERT( !(end()<begin()), "size() unspecified if end()<begin()" );
        return size_type(my_end-my_begin);
    }

    //! The grain size for this range.
    size_type grainsize() const {return my_grainsize;}

    //------------------------------------------------------------------------
    // Methods that implement Range concept
    //------------------------------------------------------------------------

    //! True if range is empty.
    bool empty() const {return !(my_begin<my_end);}

    //! True if range is divisible.
    /** Unspecified if end()<begin(). */
    bool is_divisible() const {return size()>Align && my_grainsize<size();}

    //! Split range.
    /** The new Range *this has the second half, the old range r has the first half.
        Unspecified if end()<begin() or !is_divisible(). */
    PVAlignedBlockedRange( PVAlignedBlockedRange& r, tbb::split ) :
        my_end(r.my_end),
        my_begin(do_split(r)),
        my_grainsize(r.my_grainsize)
    {}

private:
    /** NOTE: my_end MUST be declared before my_begin, otherwise the forking constructor will break. */
    Value my_end;
    Value my_begin;
    size_type my_grainsize;

    //! Auxiliary function used by forking constructor.
    /** Using this function lets us not require that Value support assignment or default construction. */
    static Value do_split( PVAlignedBlockedRange& r ) {
        __TBB_ASSERT( r.is_divisible(), "cannot split blocked_range that is not divisible" );
        Value middle = r.my_begin + (r.my_end-r.my_begin)/2u;
        middle = ((middle+Align-1)/Align)*Align;
        r.my_end = middle;
        return middle;
    }
};

} // namespace tbb

#endif /* __PV_ALIGNED_BLOCKED_RANGE_H_ */
