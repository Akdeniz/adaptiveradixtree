#pragma once

/* if C++11, implement our own make_unique */
#if ( __cplusplus == 201103L )
#include <memory>
#include <utility>
#include <functional>

namespace
{
template <class T>
inline void hash_combine( std::size_t& seed, const T& v )
{
  std::hash<T> hasher;
  seed ^= hasher( v ) + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 );
}
}

namespace
{
// To be used for unique pointers of single objects
template<typename T>
struct MakeUniqueTraits
{
  typedef T PointerType;
};

// To be used for unique pointers of arrays with unknown bounds
template<typename T>
struct MakeUniqueTraits<T[]>
{
  typedef T ArrayType;
};

// To be used to prohibit unique pointers of arrays with known bounds
template <typename T, size_t B>
struct MakeUniqueTraits<T[B]>
{
  typedef void ProhibitKnownBounds;
};
}

namespace std {
/// make a unique pointer of a single object
template<class T, class... Args>
unique_ptr< typename ::MakeUniqueTraits<T>::PointerType > make_unique(Args&&... args) {
  return unique_ptr< typename ::MakeUniqueTraits<T>::PointerType >( new T(std::forward<Args>(args)...) );
}

/// make a unique pointer of an array with unknown bounds
template<class T>
unique_ptr< typename ::MakeUniqueTraits<T>::ArrayType[] > make_unique(size_t n) {
  typedef typename ::MakeUniqueTraits<T>::ArrayType AT;
  return unique_ptr< AT[] >(new AT[n]());
}

/// no unique pointers of arrays with known bounds!
template <typename T, typename... _Args>
typename ::MakeUniqueTraits<T>::ProhibitKnownBounds make_unique(_Args&&...) = delete;
} // ns std
#endif /* if C++11 */