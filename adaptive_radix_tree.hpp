#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include "adaptive_radix_tree_node.hpp"

/// Defines how to iterate over tuples.
/// Briefly; it takes index vector and a starting point as input
/// and follows starting point as index for each iteration.
///
/// [index_vector]         [start]     [output values]
/// [ 2 | 4 | x | 1 | x ]  2        => 2, 4
///                        0        => 0, 2
class CIndexIterator final: public std::iterator<std::input_iterator_tag, uint32_t>
{
public:
  explicit CIndexIterator( const std::vector<uint32_t>& indexes, uint32_t start )
      : indexes_( &indexes ),
        index_(start )
  {
  }

  CIndexIterator& operator++()
  {
    this->index_ = (*indexes_)[this->index_];
    return *this;
  }

  CIndexIterator operator++( int )
  {
    CIndexIterator copy( *this );
    ++*this;
    return copy;
  }

  bool operator==( CIndexIterator const & o ) const
  {
    assert( this->indexes_ == o.indexes_ );
    return this->index_ == o.index_;
  }

  bool operator!=( const CIndexIterator& o ) const
  {
    return !( ( *this ) == o );
  }

  reference operator*()
  {
    return index_;
  }

protected:
  std::vector<value_type> const* indexes_;
  value_type index_;
};

/// Defines actions for ART leaf and intermediate nodes.
/// Use CIndexActionBase if you don't need node string contents.
class CActionBase
{
public:
  virtual ~CActionBase() = default;

  /// This function is called for each node and provides:
  /// - pointer of node
  /// - concatenated string prefix up until this node
  /// - number of levels from root node.
  virtual void HandleNode(const CArtNode * node, const std::string& prefix, uint32_t level ) = 0;

  /// This function is called only for leaf nodes and provides:
  /// - concatenated string key up until the leaf
  /// - string and ending of index iterator that provides row indexes belongs this key.
  virtual void HandleTuple( const std::string& key, CIndexIterator begin, CIndexIterator end ) = 0;
};

/// Defines actions for tuples of ART nodes.
class CIndexActionBase
{
public:
  virtual ~CIndexActionBase() = default;
  /// This function provides tuple iterator for each leaf node.
  virtual void HandleTuple( CIndexIterator begin, CIndexIterator end ) = 0;
};

//todo(demiroz): document!
//todo(demiroz): add support for int64_t!

// This implementation is based on paper named "The Adaptive Radix Tree: ARTful Indexing for Main-Memory Databases"
// http://www3.informatik.tu-muenchen.de/~leis/papers/ART.pdf
class CAdaptiveRadixTree
{
public:
  explicit CAdaptiveRadixTree(uint32_t max_index_count )
      : root_( new CArtNode256() ),
        indexes_( std::make_shared < std::vector < uint32_t >> (max_index_count) )
  {
  }

  explicit CAdaptiveRadixTree(std::shared_ptr<std::vector<uint32_t>> indexes )
      : root_( new CArtNode256() ),
        indexes_( indexes )
  {
  }

  CAdaptiveRadixTree( const CAdaptiveRadixTree & other ) = delete;
  CAdaptiveRadixTree& operator=( const CAdaptiveRadixTree& ) = delete;

  ~CAdaptiveRadixTree();

  void Swap( CAdaptiveRadixTree & other )
  {
    swap( *this, other );
  }

  friend void swap(CAdaptiveRadixTree & first, CAdaptiveRadixTree & second ) throw ()
  {
    using std::swap;
    swap( first.root_, second.root_ );
    swap( first.null_string_, second.null_string_ );
    swap( first.null_string_count_, second.null_string_count_ );
    swap( first.max_string_length_, second.max_string_length_ );
    swap( first.unique_string_count_, second.unique_string_count_ );
    swap( first.total_string_length_, second.total_string_length_ );
    swap( first.suffix_table_, second.suffix_table_ );
    swap( first.indexes_, second.indexes_ );
  }

  void AddEntry( const char* key, size_t key_length, uint32_t value );

  void Traverse( CActionBase & action ) const
  {
    if ( root_ )
    {
      std::string key;
      TraverseRecursive( root_, action, key, 0 );
    }
  }

  void TraverseIndexes( CIndexActionBase & action ) const
  {
    if ( root_ )
    {
      TraverseIndexRecursive( root_, action );
    }
  }

  void Reset();

  std::unique_ptr<CAdaptiveRadixTree> Split();

  void Join( CAdaptiveRadixTree & other )
  {
    // Merge null string positions first.
    CIndexIterator it = other.GetNullStringBegin(), end = other.GetNullStringEnd();
    while ( it != end )
    {
      auto value = *it;  // cache the value
      ++it;
      AddNullString( value );
    }

    Merge( &root_, &(other.root_), other.suffix_table_ );

    total_string_length_ += other.GetTotalStringLength();
    max_string_length_ = std::max( max_string_length_, other.GetMaxStringLength() );
  }

  /// Handle NULL string separately.
  void AddNullString( uint32_t value )
  {
    assert( value < indexes_->size() );  // we may resize the index vector anyway, but we have to synchronize it because
        // we are using shared index vector for joinable ARTs.
    indexes_->operator[]( value ) = null_string_;
    null_string_ = value;
    ++null_string_count_;
  }

  void Reserve( int64_t new_capacity )
  {
    indexes_->reserve( new_capacity );
  }

  void Resize( int64_t new_size )
  {
    indexes_->resize( new_size );
  }

  CIndexIterator GetNullStringBegin()
  {
    return CIndexIterator( *indexes_, null_string_ );
  }

  CIndexIterator GetNullStringEnd()
  {
    return CIndexIterator( *indexes_, CArtNode::LAST_INDEX_IDENTIFIER );
  }

  uint32_t GetNullStringCount() const
  {
    return null_string_count_;
  }

  size_t GetMaxStringLength() const
  {
    return max_string_length_;
  }

  size_t GetUniqueStringCount() const
  {
    if ( null_string_ == CArtNode::LAST_INDEX_IDENTIFIER )
    {
      return unique_string_count_;
    }
    else
    {
      return unique_string_count_ + 1;
    }
  }

  size_t GetIndexVectorLength() const
  {
    return indexes_->size();
  }

  size_t GetTotalStringLength() const
  {
    return total_string_length_;
  }

private:
  void TraverseRecursive(CArtNode * iNode, CActionBase & action, std::string& key, int level ) const;

  void TraverseIndexRecursive(CArtNode * iNode, CIndexActionBase & action ) const;

  CArtNode ** FindChild(CArtNode * node, uint8_t c ) const;

  CArtNode ** InsertInNode(CArtNode ** base_node, uint8_t c, CArtNode * child_node );

  void InsertValue(CArtNode ** node_base, CArtNode * node, uint32_t value );

  void MovePrefix(CArtNode * input_node, std::string& other_suffix_table );

  void Merge(CArtNode ** left, CArtNode ** right, std::string& right_suffix_table_ );

  void MergeChildNodes(CArtNode ** left, CArtNode * right, std::string& right_suffix_table_ );

private:
  CArtNode * root_; // todo(demiroz): unique_ptr?
  uint32_t null_string_ = CArtNode::LAST_INDEX_IDENTIFIER;
  uint32_t null_string_count_ = 0;
  size_t max_string_length_ = 0;
  size_t unique_string_count_ = 0;
  size_t total_string_length_ = 0;
  std::string suffix_table_;
  std::shared_ptr<std::vector<uint32_t>> indexes_;
};
