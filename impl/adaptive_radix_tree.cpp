#include "adaptive_radix_tree.hpp"

// Check for 64/32 bit system
#if _WIN32 || _WIN64
#if _WIN64
#define ENVIRONMENT_64 1
#endif
#endif

#if __GNUC__
#if __x86_64__ || __ppc64__ || __PPC64__ || __aarch64__
#define ENVIRONMENT_64 1
#endif
#endif

#include <algorithm>
#include <cassert>

#if ENVIRONMENT_64
#include <immintrin.h>
#endif

#include <iostream>
#include <memory>
#include <string>

#include "adaptive_radix_tree_node.hpp"
#include "utils.hpp"

CArtNode ** CAdaptiveRadixTree::FindChild(CArtNode * node, uint8_t c ) const
{
  switch ( node->node_type_ )
  {
    case CArtNode::Type::Fanout4:
    {
      CArtNode4 * node_4 = static_cast<CArtNode4 *>( node );
      for ( unsigned i = 0; i < node_4->children_count_; ++i )
      {
        if ( node_4->key_[i] == c )
        {
          return &node_4->child_[i];
        }
      }
      return nullptr;
    }
    break;

    case CArtNode::Type::Fanout16:
    {
      CArtNode16 * node_16 = static_cast<CArtNode16 *>( node );
#if ENVIRONMENT_64
      __m128i cmp = _mm_cmpeq_epi8( _mm_set1_epi8( detail::Helper::FlipSign( c ) ),
                                    _mm_loadu_si128( reinterpret_cast<__m128i*>( node_16->key_ ) ) );
      unsigned bitfield = _mm_movemask_epi8( cmp ) & ( ( 1 << node_16->children_count_ ) - 1 );
      if ( bitfield )
      {
        return &node_16->child_[detail::Helper::ctz( bitfield )];
      }
      else
      {
        return nullptr;
      }
#else
      for ( unsigned i = 0; i < node_16->children_count_; ++i )
      {
        if ( node_16->key_[i] == c )
        {
          return &node_16->child_[i];
        }
      }
      return nullptr;
#endif
    }
    break;

    case CArtNode::Type::Fanout48:
    {
      CArtNode48 * node_48 = static_cast<CArtNode48 *>( node );
      if ( node_48->child_index_[c] != CArtNode48::EMPTY_MARKER )
      {
        return &node_48->child_[node_48->child_index_[c]];
      }
      else
      {
        return nullptr;
      }
    }
    break;

    case CArtNode::Type::Fanout256:
    {
      CArtNode256 * node_256 = static_cast<CArtNode256 *>( node );
      if ( node_256->child_[c] != CArtNode256::EMPTY_NODE )
      {
        return &( node_256->child_[c] );
      }
      else
      {
        return nullptr;
      }
    }
    break;

    default:
    {
      assert( false );
      return nullptr;
    }
    break;
  }
}

CArtNode **CAdaptiveRadixTree::InsertInNode(CArtNode ** base_node, uint8_t c, CArtNode * child_node )
{
  switch ( ( *base_node )->node_type_ )
  {
    case CArtNode::Type::Fanout4:
    {
      CArtNode4 * node = static_cast<CArtNode4 *>( *base_node );
      if ( node->children_count_ < 4 )
      {
        // Insert element by swapping if necessary.
        unsigned pos;
        for ( pos = 0; ( pos < node->children_count_ ) && ( node->key_[pos] < c ); ++pos )
          ;
        memmove( node->key_ + pos + 1, node->key_ + pos, node->children_count_ - pos );
        memmove( node->child_ + pos + 1, node->child_ + pos, ( node->children_count_ - pos ) * sizeof( uintptr_t ) );
        node->key_[pos] = c;
        node->child_[pos] = child_node;
        ++node->children_count_;
        return &node->child_[pos];
      }
      else
      {
        // Grow to CArtNode16
        CArtNode16 * newNode = new CArtNode16();

        *base_node = newNode;

        newNode->children_count_ = node->children_count_;
        newNode->prefix_length_ = node->prefix_length_;
        newNode->value_ = node->value_;
        newNode->prefix_position_ = node->prefix_position_;
        newNode->end_of_string_ = node->end_of_string_;

        for ( unsigned i = 0; i < 4; ++i )
        {
#if ENVIRONMENT_64
          newNode->key_[i] = detail::Helper::FlipSign( node->key_[i] );
#else
          newNode->key_[i] = node->key_[i];
#endif
        }
        memcpy( newNode->child_, node->child_, node->children_count_ * sizeof( uintptr_t ) );

        node->children_count_ = 0;  // prevent deletion of children
        delete node;
        return InsertInNode( base_node, c, child_node );
      }
    }
    break;

    case CArtNode::Type::Fanout16:
    {
      CArtNode16 * node = static_cast<CArtNode16 *>( *base_node );
      if ( node->children_count_ < 16 )
      {
        // Insert element
#if ENVIRONMENT_64
        uint8_t keyByteFlipped = detail::Helper::FlipSign( c );
        __m128i cmp = _mm_cmplt_epi8( _mm_set1_epi8( keyByteFlipped ),
                                      _mm_loadu_si128( reinterpret_cast<__m128i*>( node->key_ ) ) );
        uint16_t bitfield = _mm_movemask_epi8( cmp ) & ( 0xFFFF >> ( 16 - node->children_count_ ) );
        unsigned pos = bitfield ? detail::Helper::ctz( bitfield ) : node->children_count_;
#else
        uint8_t keyByteFlipped = c;
        unsigned pos;
        for ( pos = 0; ( pos < node->children_count_ ) && ( node->key_[pos] < c ); ++pos );
#endif /* ENVIRONMENT_64 */
        memmove( node->key_ + pos + 1, node->key_ + pos, node->children_count_ - pos );
        memmove( node->child_ + pos + 1, node->child_ + pos, ( node->children_count_ - pos ) * sizeof( uintptr_t ) );
        node->key_[pos] = keyByteFlipped;
        node->child_[pos] = child_node;
        ++node->children_count_;
        return &node->child_[pos];
      }
      else
      {
        // Grow to CArtNode48
        CArtNode48 * new_node = new CArtNode48();
        *base_node = new_node;
        memcpy( new_node->child_, node->child_, node->children_count_ * sizeof( uintptr_t ) );
        for ( unsigned i = 0; i < node->children_count_; ++i )
        {
#if ENVIRONMENT_64
          new_node->child_index_[detail::Helper::FlipSign( node->key_[i] )] = i;
#else
          new_node->child_index_[node->key_[i]] = i;
#endif
        }

        new_node->children_count_ = node->children_count_;
        new_node->prefix_length_ = node->prefix_length_;
        new_node->value_ = node->value_;
        new_node->prefix_position_ = node->prefix_position_;
        new_node->end_of_string_ = node->end_of_string_;

        node->children_count_ = 0;  // prevent deletion of children
        delete node;
        return InsertInNode( base_node, c, child_node );
      }
    }
    break;

    case CArtNode::Type::Fanout48:
    {
      CArtNode48 * node = static_cast<CArtNode48 *>( *base_node );
      if ( node->children_count_ < 48 )
      {
        // Insert element
        unsigned pos = node->children_count_;
        if ( node->child_[pos] )
        {
          for ( pos = 0; node->child_[pos] != NULL; ++pos )
            ;
        }
        node->child_[pos] = child_node;
        node->child_index_[c] = pos;
        ++node->children_count_;
        return &node->child_[pos];
      }
      else
      {
        // Grow to Node256
        CArtNode256 * newNode = new CArtNode256();
        for ( unsigned i = 0; i < 256; ++i )
        {
          if ( node->child_index_[i] != 48 )
          {
            newNode->child_[i] = node->child_[node->child_index_[i]];
          }
        }

        newNode->children_count_ = node->children_count_;
        newNode->prefix_length_ = node->prefix_length_;
        newNode->value_ = node->value_;
        newNode->prefix_position_ = node->prefix_position_;
        newNode->end_of_string_ = node->end_of_string_;

        *base_node = newNode;

        node->children_count_ = 0;  // prevent deletion of children
        delete node;
        return InsertInNode( base_node, c, child_node );
      }
    }
    break;

    case CArtNode::Type::Fanout256:
    {
      CArtNode256 * node = static_cast<CArtNode256 *>( *base_node );
      ++node->children_count_;
      node->child_[(uint8_t)c] = child_node;
      return &node->child_[(uint8_t)c];
    }
    break;

    default:
    {
      assert( false );
      return nullptr;
    }
    break;
  }
}

void CAdaptiveRadixTree::InsertValue(CArtNode ** node_base, CArtNode * node, uint32_t value )
{
  if ( !( node->end_of_string_ ) )
  {
    node->end_of_string_ = true;
    ++unique_string_count_;  //< first insertion.
  }

  uint32_t index = node->value_;
  assert( value < indexes_->size() );  // we may resize the index vector anyway, but we have to synchronize it
  // because we are using shared index vector for joinable ARTs.
  indexes_->operator[]( value ) = index;
  node->value_ = value;
}

void CAdaptiveRadixTree::AddEntry(const char* key, size_t key_length, uint32_t value )
{
  total_string_length_ += key_length;
  max_string_length_ = std::max( max_string_length_, key_length );

  CArtNode ** node_base = &root_;
  size_t depth = 0;
  size_t mismatch_position = 0;

  CArtNode * node = *node_base;

  while ( node )
  {
    // how much of prefix matches with key?
    for ( ; depth + mismatch_position < key_length && mismatch_position < node->prefix_length_; ++mismatch_position )
    {
      if ( key[depth + mismatch_position] != suffix_table_[node->prefix_position_ + mismatch_position] )
      {
        break;
      }
    }

    // if all of prefix is matched with key, that means we found end of string so insert numeric part!
    if ( depth + mismatch_position == key_length && mismatch_position == node->prefix_length_ )
    {
      InsertValue( node_base, node, value );
      return;
    }

    // only part of prefix is matched with key. {key: alize, prefix: alt} => mismatched_position: 2
    if ( mismatch_position < node->prefix_length_ )
    {
      CArtNode * new_node = new CArtNode4();

      *node_base = new_node;

      // if at least one char is matched between key and prefix, assign this part to new node as prefix.
      if ( mismatch_position != 0 )
      {
        new_node->prefix_position_ = node->prefix_position_;
        new_node->prefix_length_ = mismatch_position;

        node->prefix_length_ -= mismatch_position;
        node->prefix_position_ += mismatch_position;
      }

      // handle unmatched prefix part
      // use the same node (updated its prefix info) as child of new node.
      InsertInNode( node_base, suffix_table_[node->prefix_position_], node );
      --node->prefix_length_;
      ++node->prefix_position_;

      // handle unmatched key part
      if ( depth + mismatch_position < key_length )
      {
        // add unmatched key part as separate Node4 & continue.
        size_t key_offset = depth + mismatch_position + 1; // +1 for addressing char.
        size_t remaining_length = key_length - key_offset;
        CArtNode * new_node = new CArtNode4();
        new_node->prefix_length_ = static_cast<uint32_t>( remaining_length );

        if ( remaining_length )
        {
          new_node->prefix_position_ = static_cast<uint32_t>( suffix_table_.size() );
          suffix_table_.append( key + key_offset, remaining_length );
        }

        node_base = InsertInNode( node_base, key[depth + mismatch_position], new_node );
        depth += mismatch_position + 1;
        mismatch_position = 0;
        node = *node_base;
        continue;
      }
      else
      {
        InsertValue( node_base, new_node, value );
        return;
      }
    }
    // prefix data is subsumed by key => {key: alize, prefix: ali} => mismatched_position: 3
    else if ( depth + mismatch_position < key_length )
    {
      CArtNode ** result = FindChild(node, key[depth + mismatch_position] );

      if ( result )  // if child exists we will continue to match its content with remaining key.
      {
        node_base = result;
        depth += mismatch_position + 1;
        mismatch_position = 0;
        node = *node_base;
        continue;
      }
      else  // child does not exists, create&insert a node and continue on that.
      {
        CArtNode * new_node = new CArtNode4();
        size_t key_offset = depth + mismatch_position + 1; // +1 for addressing char.
        new_node->prefix_length_ = static_cast<uint32_t>( key_length - key_offset );

        if ( new_node->prefix_length_ )
        {
          new_node->prefix_position_ = static_cast<uint32_t>( suffix_table_.size() );
          suffix_table_.append( key + key_offset, new_node->prefix_length_ );
        }

        node_base = InsertInNode( node_base, key[depth + mismatch_position], new_node );
        depth += mismatch_position + 1;
        mismatch_position = 0;
        node = *node_base;
        // todo: when we insert a new node and assign everything left of key, we can skip comparison at the beginning
        // of loop.
        continue;
      }
    }
  }
}

std::unique_ptr<CAdaptiveRadixTree> CAdaptiveRadixTree::Split()
{
  return std::make_unique <CAdaptiveRadixTree> (this->indexes_);
}

// todo(demiroz): compare performance with version using stack structure.
void CAdaptiveRadixTree::TraverseRecursive(CArtNode * iNode, CActionBase & action, std::string& key, int level ) const
{
  action.HandleNode( iNode, key, level );

  if ( iNode->prefix_length_ )
  {
    key.append( suffix_table_, iNode->prefix_position_, iNode->prefix_length_ );
    level += iNode->prefix_length_;
  }

  if ( iNode->end_of_string_ )
  {
    action.HandleTuple( key, CIndexIterator(*indexes_, iNode->value_ ),
                        CIndexIterator(*indexes_, CArtNode::LAST_INDEX_IDENTIFIER ) );
  }

  switch ( iNode->node_type_ )
  {
    case CArtNode::Type::Fanout4:
    {
      auto node = static_cast<CArtNode4 *>( iNode );

      for ( int i = 0; i < node->children_count_; ++i )
      {
        key.push_back( node->key_[i] );
        TraverseRecursive( node->child_[i], action, key, level + 1 );
        key.resize( level );
      }
    }
    break;

    case CArtNode::Type::Fanout16:
    {
      auto node = static_cast<CArtNode16 *>( iNode );

      for ( int i = 0; i < node->children_count_; ++i )
      {
#if ENVIRONMENT_64
        auto ch = detail::Helper::FlipSign( node->key_[i] );
#else
        auto ch = node->key_[i];
#endif
        if ( !ch )
        {
          continue;
        }

        key.push_back( ch );
        if ( node->child_[i] )
        {
          TraverseRecursive( node->child_[i], action, key, level + 1 );
        }
        key.resize( level );
      }
    }
    break;

    case CArtNode::Type::Fanout48:
    {
      auto node = static_cast<CArtNode48 *>( iNode );

      if ( node->children_count_ )
      {
        for ( int i = 0; i < 256; ++i )
        {
          if ( node->child_index_[i] != CArtNode48::EMPTY_MARKER )
          {
            key.push_back( static_cast<char>( i ) );
            if ( node->child_[node->child_index_[i]] )
            {
              TraverseRecursive( node->child_[node->child_index_[i]], action, key, level + 1 );
            }
            key.resize( level );
          }
        }
      }
    }
    break;

    case CArtNode::Type::Fanout256:
    {
      auto node = static_cast<CArtNode256 *>( iNode );

      if ( node->children_count_ )
      {
        for ( int i = 0; i < 256; ++i )
        {
          if ( node->child_[i] != CArtNode256::EMPTY_NODE )  //< node is different than empty node
          {
            key.push_back( static_cast<char>( i ) );
            TraverseRecursive( node->child_[i], action, key, level + 1 );
            key.resize( level );
          }
        }
      }
    }
    break;
  }
}

void CAdaptiveRadixTree::TraverseIndexRecursive(CArtNode * iNode, CIndexActionBase & action ) const
{
  if ( iNode->end_of_string_ )
  {
    action.HandleTuple(CIndexIterator(*indexes_, iNode->value_ ), CIndexIterator(*indexes_, CArtNode::LAST_INDEX_IDENTIFIER ) );
  }

  switch ( iNode->node_type_ )
  {
    case CArtNode::Type::Fanout4:
    {
      auto node = static_cast<CArtNode4 *>( iNode );

      for ( int i = 0; i < node->children_count_; ++i )
      {
        TraverseIndexRecursive( node->child_[i], action );
      }
    }
    break;

    case CArtNode::Type::Fanout16:
    {
      auto node = static_cast<CArtNode16 *>( iNode );

      for ( int i = 0; i < node->children_count_; ++i )
      {
#if ENVIRONMENT_64
        auto ch = detail::Helper::FlipSign( node->key_[i] );
#else
        auto ch = node->key_[i];
#endif
        if ( !ch )
        {
          continue;
        }

        if ( node->child_[i] )
        {
          TraverseIndexRecursive( node->child_[i], action );
        }
      }
    }
    break;

    case CArtNode::Type::Fanout48:
    {
      auto node = static_cast<CArtNode48 *>( iNode );

      if ( node->children_count_ )
      {
        for ( int i = 0; i < 256; ++i )
        {
          if ( node->child_index_[i] != CArtNode48::EMPTY_MARKER )
          {
            if ( node->child_[node->child_index_[i]] )
            {
              TraverseIndexRecursive( node->child_[node->child_index_[i]], action );
            }
          }
        }
      }
    }
    break;

    case CArtNode::Type::Fanout256:
    {
      auto node = static_cast<CArtNode256 *>( iNode );

      if ( node->children_count_ )
      {
        for ( int i = 0; i < 256; ++i )
        {
          if ( node->child_[i] != CArtNode256::EMPTY_NODE )  //< node is different than empty node
          {
            TraverseIndexRecursive( node->child_[i], action );
          }
        }
      }
    }
    break;
  }
}

void CAdaptiveRadixTree::Reset()
{
  if ( root_ )
  {
    detail::Helper::DeleteNode(root_ );
  }

  root_ = new CArtNode256();

  null_string_ = CArtNode::LAST_INDEX_IDENTIFIER;
  null_string_count_ = 0;
  unique_string_count_ = 0;
  std::string().swap( suffix_table_ );
}

void CAdaptiveRadixTree::MovePrefix(CArtNode * input_node, std::string& other_suffix_table )
{
  if ( input_node->prefix_length_ )
  {
    size_t new_prefix_position = suffix_table_.size();
    suffix_table_.append( other_suffix_table, input_node->prefix_position_, input_node->prefix_length_ );
    input_node->prefix_position_ = new_prefix_position;
  }

  // no need to do anything for terminator nodes except increasing unique string count.
  if ( input_node->end_of_string_ )
  {
    ++unique_string_count_;
  }

  switch ( input_node->node_type_ )
  {
    case CArtNode::Type::Fanout4:
    {
      auto node = static_cast<CArtNode4 *>( input_node );

      for ( int i = 0; i < node->children_count_; ++i )
      {
        MovePrefix( node->child_[i], other_suffix_table );
      }
    }
    break;

    case CArtNode::Type::Fanout16:
    {
      auto node = static_cast<CArtNode16 *>( input_node );

      for ( int i = 0; i < node->children_count_; ++i )
      {
#if ENVIRONMENT_64
        auto ch = detail::Helper::FlipSign( node->key_[i] );
#else
        auto ch = node->key_[i];
#endif
        if ( !ch )
        {
          continue;
        }

        if ( node->child_[i] )
        {
          MovePrefix( node->child_[i], other_suffix_table );
        }
      }
    }
    break;

    case CArtNode::Type::Fanout48:
    {
      auto node = static_cast<CArtNode48 *>( input_node );

      if ( node->children_count_ )
      {
        for ( int i = 0; i < 256; ++i )
        {
          if ( node->child_index_[i] != CArtNode48::EMPTY_MARKER )
          {
            if ( node->child_[node->child_index_[i]] )
            {
              MovePrefix( node->child_[node->child_index_[i]], other_suffix_table );
            }
          }
        }
      }
    }
    break;

    case CArtNode::Type::Fanout256:
    {
      auto node = static_cast<CArtNode256 *>( input_node );

      if ( node->children_count_ )
      {
        for ( int i = 0; i < 256; ++i )
        {
          if ( node->child_[i] != CArtNode256::EMPTY_NODE )  //< node is different than empty node
          {
            MovePrefix( node->child_[i], other_suffix_table );
          }
        }
      }
    }
    break;
  }
}

void CAdaptiveRadixTree::Merge(CArtNode ** left, CArtNode ** right, std::string& right_suffix_table_ )
{
  size_t mismatch_position = 0;
  CArtNode ** node_base = left;
  CArtNode * node_left = *left;
  CArtNode * node_right = *right;

  // how much of prefix matches with left prefix?
  for ( ; mismatch_position < node_left->prefix_length_ && mismatch_position < node_right->prefix_length_;
        ++mismatch_position )
  {
    if ( suffix_table_[node_left->prefix_position_ + mismatch_position] !=
         right_suffix_table_[node_right->prefix_position_ + mismatch_position] )
    {
      break;
    }
  }

  if ( mismatch_position == node_left->prefix_length_ && mismatch_position == node_right->prefix_length_ )
  {
    // prefix is ok, handle child nodes.
    MergeChildNodes( left, node_right, right_suffix_table_ );
    node_right->children_count_ = 0;
    detail::Helper::DeleteNode(node_right );
    *right = nullptr;
    return;
  }

  // only part of left prefix is matched with right prefix. {left prefix: alt, right prefix: alize, } =>
  // mismatched_position: 2
  if ( mismatch_position < node_left->prefix_length_ )
  {
    CArtNode * new_node = new CArtNode4();
    *node_base = new_node;

    // if at least one char is matched between left and right prefix, assign this part to new node as prefix.
    if ( mismatch_position != 0 )
    {
      new_node->prefix_position_ = node_left->prefix_position_;
      new_node->prefix_length_ = mismatch_position;

      node_left->prefix_length_ -= mismatch_position;
      node_left->prefix_position_ += mismatch_position;

      node_right->prefix_length_ -= mismatch_position;
      node_right->prefix_position_ += mismatch_position;
    }

    // handle unmatched left prefix part
    // use the same node (updated its prefix info) as child of new node.
    InsertInNode( node_base, suffix_table_[node_left->prefix_position_], node_left );
    --node_left->prefix_length_;
    ++node_left->prefix_position_;

    // handle unmatched right prefix part
    if ( node_right->prefix_length_ > 0 )
    {
      // add unmatched key part as separate Node4 & continue.
      --node_right->prefix_length_;                                                // -1 for addressing char.
      char addressing_char = right_suffix_table_[node_right->prefix_position_++];  // +1 for addressing char

      MovePrefix( node_right, right_suffix_table_ );
      node_base = InsertInNode( node_base, addressing_char, node_right );
      return;
    }
    else
    {
      // add all right child nodes to new node.
      MergeChildNodes( node_base, node_right, right_suffix_table_ );
      node_right->children_count_ = 0;
      detail::Helper::DeleteNode(node_right );
      *right = nullptr;
      return;
    }
  }
  // left prefix data is subsumed by right prefix data => {left prefix: ali, right prefix: alize} =>
  // mismatched_position: 3
  else if ( mismatch_position < node_right->prefix_length_ )
  {
    node_right->prefix_length_ -= mismatch_position;
    node_right->prefix_position_ += mismatch_position;

    char addressing_char = right_suffix_table_[node_right->prefix_position_];
    CArtNode ** left_child = FindChild(node_left, addressing_char );

    if ( left_child )  // if child exists we will continue to match its content with remaining key.
    {
      ++node_right->prefix_position_;  // discard addressing character
      --node_right->prefix_length_;
      Merge( left_child, right, right_suffix_table_ );
      return;
    }
    else  // child does not exists, insert right into left as child.
    {
      ++node_right->prefix_position_;  // discard addressing character
      --node_right->prefix_length_;

      MovePrefix( node_right, right_suffix_table_ );
      node_base = InsertInNode( node_base, addressing_char, node_right );
      return;
    }
  }
  assert( false );
}

void CAdaptiveRadixTree::MergeChildNodes(CArtNode ** left, CArtNode * right, std::string& right_suffix_table_ )
{
  if ( right->end_of_string_ )
  {
    auto it = CIndexIterator(*indexes_, right->value_ ), end = CIndexIterator(*indexes_, CArtNode::LAST_INDEX_IDENTIFIER );
    for ( ; it != end; )
    {
      auto value = *it;  // cache it!
      ++it;
      InsertValue( left, *left, value );
    }
  }

  switch ( right->node_type_ )
  {
    case CArtNode::Type::Fanout4:
    {
      auto right_node = static_cast<CArtNode4 *>( right );

      if ( right_node->children_count_ )
      {
        for ( int i = 0; i < right_node->children_count_; ++i )
        {
          CArtNode ** left_child = FindChild(*left, right_node->key_[i] );
          // if child exists we will continue to match its content with remaining key.
          if ( left_child )
          {
            Merge( left_child, &( right_node->child_[i] ), right_suffix_table_ );
          }
          else
          {
            MovePrefix( right_node->child_[i], right_suffix_table_ );
            // insert unique child in current left node as child.
            InsertInNode( left, right_node->key_[i], right_node->child_[i] );
          }
        }
      }
    }
    break;

    case CArtNode::Type::Fanout16:
    {
      auto right_node = static_cast<CArtNode16 *>( right );

      if ( right_node->children_count_ )
      {
        for ( int i = 0; i < right_node->children_count_; ++i )
        {
#if ENVIRONMENT_64
          auto ch = detail::Helper::FlipSign( right_node->key_[i] );
#else
          auto ch = right_node->key_[i];
#endif
          if ( !ch )
          {
            continue;
          }

          if ( right_node->child_[i] )
          {
            CArtNode ** left_child = FindChild(*left, ch );
            // if child exists we will continue to match its content with remaining key.
            if ( left_child )
            {
              Merge( left_child, &( right_node->child_[i] ), right_suffix_table_ );
            }
            else
            {
              MovePrefix( right_node->child_[i], right_suffix_table_ );
              InsertInNode( left, ch, right_node->child_[i] );  // insert unique child in current left node as child.
            }
          }
        }
      }
    }
    break;

    case CArtNode::Type::Fanout48:
    {
      auto right_node = static_cast<CArtNode48 *>( right );

      if ( right_node->children_count_ )
      {
        for ( int i = 0; i < 256; ++i )
        {
          if ( right_node->child_index_[i] != CArtNode48::EMPTY_MARKER )
          {
            CArtNode ** left_child = FindChild(*left, (char)i );
            // if child exists we will continue to match its content with remaining key.
            if ( left_child )
            {
              Merge( left_child, &( right_node->child_[right_node->child_index_[i]] ), right_suffix_table_ );
            }
            else
            {
              MovePrefix( right_node->child_[right_node->child_index_[i]], right_suffix_table_ );
              // insert unique child in current left node as child.
              InsertInNode( left, i, right_node->child_[right_node->child_index_[i]] );
            }
          }
        }
      }
    }
    break;

    case CArtNode::Type::Fanout256:
    {
      auto right_node = static_cast<CArtNode256 *>( right );

      if ( right_node->children_count_ )
      {
        for ( int i = 0; i < 256; ++i )
        {
          if ( right_node->child_[i] != CArtNode256::EMPTY_NODE )  //< node is different than empty node
          {
            CArtNode ** left_child = FindChild(*left, (char)i );
            // if child exists we will continue to match its content with remaining key.
            if ( left_child )
            {
              Merge( left_child, &( right_node->child_[i] ), right_suffix_table_ );
            }
            else
            {
              MovePrefix( right_node->child_[i], right_suffix_table_ );
              // insert unique child in current left node as child.
              InsertInNode( left, (char)i, right_node->child_[i] );
            }
          }
        }
      }
    }
    break;
  }
}

CAdaptiveRadixTree::~CAdaptiveRadixTree() {
  if ( root_ )
  {
    detail::Helper::DeleteNode(root_ );
  }
}
