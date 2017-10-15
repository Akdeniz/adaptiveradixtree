#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

#include <cstdint>
#include <cstring>

struct CArtNode
{
  // Represents final index for tuples.
  static const uint32_t LAST_INDEX_IDENTIFIER = -1;

  enum Type
  {
    Fanout4,
    Fanout16,
    Fanout48,
    Fanout256,
  };

  explicit CArtNode( Type type )
      : prefix_length_( 0 ),
        prefix_position_( 0 ),
        value_( LAST_INDEX_IDENTIFIER ),
        children_count_( 0 ),
        node_type_( type ),
        end_of_string_( false )
  {
  }

  uint32_t prefix_length_;
  uint32_t prefix_position_;  //< prefix position in suffix table.
  uint32_t value_;            //< only meaningful if end of string.
  uint16_t children_count_;
  uint8_t node_type_;
  bool end_of_string_;
};

struct CArtNode4: CArtNode
{
  CArtNode4() : CArtNode(CArtNode::Type::Fanout4 )
  {
    memset( key_, 0, sizeof( key_ ) );
    memset( child_, 0, sizeof( child_ ) );
  }
  ~CArtNode4();

  uint8_t key_[4];
  CArtNode * child_[4];
};

struct CArtNode16: CArtNode
{
  CArtNode16() : CArtNode(CArtNode::Type::Fanout16 )
  {
    memset( key_, 0, sizeof( key_ ) );
    memset( child_, 0, sizeof( child_ ) );
  }
  ~CArtNode16();

  uint8_t key_[16];
  CArtNode * child_[16];
};

struct CArtNode48: CArtNode
{
  CArtNode48() : CArtNode(CArtNode::Type::Fanout48 )
  {
    memset( child_index_, EMPTY_MARKER, sizeof( child_index_ ) );
    memset( child_, 0, sizeof( child_ ) );
  }
  ~CArtNode48();

  static const uint8_t EMPTY_MARKER = 48;

  uint8_t child_index_[256];
  CArtNode * child_[48];
};

struct CArtNode256: CArtNode
{
  CArtNode256() : CArtNode(CArtNode::Type::Fanout256 )
  {
    memset( child_, ~0, sizeof( child_ ) );
  }
  ~CArtNode256();

  static CArtNode * EMPTY_NODE;

  CArtNode * child_[256];
};

namespace detail {
struct Helper {
  static void DeleteNode(CArtNode *node) {
    switch (node->node_type_) {
      case CArtNode::Fanout4:
        delete static_cast<CArtNode4 *>(node);
        break;
      case CArtNode::Fanout16:
        delete static_cast<CArtNode16 *>(node);
        break;
      case CArtNode::Fanout48:
        delete static_cast<CArtNode48 *>(node);
        break;
      case CArtNode::Fanout256:
        delete static_cast<CArtNode256 *>(node);
        break;
    }
  }

  static uint8_t FlipSign(uint8_t keyByte) {
    // Flip the sign bit, enables signed SSE comparison of unsigned values, used by CArtNode16
    return keyByte ^ 128;
  }

  static unsigned ctz(uint16_t x) {
// Count trailing zeros, only defined for x>0
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    // Adapted from Hacker's Delight
    unsigned n = 1;
    if ( ( x & 0xFF ) == 0 )
    {
      n += 8;
      x = x >> 8;
    }
    if ( ( x & 0x0F ) == 0 )
    {
      n += 4;
      x = x >> 4;
    }
    if ( ( x & 0x03 ) == 0 )
    {
      n += 2;
      x = x >> 2;
    }
    return n - ( x & 1 );
#endif
  }
};
} //< ns detail

inline CArtNode4::~CArtNode4()
{
  for ( int i = 0; i < children_count_; ++i )
  {
    detail::Helper::DeleteNode(child_[i] );
  }
}

inline CArtNode16::~CArtNode16()
{
  for ( int i = 0; i < children_count_; ++i )
  {
#if BOOST_ARCH_X86_64
    auto ch = detail::Helper::FlipSign( key_[i] );
#else
    auto ch = key_[i];
#endif
    if ( !ch )
    {
      continue;
    }
    detail::Helper::DeleteNode(child_[i] );
  }
}

inline CArtNode48::~CArtNode48()
{
  if ( children_count_ )
  {
    for ( int i = 0; i < 256; ++i )
    {
      if ( child_index_[i] != CArtNode48::EMPTY_MARKER )
      {
        detail::Helper::DeleteNode(child_[child_index_[i]] );
      }
    }
  }
}

inline CArtNode256::~CArtNode256()
{
  if ( children_count_ )
  {
    for ( int i = 0; i < 256; ++i )
    {
      if ( child_[i] != CArtNode256::EMPTY_NODE )
      {
        detail::Helper::DeleteNode(child_[i] );
      }
    }
  }
}
