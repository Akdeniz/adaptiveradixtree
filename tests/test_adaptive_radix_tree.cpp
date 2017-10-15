#include <gtest/gtest.h>

#include <iterator>
#include <random>
#include <string>
#include <utility>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "adaptive_radix_tree.hpp"
#include "utils.hpp"

namespace
{
struct TestParam
{
  size_t seed;
  int string_count;
  size_t min_string_length;
  size_t max_string_length;
};
}

class ConstructARTWithRandomStrings : public testing::TestWithParam<TestParam>
{
protected:
  ConstructARTWithRandomStrings()
      : string_pos_generator_( 0, alphabet_.size() - 1 ),
        string_length_generator_( GetParam().min_string_length, GetParam().max_string_length ),
        tree_( GetParam().string_count ),
        rng_( GetParam().seed )
  {
    std::string str;
    str.reserve( GetParam().max_string_length );

    for ( int i = 0; i < GetParam().string_count; ++i )
    {
      auto str_len = string_length_generator_( rng_ );
      str.resize( str_len );
      std::generate_n( str.begin(), str.size(), [&]() -> char { return alphabet_[string_pos_generator_( rng_ )]; } );
      tree_.AddEntry( str.c_str(), str.size(), i );
      values_[str].push_back( i );
    }
  }

protected:
  std::string alphabet_ = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::uniform_int_distribution<size_t> string_pos_generator_;
  std::uniform_int_distribution<size_t> string_length_generator_;
  CAdaptiveRadixTree tree_;
  std::default_random_engine rng_;
  std::map<std::string, std::vector<int>> values_;
};

TEST_P( ConstructARTWithRandomStrings, TraverseCheck )
{
  class CTraverser : public CActionBase
  {
    virtual void HandleNode( CArtNode const*, std::string const&, uint32_t )
    {
    }
    virtual void HandleTuple( std::string const& str, CIndexIterator begin, CIndexIterator end )
    {
      ++unique_string_count;
      auto length = std::distance( begin, end );
      index_count += length;
      total_char_count += ( str.size() * length );
    }

  public:
    size_t total_char_count = 0;
    size_t unique_string_count = 0;
    size_t index_count = 0;
  };

  CTraverser traverser;
  tree_.Traverse( traverser );

  ASSERT_EQ( values_.size(), traverser.unique_string_count );

  size_t values_total_char_count = 0;
  size_t values_index_count = 0;
  for ( auto it = values_.begin(); it != values_.end(); ++it )
  {
    values_index_count += it->second.size();
    values_total_char_count += ( it->first.size() * it->second.size() );
  }

  ASSERT_EQ( values_index_count, traverser.index_count );
  ASSERT_EQ( values_total_char_count, traverser.total_char_count );
}

INSTANTIATE_TEST_CASE_P( ConstructARTWithRandomStringsInstantiation, ConstructARTWithRandomStrings,
                         ::testing::Values<TestParam>( TestParam{0xDEADBEEF, 1000000, 5, 1000},
                                                       TestParam{std::random_device()(), 100000, 50, 100} ) );
