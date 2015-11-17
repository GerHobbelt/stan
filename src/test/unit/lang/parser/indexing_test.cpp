#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(LangGrammarsIndexing, goodLb) {
  test_parsable("indexing/good-lb");
}
TEST(LangGrammarsIndexing, goodUb) {
  test_parsable("indexing/good-ub");
}
TEST(LangGrammarsIndexing, goodLub) {
  test_parsable("indexing/good-lub");
}
TEST(LangGrammarsIndexing, goodInts) {
  test_parsable("indexing/good-ints");
}
TEST(LangGrammarsIndexing, goodOmni) {
  test_parsable("indexing/good-omni");
}
TEST(LangGrammarsIndexing, goodOmni2) {
  test_parsable("indexing/good-omni2");
}
TEST(LangGrammarsIndexing, goodOmni3) {
  test_parsable("indexing/good-omni3");
}
