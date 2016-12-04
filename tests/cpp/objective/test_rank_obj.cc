// Copyright by Contributors
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(PairwiseRankObj, GPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("rank:pairwise");
  std::vector<std::pair<std::string, std::string> > args;
  args.push_back(std::make_pair("num_pairsample", "2"));
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1,  0.9,    1,     0,   0.1,   0.9,     1},
                   {   0,    0,    0,    0,     1,     1,     1,     1},
                   {   1,    1,    1,    1,     1,     1,     1,     1},
                   {0.52, 0.64, 1.29, 2.07, -1.70, -1.90, -0.65, -0.27},
                   {0.65, 0.70, 0.88, 1.25,  1.04,  1.33,  0.70,  0.40});

  args.clear();
  args.push_back(std::make_pair("fix_list_weight", "0.1"));
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,    1,     0,     1},
                   {   0,    0,     1,     1},
                   {   1,    1,     1,     1},
                   {0.02, 0.03, -0.04, -0.01},
                   {0.02, 0.02,  0.03,  0.01});

  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

TEST(LambdaRankObjNDCG, GPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("rank:ndcg");
  std::vector<std::pair<std::string, std::string> > args;
  args.push_back(std::make_pair("num_pairsample", "5"));
  obj->Configure(args);
  CheckObjFunction(obj,
                   {    0,   0.1,   0.9,     1,      0,    0.1,    0.9,      1},
                   {    0,     0,     0,     0,      1,      1,      1,      1},
                   {    1,     1,     1,     1,      1,      1,      1,      1},
                   {0.045, 0.035, 0.062, 0.619, -0.364, -0.234, -0.058, -0.106},
                   {0.063, 0.045, 0.048, 0.374,  0.201,  0.139,  0.061,  0.129},
                   0.001);
}
