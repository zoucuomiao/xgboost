// Copyright by Contributors
#include <xgboost/objective.h>

#include "../helpers.h"

#include <random>
#include <limits>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>

int test_rnd() {
  std::mt19937 rnd(1111);
  double sum = 0;
  for (int i = 0; i < 100; i++ ) {
    int val = std::uniform_int_distribution<unsigned>(0, 10)(rnd);
    sum += val;
    std::cout << val << ",";
  }
  std::cout << "\n";
  std::cout << "Sum : " << sum << "\n";
  return 0;
}

TEST(PairwiseRankObj, GPair) {
  DisableOpenMP();
  test_rnd();
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
  DisableOpenMP();
  test_rnd();
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

TEST(LambdaRankObjMAP, GPair) {
  DisableOpenMP();
  test_rnd();
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("rank:map");
  std::vector<std::pair<std::string, std::string> > args;
  args.push_back(std::make_pair("num_pairsample", "3"));
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,    0.1,   0.9,     1,      0,    0.1,    0.9,      1},
                   {   0,      0,     0,     0,      1,      1,      1,      1},
                   {   1,      1,     1,     1,      1,      1,      1,      1},
                   {0.031, 0.029, 0.085, 0.545, -0.267, -0.306, -0.062, -0.055},
                   {0.041, 0.036, 0.058, 0.330,  0.148,  0.185,  0.064,  0.067},
                   0.001);
}
