// Copyright by Contributors
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(LinearRegression, GPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:linear");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {0, 0.1, 0.9,   1,    0,  0.1, 0.9,  1},
                   {0,   0,   0,   0,    1,    1,    1, 1},
                   {1,   1,   1,   1,    1,    1,    1, 1},
                   {0, 0.1, 0.9, 1.0, -1.0, -0.9, -0.1, 0},
                   {1,   1,   1,   1,    1,    1,    1, 1});
}

TEST(LogisticRegression, GPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:logistic");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1,  0.9,    1,    0,   0.1,  0.9,      1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5, 0.52, 0.71, 0.73, -0.5, -0.47, -0.28, -0.26},
                   {0.25, 0.24, 0.20, 0.19, 0.25,  0.24,  0.20,  0.19});
}

TEST(LogisticRaw, GPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("binary:logitraw");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1,  0.9,    1,    0,   0.1,   0.9,     1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5, 0.52, 0.71, 0.73, -0.5, -0.47, -0.28, -0.26},
                   {0.25, 0.24, 0.20, 0.19, 0.25,  0.24,  0.20,  0.19});
}
