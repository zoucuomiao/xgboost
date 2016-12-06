// Copyright by Contributors
#include "../../src/tree/updater_colmaker.cc"
#include <xgboost/data.h>

#include "../helpers.h"

TEST(UpdaterColMaker, Basic) {
  std::string tmp_file = Create5NodeModellingData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());
  std::vector<bool> enable(dmat->info().num_col, true);
  dmat->InitColAccess(enable, 1, dmat->info().num_row);
  ASSERT_EQ(dmat->info().num_col, 2);
  ASSERT_EQ(dmat->info().num_row, 10);

  std::vector<std::pair<std::string, std::string> > cfg;
  cfg.push_back(std::make_pair("max_depth", "3"));
  cfg.push_back(std::make_pair("num_feature", "2"));

  std::vector<std::string> updaters = {"grow_colmaker", "distcol"};
  for (int i = 0; i < updaters.size(); ++i) {
    xgboost::TreeUpdater * up = xgboost::TreeUpdater::Create(updaters[i]);
    std::cout << "Running updater_colmaker with: " << updaters[i];
    up->Init(cfg);

    xgboost::RegTree tree;
    tree.InitModel();
    tree.param.InitAllowUnknown(cfg);
    std::vector<xgboost::RegTree*> new_trees(1);
    new_trees[0] = &tree;

    // use objective function for regression tree where hessian is 1
    // and gradient = pred - label. As pred is 0, grad = -label
    std::vector<xgboost::bst_gpair> gpair(dmat->info().num_row);
    for (int i = 0; i < dmat->info().num_row; ++i) {
      gpair[i] = xgboost::bst_gpair(-dmat->info().labels[i], 1);
    }

    up->Update(gpair, dmat, new_trees);

    // test the tree structure which was built
    ASSERT_EQ(tree.param.num_nodes, 5);
    EXPECT_TRUE(tree[0].is_root());
    ASSERT_FALSE(tree[0].is_leaf());
    EXPECT_EQ(tree[0].split_index(), 0);
    EXPECT_FLOAT_EQ(tree[0].split_cond(), 0.25);
    EXPECT_TRUE(tree[tree[0].cleft()].is_leaf());
    EXPECT_FLOAT_EQ(tree[tree[0].cleft()].leaf_value(), 0);
    ASSERT_FALSE(tree[tree[0].cright()].is_leaf());
    EXPECT_FLOAT_EQ(tree[tree[0].cright()].split_cond(), 0.75);
    EXPECT_EQ(tree[tree[0].cright()].split_index(), 0);
    EXPECT_TRUE(tree[tree[tree[0].cright()].cleft()].is_leaf());
    EXPECT_TRUE(tree[tree[tree[0].cright()].cright()].is_leaf());
    EXPECT_FLOAT_EQ(tree[tree[tree[0].cright()].cleft()].leaf_value(), 0.25);
    EXPECT_FLOAT_EQ(tree[tree[tree[0].cright()].cright()].leaf_value(), 0);

    delete up;
  }
}
