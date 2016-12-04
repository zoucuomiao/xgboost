// Copyright by Contributors
#include <xgboost/tree_model.h>

#include "../helpers.h"

TEST(RegTree, TreeCreationDeletion) {
  xgboost::RegTree tree;

  // Initialize the root node
  tree.InitModel();
  EXPECT_EQ(tree.param.num_nodes, 1);
  EXPECT_TRUE(tree[0].is_root());
  EXPECT_TRUE(tree[0].is_leaf());
  EXPECT_EQ(tree.GetDepth(0), 0);

  // Set a split condition on root node
  tree[0].set_split(1, 0.5, true);
  tree.stat(0).loss_chg = 10;
  tree.stat(0).sum_hess = 100;
  tree.stat(0).base_weight = 50;
  EXPECT_EQ(tree[0].split_index(), 1);
  EXPECT_TRUE(tree[0].default_left());
  EXPECT_EQ(tree[0].split_cond(), 0.5);

  // Add depth 1 to root node
  tree.AddChilds(0);
  EXPECT_EQ(tree[0].cleft(), 1);
  EXPECT_EQ(tree[0].cright(), 2);
  EXPECT_TRUE(tree[tree[0].cleft()].is_left_child());
  EXPECT_FALSE(tree[tree[0].cright()].is_left_child());
  EXPECT_FALSE(tree[0].is_leaf());
  tree[1].set_leaf(1);
  tree[2].set_split(1, 0.75);
  EXPECT_EQ(tree.GetDepth(1), 1);
  EXPECT_EQ(tree.GetDepth(1, true), 1);

  // Create a second level branch at node 2
  tree.AddChilds(2);
  EXPECT_EQ(tree[2].cleft(), 3);
  EXPECT_EQ(tree[2].cright(), 4);
  EXPECT_EQ(tree[2].cdefault(), tree[2].cright());
  EXPECT_FALSE(tree[2].is_leaf());
  tree[3].set_leaf(0);
  tree[4].set_leaf(1);
  EXPECT_TRUE(tree[3].is_leaf());

  // Delete the second level branch at node 2
  tree.CollapseToLeaf(2, 0.5);
  EXPECT_TRUE(tree[2].is_leaf());
  EXPECT_EQ(tree[2].leaf_value(), 0.5);

  // Write to a file
  std::string tmp_file = TempFileName();
  dmlc::Stream * fs = dmlc::Stream::Create(tmp_file.c_str(), "w");
  tree.Save(fs);
  delete fs;

  ASSERT_EQ(GetFileSize(tmp_file), 328)
    << "Expected saved binary file size to be same as object size";

  fs = dmlc::Stream::Create(tmp_file.c_str(), "r");
  xgboost::RegTree tree_read;
  tree_read.Load(fs);
  EXPECT_EQ(tree_read.num_extra_nodes(), tree.num_extra_nodes());
  EXPECT_EQ(tree_read.MaxDepth(), tree.MaxDepth());
  EXPECT_EQ(tree_read.MaxDepth(), 1);

  std::remove(tmp_file.c_str());
}
