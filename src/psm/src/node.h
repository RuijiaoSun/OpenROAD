/*
BSD 3-Clause License

Copyright (c) 2020, The Regents of the University of Minnesota

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __IRSOLVER_NODE__
#define __IRSOLVER_NODE__

#include <map>

#include "odb/db.h"
#include "utl/Logger.h"
namespace psm {
using odb::dbInst;

using odb::Point;
using BBox = std::pair<int, int>;
using NodeIdx = int;  // TODO temp as it interfaces with SUPERLU
using GMatLoc = std::pair<NodeIdx, NodeIdx>;

//! Data structure for the Dictionary of Keys Matrix
struct DokMatrix
{
  NodeIdx num_rows;
  NodeIdx num_cols;
  std::map<GMatLoc, double> values;  // pair < col_num, row_num >
};

//! Data structure for the Compressed Sparse Column Matrix
struct CscMatrix
{
  NodeIdx num_rows;
  NodeIdx num_cols;
  NodeIdx nnz;
  std::vector<NodeIdx> row_idx;
  std::vector<NodeIdx> col_ptr;
  std::vector<double> values;
};

//! Node class which stores the properties of the node of the PDN
class Node
{
 public:
  Node() : bBox_(std::make_pair(0.0, 0.0)) {}
  //! Get the layer number of the node
  int getLayerNum();
  //! Set the layer number of the node
  void setLayerNum(int layer);
  //! Get the location of the node
  Point getLoc();
  //! Set the location of the node using x and y coordinates
  void setLoc(int x, int y);
  //! Set the location of the node using x,y and layer information
  void setLoc(int x, int y, int l);
  //! Get location of the node in G matrix
  NodeIdx getGLoc();
  //! Get location of the node in G matrix
  void setGLoc(NodeIdx loc);
  //! Function to print node details
  void print(utl::Logger* logger);
  //! Function to set the bounding box of the stripe
  void setBbox(int dX, int dY);
  //! Function to get the bounding box of the stripe
  BBox getBbox();
  //! Function to update the stripe
  void updateMaxBbox(int dX, int dY);
  //! Function to set the current value at a particular node
  void setCurrent(double t_current);
  //! Function to get the value of current at a node
  double getCurrent();
  //! Function to add the current source
  void addCurrentSrc(double t_current);
  //! Function to set the value of the voltage source
  void setVoltage(double t_voltage);
  //! Function to get the value of the voltage source
  double getVoltage();

  bool getConnected();

  void setConnected();

  bool hasInstances();

  std::vector<dbInst*> getInstances();

  void addInstance(dbInst* inst);

 private:
  int layer_;
  Point loc_;
  NodeIdx node_loc_{0};
  BBox bBox_;
  double current_src_{0.0};
  double voltage_{0.0};
  bool connected_{false};
  bool has_instances_{false};
  std::vector<dbInst*> connected_instances_;
};
}  // namespace psm
#endif
