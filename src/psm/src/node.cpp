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

#include "node.h"

#include <iostream>
#include <vector>

namespace psm {

using std::make_pair;
using std::max;
using std::pair;
using std::vector;

//! Get the layer number of the node
/*
 * \return Layer number of the node
 * */
int Node::getLayerNum()
{
  return layer_;
}

//! Set the layer number of the node
/*
 \param t_layer Layer number
*/
void Node::setLayerNum(int t_layer)
{
  layer_ = t_layer;
}

//! Get the location of the node
/*
 * \return NodeLoc which is x and y index
 * */
Point Node::getLoc()
{
  return loc_;
}

//! Set the location of the node using x and y coordinates
/*
 * \param t_x x index
 * \param t_y y index
 * */
void Node::setLoc(int t_x, int t_y)
{
  loc_ = Point(t_x, t_y);
}

//! Set the location of the node using x,y and layer information
/*
 * \param t_x x index
 * \param t_y y index
 * \param t_l Layer number
 * */
void Node::setLoc(int t_x, int t_y, int t_l)
{
  setLayerNum(t_l);
  setLoc(t_x, t_y);
}

//! Get location of the node in G matrix
/*
 * \return Location in G matrix
 */
NodeIdx Node::getGLoc()
{
  return node_loc_;
}

//! Set location of the node in G matrix
/*
 * \param t_loc Location in the G matrix
 */
void Node::setGLoc(NodeIdx t_loc)
{
  node_loc_ = t_loc;
}

//! Function to print node details
void Node::print(utl::Logger* logger)
{
  logger->report("Node: {}", node_loc_);
  logger->report(
      "  Location: Layer {}, x {}, y {}", layer_, loc_.getX(), loc_.getY());
  logger->report("  Bounding box: x {}, y {} ", bBox_.first, bBox_.second);
  logger->report("  Current: {:5.4e}A", current_src_);
  logger->report("  Voltage: {:5.4e}V", voltage_);
  logger->report("  Has connection: {}", connected_ ? "true" : "false");
  logger->report("  Has instances:  {}", has_instances_ ? "true" : "false");
}

//! Function to set the bounding box of the stripe
void Node::setBbox(int t_dX, int t_dY)
{
  bBox_ = make_pair(t_dX, t_dY);
}

//! Function to get the bounding box of the stripe
BBox Node::getBbox()
{
  return bBox_;
}

//! Function to update the stripe
/*
 * \param t_dX Change in the x value
 * \param t_dY Change in the y value
 * \return nothing
 */
void Node::updateMaxBbox(int t_dX, int t_dY)
{
  BBox nodeBbox = bBox_;
  int dx = max(nodeBbox.first, t_dX);
  int dy = max(nodeBbox.second, t_dY);
  setBbox(dx, dy);
}

//! Function to set the current value at a particular node
/*
 * \param t_current Current magnitude
 */
void Node::setCurrent(double t_current)
{
  current_src_ = t_current;
}

//! Function to get the value of current at a node
double Node::getCurrent()
{
  return current_src_;
}

//! Function to add the current source
/*
 * \param t_current Value of current source
 */
void Node::addCurrentSrc(double t_current)
{
  double node_cur = getCurrent();
  setCurrent(node_cur + t_current);
}

//! Function to set the value of the voltage source
/*
 * \param t_voltage Voltage source magnitude
 * */
void Node::setVoltage(double t_voltage)
{
  voltage_ = t_voltage;
}

//! Function to get the value of the voltage source
/*
 * \return Voltage value at the node
 */
double Node::getVoltage()
{
  return voltage_;
}

bool Node::getConnected()
{
  return connected_;
}

void Node::setConnected()
{
  connected_ = true;
}

bool Node::hasInstances()
{
  return has_instances_;
}

vector<dbInst*> Node::getInstances()
{
  return connected_instances_;
}

void Node::addInstance(dbInst* inst)
{
  has_instances_ = true;
  connected_instances_.push_back(inst);
}
}  // namespace psm
