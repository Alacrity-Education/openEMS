/*
*	Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GPU_SETUP_MODEL_H
#define GPU_SETUP_MODEL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class Operator;
class CSPrimitives;

/*!
  The material pass of the operator build (Operator::Calc_EC and the first-hit part of Operator::CalcPEC) as plain
  data for a GPU backend, see GPU_Backend::SetupOperator(). The structs below hold no pointer to CSXCAD, no virtual
  and no std::string in the arrays a kernel reads. The model is exact: a backend that runs the algorithm of
  gpu_setup_kernels.h on it produces, for every edge, the same floats, the same first-hit primitive and the same
  used flags as the CPU path, bit for bit.
  */

//! One material or metal property: the values GetEpsilonWeighted(ny, ...) and friends return, with the isotropy rule and the constant weight applied
struct GPU_SetupProperty
{
	double eps[3], kappa[3], mue[3], sigma[3];   //!< for ny = 0..2: weight * value as CSPropMaterial computes the product (the weights are constants by qualification)
	int isMaterial;      //!< a CSPropMaterial (else a hit means the background material)
	int isMetal;         //!< GetType() == METAL exactly (the PEC rule; a conducting sheet is METAL|CONDUCTINGSHEET and is not PEC)
};

//! One primitive of the material list or the edge list
struct GPU_SetupPrimitive
{
	int type;            //!< 0 = CSPrimBox, 1 = CSPrimPolygon, 2 = CSPrimLinPoly
	int property;        //!< index into GPU_SetupModel::properties
	int bboxFilter;      //!< CSPrimitives::IsInsideBox() can decide (valid bounding box, no transformation): else the column filter keeps the primitive
	double bbox[6];      //!< m_BoundBox as IsInsideBox() and CSPrimPolygon::IsInside() read it
	double start[3], stop[3];   //!< box: the corners as CSPrimBox::IsInside() reads them, unsorted
	int normDir;         //!< polygon, linpoly: m_NormDir; the vertices are the (Coord[nP], Coord[nPP]) pairs of nP = (normDir+1)%3, nPP = (normDir+2)%3
	size_t vertexOffset; //!< into GPU_SetupModel::vertices
	size_t vertexCount;  //!< 2*vertexCount doubles (x0 y0 x1 y1 ...), 0 for a polygon whose IsInside returns false
};

//! Everything the pass reads
struct GPU_SetupModel
{
	unsigned int numLines[3];
	std::vector<double> lines[3];   //!< discLines
	double gridDelta;
	double tol;                     //!< CSX->GetDrawingTolerance(), the tolerance of every inside test
	double bg[4];                   //!< the background epsilon, kappa, mue and sigma
	std::vector<GPU_SetupProperty> properties;
	std::vector<GPU_SetupPrimitive> primitives;
	std::vector<double> vertices;
	std::vector<int> matList;       //!< CSX->GetAllPrimitives(true, MATERIAL) in its order, as indices into primitives (never sorted again)
	std::vector<int> edgeList;      //!< CSX->GetAllPrimitives(true, MATERIAL | METAL) in its order, the order of the first-hit table
	bool storage[4];                //!< the arrays m_epsR_ptr, m_kappa_ptr, m_mueR_ptr and m_sigma_ptr are requested
	std::vector<CSPrimitives*> primPtr;   //!< host only: primitive i, to mark it used and to check the lists against the PEC pass

	//! Flatten \a op, false with the reason when the model does not qualify
	bool Build(const Operator* op, std::string& reason);
	//! The qualification of Build(), false with the reason
	static bool Qualify(const Operator* op, std::string& reason);
};

//! Host memory owned by the caller (Operator_GPU), filled by GPU_Backend::SetupOperator()
/*!
  Layouts, with cells = numLines[0]*numLines[1]*numLines[2] and c = x + numLines[0]*(y + numLines[1]*z) (the order of AdrOp::GetPos):
  EC[k] and storage[k] hold n*cells + c, k = 0..3 the capacitance, conductivity, inductance and magnetic loss (EC_C, EC_G, EC_L, EC_R);
  firstHit holds ((n*numLines[0] + x)*numLines[1] + y)*numLines[2] + z, the memory order of ArrayNIJK (N-I-J-K).
  */
struct GPU_SetupResult
{
	float* EC[4];
	float* storage[4];       //!< NULL where the array is not requested (GPU_SetupModel::storage)
	int32_t* firstHit;       //!< per edge: index into GPU_SetupModel::edgeList, -1 none
	uint8_t* used;           //!< per primitive of the model: 1 when a sample or an edge query hit it
	int nanCount;            //!< not 0: an effective material parameter is NaN or Inf at the first edge in the loop order (x, y, z, n) below
	unsigned int nanN, nanK, nanPos[3];
};

#endif // GPU_SETUP_MODEL_H
