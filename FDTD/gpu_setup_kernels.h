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

#ifndef GPU_SETUP_KERNELS_H
#define GPU_SETUP_KERNELS_H

#include <algorithm>
#include <cstdint>

#include "gpu_setup_model.h"
#include "tools/constants.h"

/*!
  The algorithm of the material pass, both for the reference backend (on the CPU) and, as the same text, for the kernels
  of a device backend. Each function mirrors one function of the CPU path line for line (the CPU line is quoted above it).
  Plain inline functions on raw pointers, double only, no std::vector, no virtual, no exception, no transcendental
  function; the only calls into std:: are std::min and std::max on doubles. The host must be built without
  contraction (-ffp-contract=off): no expression below may be fused.
  */

//! The model as raw pointers, what a kernel receives
struct GPU_SetupView
{
	unsigned int numLines[3];
	const double* lines[3];
	double gridDelta;
	double tol;
	double bg[4];
	const GPU_SetupProperty* properties;
	const GPU_SetupPrimitive* primitives;
	const double* vertices;
	const int* matList;
	int matCount;
	const int* edgeList;
	int edgeCount;
};

namespace GPU_Setup
{

/***************************** the mesh *****************************/

// Operator::GetDiscLine (operator.cpp 152)
inline double DiscLine(const GPU_SetupView& v, int n, unsigned int pos, bool dual)
{
	if (pos>=v.numLines[n]) return 0.0;
	if (dual==false)
		return v.lines[n][pos];

	// return dual mesh node
	if (pos<v.numLines[n]-1)
		return 0.5*(v.lines[n][pos] + v.lines[n][pos+1]);

	// dual node for the last line (outside the field domain)
	return v.lines[n][pos] + 0.5*(v.lines[n][pos] - v.lines[n][pos-1]);
}

// Operator::GetDiscDelta (operator.cpp 167)
inline double DiscDelta(const GPU_SetupView& v, int n, unsigned int pos, bool dual)
{
	if (pos>=v.numLines[n]) return 0.0;
	double delta=0;
	if (dual==false)
	{
		if (pos<v.numLines[n]-1)
			delta = DiscLine(v,n,pos+1,false) - DiscLine(v,n,pos,false);
		else
			delta = DiscLine(v,n,pos,false) - DiscLine(v,n,pos-1,false);
		return delta;
	}
	else
	{
		if (pos>0)
			delta = DiscLine(v,n,pos,true) - DiscLine(v,n,pos-1,true);
		else
			delta = DiscLine(v,n,1,false) - DiscLine(v,n,0,false);
		return delta;
	}
}

// Operator::GetRawDiscDelta (operator.cpp 1470)
inline double RawDiscDelta(const GPU_SetupView& v, int ny, int pos)
{
	if (pos<0)
		return (v.lines[ny][0] - v.lines[ny][1]);
	if (pos>=(int)v.numLines[ny]-1)
		return (v.lines[ny][v.numLines[ny]-2] - v.lines[ny][v.numLines[ny]-1]);

	return (v.lines[ny][pos+1] - v.lines[ny][pos]);
}

// Operator::GetEdgeLength (operator.cpp 220): GetDiscDelta(n,pos[n],dualMesh)*gridDelta
inline double EdgeLength(const GPU_SetupView& v, int n, const unsigned int pos[3], bool dual)
{
	return DiscDelta(v,n,pos[n],dual)*v.gridDelta;
}

// Operator::GetNodeWidth (operator.h 176): GetEdgeLength(ny,pos,!dualMesh)
inline double NodeWidth(const GPU_SetupView& v, int ny, const unsigned int pos[3], bool dual)
{
	return EdgeLength(v,ny,pos,!dual);
}

// Operator::GetNodeWidth, the int version (operator.cpp 233)
inline double NodeWidth(const GPU_SetupView& v, int ny, const int pos[3], bool dual)
{
	if ( (pos[0]<0) || (pos[1]<0) || (pos[2]<0) )
		return 0.0;

	unsigned int uiPos[]={(unsigned int)pos[0],(unsigned int)pos[1],(unsigned int)pos[2]};
	return NodeWidth(v, ny, uiPos, dual);
}

// Operator::GetNodeArea (operator.cpp 243), also GetEdgeArea
inline double NodeArea(const GPU_SetupView& v, int ny, const unsigned int pos[3], bool dual)
{
	int nyP = (ny+1)%3;
	int nyPP = (ny+2)%3;
	return NodeWidth(v,nyP,pos,dual) * NodeWidth(v,nyPP,pos,dual);
}

// Operator::GetNodeArea, the int version (operator.cpp 250)
inline double NodeArea(const GPU_SetupView& v, int ny, const int pos[3], bool dual)
{
	if ( (pos[0]<0) || (pos[1]<0) || (pos[2]<0) )
		return 0.0;

	unsigned int uiPos[]={(unsigned int)pos[0],(unsigned int)pos[1],(unsigned int)pos[2]};
	return NodeArea(v, ny, uiPos, dual);
}

// Operator::GetYeeCoords (operator.cpp 190)
inline bool YeeCoords(const GPU_SetupView& v, int ny, const unsigned int pos[3], double* coords, bool dual)
{
	for (int n=0;n<3;++n)
		coords[n]=DiscLine(v,n,pos[n],dual);
	coords[ny]=DiscLine(v,ny,pos[ny],!dual);

	//check if position is inside the FDTD domain
	if (dual==false) //main grid
	{
		if (pos[ny]>=v.numLines[ny]-1)
			return false;
	}
	else	//dual grid
	{
		int nP = (ny+1)%3;
		int nPP = (ny+2)%3;
		if ((pos[nP]>=v.numLines[nP]-1) || (pos[nPP]>=v.numLines[nPP]-1))
			return false;
	}
	return true;
}

// Operator::PrimitivesBoundBox (operator.cpp 2105): the lines pos-1 to pos+1 of x and y, the full range of z (a negative pos)
inline void ColumnBox(const GPU_SetupView& v, int posX, int posY, double boundBox[6])
{
	int BBpos[3] = {posX, posY, -1};
	for (int n=0;n<3;++n)
	{
		if (BBpos[n]<0)
		{
			boundBox[2*n]   = DiscLine(v,n,0,false);
			boundBox[2*n+1] = DiscLine(v,n,v.numLines[n]-1,false);
		}
		else
		{
			boundBox[2*n]   = DiscLine(v,n, std::max(0, BBpos[n]-1),false);
			boundBox[2*n+1] = DiscLine(v,n, std::min(int(v.numLines[n])-1, BBpos[n]+1),false);
		}
	}
}

// Operator::PassQueryBox (operator.cpp 2128): the column box with its upper end at the last line moved to the dual line of
// the last line, the box that holds every point the passes query in the column
inline void QueryBox(const GPU_SetupView& v, int posX, int posY, double queryBox[6])
{
	ColumnBox(v,posX,posY,queryBox);
	int BBpos[3] = {posX, posY, -1};
	for (int n=0;n<3;++n)
		if ((BBpos[n]<0) || (BBpos[n]+1>=int(v.numLines[n])-1))
			queryBox[2*n+1] = DiscLine(v,n,v.numLines[n]-1,true);
}

/***************************** the inside tests *****************************/

// CoordInRange (CSPrimitives.cpp 54), the cartesian branch: dist = tol in all three axes
inline bool CoordInRange(const double* p, const double* start, const double* stop, double tol)
{
	for (int n=0;n<3;++n)
		if ((p[n]<std::min(start[n],stop[n])-tol) || (p[n]>std::max(start[n],stop[n])+tol))
			return false;
	return true;
}

// CSPrimBox::IsInside (CSPrimBox.cpp 114): no transformation, a cartesian coordinate system
inline bool BoxInside(const GPU_SetupPrimitive& prim, const double* Coord, double tol)
{
	double pos[3] = {Coord[0],Coord[1],Coord[2]};
	return CoordInRange(pos, prim.start, prim.stop, tol);
}

// CSPrimPolygon::IsInside (CSPrimPolygon.cpp 157), also CSPrimLinPoly::IsInside (CSPrimLinPoly.cpp 94), whose extrusion is in bbox
inline bool PolygonInside(const GPU_SetupPrimitive& prim, const double* vertices, const double* inCoord, double tol)
{
	// if (vCoords.size()<2) return false;
	if (prim.vertexCount==0) return false;

	double Coord[3] = {inCoord[0],inCoord[1],inCoord[2]};

	for (unsigned int n=0;n<3;++n)
		if ((prim.bbox[2*n]-tol>Coord[n]) || (prim.bbox[2*n+1]+tol<Coord[n])) return false;

	const double* vCoords = vertices + prim.vertexOffset;
	double x=0,y=0;
	int nP = (prim.normDir+1)%3;
	int nPP = (prim.normDir+2)%3;
	x = Coord[nP];
	y = Coord[nPP];

	int wn = 0;

	size_t np = prim.vertexCount;
	double x1 = vCoords[2*np-2];
	double y1 = vCoords[2*np-1];
	double x2 = vCoords[0];
	double y2 = vCoords[1];
	bool startover = y1 >= y ? true : false;
	bool endover;

	for (size_t i=0;i<np;++i)
	{
		x2 = vCoords[2*i];
		y2 = vCoords[2*i+1];

		//check if coord is on a cartesian edge exactly
		if ((x2==x1) && (x1==x) && ( ((y<y1) && (y>y2)) || ((y>y1) && (y<y2)) ))
			return true;
		if ((y2==y1) && (y1==y) && ( ((x<x1) && (x>x2)) || ((x>x1) && (x<x2)) ))
			return true;

		endover = y2 >= y ? true : false;
		if (startover != endover)
		{
			if ((y2 - y)*(x2 - x1) <= (y2 - y1)*(x2 - x))
			{
				if (endover) wn ++;
			}
			else
			{
				if (!endover) wn --;
			}
		}
		startover = endover;
		y1 = y2;
		x1 = x2;
	}
	// return true if polygon is inside the polygon
	if (wn != 0)
		return true;

	// or on an edge or corner, or within the tolerance of it
	x1 = vCoords[2*np-2];
	y1 = vCoords[2*np-1];
	for (size_t i=0;i<np;++i)
	{
		x2 = vCoords[2*i];
		y2 = vCoords[2*i+1];
		double dx = x2-x1, dy = y2-y1;
		double len2 = dx*dx+dy*dy;
		double t = len2>0 ? ((x-x1)*dx+(y-y1)*dy)/len2 : 0;
		t = std::max(0.0, std::min(1.0, t));
		double ex = x1+t*dx-x, ey = y1+t*dy-y;
		if (ex*ex+ey*ey<=tol*tol)
			return true;
		x1 = x2;
		y1 = y2;
	}
	return false;
}

// the IsInside() of the primitive's class, called by ContinuousStructure::GetPropertyByCoordPriority as IsInside(coord, dDrawingTol)
inline bool Inside(const GPU_SetupView& v, int prim, const double* coord)
{
	const GPU_SetupPrimitive& p = v.primitives[prim];
	if (p.type==0)
		return BoxInside(p, coord, v.tol);
	return PolygonInside(p, v.vertices, coord, v.tol);
}

// CSPrimitives::IsInsideBox (CSPrimitives.cpp 182): -1 not inside, +1 inside, 0 unknown (bboxFilter 0 stands for the three conditions that return 0)
inline int InsideBox(const GPU_SetupPrimitive& p, const double* boundbox)
{
	if (p.bboxFilter==0)
		return 0;

	for (int i=0;i<3;++i)
	{
		int d_l = 2*i;
		int d_u = d_l+1;
		if ((boundbox[d_l]<p.bbox[d_l]) && (boundbox[d_l]<p.bbox[d_u]) && (boundbox[d_u]<p.bbox[d_l]) && (boundbox[d_u]<p.bbox[d_u]))
			return -1;
		if ((boundbox[d_l]>p.bbox[d_l]) && (boundbox[d_l]>p.bbox[d_u]) && (boundbox[d_u]>p.bbox[d_l]) && (boundbox[d_u]>p.bbox[d_u]))
			return -1;
	}
	return 1;
}

// Operator::PassPrimOutside (operator.cpp 2139), negated, with the reach of Operator::BeginPassPrimitives (operator.cpp 2190):
// false when no point of queryBox can be inside the primitive by the first range test of its own inside test
// (box: CoordInRange's min/max -+ tol; polygon and extruded polygon: the bounding box -+ tol of PolygonInside)
inline bool MayContain(const GPU_SetupPrimitive& p, const double* queryBox, double tol)
{
	for (int n=0;n<3;++n)
	{
		double reachLo, reachHi;
		if (p.type==0)
		{
			reachLo = std::min(p.start[n],p.stop[n])-tol;
			reachHi = std::max(p.start[n],p.stop[n])+tol;
		}
		else
		{
			reachLo = p.bbox[2*n]-tol;
			reachHi = p.bbox[2*n+1]+tol;
		}
		if ((queryBox[2*n+1]<reachLo) || (queryBox[2*n]>reachHi))
			return false;
	}
	return true;
}

// Operator::FirstPrimitiveIndex (operator.cpp 1503), the loop of ContinuousStructure::GetPropertyByCoordPriority:
// the first entry of the list whose inside test is true, its position in the list, -1 for none
inline int FirstHit(const GPU_SetupView& v, const double* coord, const int* list, int listLen)
{
	for (int i=0;i<listLen;++i)
		if (Inside(v, list[i], coord))
			return i;
	return -1;
}

/***************************** the sample *****************************/

//! The two values of Operator::GetMaterialPair (operator.cpp 1539) at \a coord: type 0 epsilon and kappa, type 2 mue and sigma; marks the hit primitive used
inline void MaterialPair(const GPU_SetupView& v, int ny, const double* coord, int MatType, const int* list, int listLen, double* value, uint8_t* used)
{
	int hit = FirstHit(v, coord, list, listLen);
	const GPU_SetupProperty* mat = NULL;
	if (hit>=0)
	{
		used[list[hit]] = 1; // markAsUsed
		const GPU_SetupProperty& prop = v.properties[v.primitives[list[hit]].property];
		if (prop.isMaterial)
			mat = &prop; // FindMaterial: NULL for a property that is no CSPropMaterial
	}
	if (mat)
	{
		// MaterialValue (operator.cpp 1553): the weighted values of the property
		value[0] = (MatType==0) ? mat->eps[ny] : mat->mue[ny];
		value[1] = (MatType==0) ? mat->kappa[ny] : mat->sigma[ny];
	}
	else
	{
		value[0] = (MatType==0) ? v.bg[0] : v.bg[2];
		value[1] = (MatType==0) ? v.bg[1] : v.bg[3];
	}
}

// Operator::AverageMatQuarterCell (operator.cpp 1711); returns false instead of throwing when an EffMat is NaN or Inf (index in \a badK)
inline bool QuarterCell(const GPU_SetupView& v, int ny, const unsigned int* pos, double* EffMat, const int* list, int listLen, uint8_t* used, int* badK)
{
	int n=ny;
	double coord[3];
	double shiftCoord[3];
	int nP = (n+1)%3;
	int nPP = (n+2)%3;
	coord[0] = v.lines[0][pos[0]];
	coord[1] = v.lines[1][pos[1]];
	coord[2] = v.lines[2][pos[2]];
	double delta=RawDiscDelta(v,n,pos[n]);
	double deltaP=RawDiscDelta(v,nP,pos[nP]);
	double deltaPP=RawDiscDelta(v,nPP,pos[nPP]);
	double delta_M=RawDiscDelta(v,n,pos[n]-1);
	double deltaP_M=RawDiscDelta(v,nP,pos[nP]-1);
	double deltaPP_M=RawDiscDelta(v,nPP,pos[nPP]-1);

	int loc_pos[3] = {(int)pos[0],(int)pos[1],(int)pos[2]};
	double A_n;
	double area = 0;
	double mat[2];

	//******************************* epsilon,kappa averaging *****************************//
	//shift up-right
	shiftCoord[n] = coord[n]+delta*0.5;
	shiftCoord[nP] = coord[nP]+deltaP*0.25;
	shiftCoord[nPP] = coord[nPP]+deltaPP*0.25;
	A_n = NodeArea(v,ny,loc_pos,true);
	MaterialPair(v, n, shiftCoord, 0, list, listLen, mat, used);
	EffMat[0] = mat[0]*A_n;
	EffMat[1] = mat[1]*A_n;
	area+=A_n;

	//shift up-left
	shiftCoord[n] = coord[n]+delta*0.5;
	shiftCoord[nP] = coord[nP]-deltaP_M*0.25;
	shiftCoord[nPP] = coord[nPP]+deltaPP*0.25;

	--loc_pos[nP];
	A_n = NodeArea(v,ny,loc_pos,true);
	MaterialPair(v, n, shiftCoord, 0, list, listLen, mat, used);
	EffMat[0] += mat[0]*A_n;
	EffMat[1] += mat[1]*A_n;
	area+=A_n;

	//shift down-right
	shiftCoord[n] = coord[n]+delta*0.5;
	shiftCoord[nP] = coord[nP]+deltaP*0.25;
	shiftCoord[nPP] = coord[nPP]-deltaPP_M*0.25;
	++loc_pos[nP];
	--loc_pos[nPP];
	A_n = NodeArea(v,ny,loc_pos,true);
	MaterialPair(v, n, shiftCoord, 0, list, listLen, mat, used);
	EffMat[0] += mat[0]*A_n;
	EffMat[1] += mat[1]*A_n;
	area+=A_n;

	//shift down-left
	shiftCoord[n] = coord[n]+delta*0.5;
	shiftCoord[nP] = coord[nP]-deltaP_M*0.25;
	shiftCoord[nPP] = coord[nPP]-deltaPP_M*0.25;
	--loc_pos[nP];
	A_n = NodeArea(v,ny,loc_pos,true);
	MaterialPair(v, n, shiftCoord, 0, list, listLen, mat, used);
	EffMat[0] += mat[0]*A_n;
	EffMat[1] += mat[1]*A_n;
	area+=A_n;

	EffMat[0]*=EPS0/area;
	EffMat[1]/=area;

	//******************************* mu,sigma averaging *****************************//
	loc_pos[0]=pos[0];
	loc_pos[1]=pos[1];
	loc_pos[2]=pos[2];
	double length=0;

	//shift down
	shiftCoord[n] = coord[n]-delta_M*0.25;
	shiftCoord[nP] = coord[nP]+deltaP*0.5;
	shiftCoord[nPP] = coord[nPP]+deltaPP*0.5;
	--loc_pos[n];
	double delta_ny = NodeWidth(v,n,loc_pos,true);
	MaterialPair(v, n, shiftCoord, 2, list, listLen, mat, used);
	EffMat[2] = delta_ny / mat[0];
	double sigma = mat[1];
	if (sigma)
		EffMat[3] = delta_ny / sigma;
	else
		EffMat[3] = 0;
	length=delta_ny;

	//shift up
	shiftCoord[n] = coord[n]+delta*0.25;
	shiftCoord[nP] = coord[nP]+deltaP*0.5;
	shiftCoord[nPP] = coord[nPP]+deltaPP*0.5;
	++loc_pos[n];
	delta_ny = NodeWidth(v,n,loc_pos,true);
	MaterialPair(v, n, shiftCoord, 2, list, listLen, mat, used);
	EffMat[2] += delta_ny / mat[0];
	sigma = mat[1];
	if (sigma)
		EffMat[3] += delta_ny / sigma;
	else
		EffMat[3] = 0;
	length+=delta_ny;

	EffMat[2] = length * MUE0 / EffMat[2];
	if (EffMat[3]) EffMat[3]=length / EffMat[3];

	// std::isnan || std::isinf of the CPU
	for (int k=0; k<4; ++k)
		if ((EffMat[k]!=EffMat[k]) || (EffMat[k]>1.7976931348623157e308) || (EffMat[k]<-1.7976931348623157e308))
		{
			*badK = k;
			return false;
		}

	return true;
}

/***************************** the edge values *****************************/

//! The stores of Operator::Calc_ECPos (operator.cpp 1408) after Calc_EffMatPos: EC[4] as doubles from EffMat, the zero rules included
inline void EdgeValues(const GPU_SetupView& v, int ny, const unsigned int* pos, const double* EffMat, double* EC)
{
	double delta = EdgeLength(v,ny,pos,false);
	double area  = NodeArea(v,ny,pos,false);

	if (delta)
	{
		EC[0] = EffMat[0] * area/delta;
		EC[1] = EffMat[1] * area/delta;
	}
	else
	{
		EC[0] = 0;
		EC[1] = 0;
	}

	delta = EdgeLength(v,ny,pos,true);
	area  = NodeArea(v,ny,pos,true);

	if (delta)
	{
		EC[2] = EffMat[2] * area/delta;
		EC[3] = EffMat[3] * area/delta;
	}
	else
	{
		EC[2] = 0;
		EC[3] = 0;
	}
}

} // namespace GPU_Setup

#endif // GPU_SETUP_KERNELS_H
