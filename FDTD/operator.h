/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef OPERATOR_H
#define OPERATOR_H

#include "tools/AdrOp.h"
#include "tools/constants.h"
#include "excitation.h"
#include "Common/operator_base.h"

#include "tools/arraylib/array_nijk.h"

class Operator_Extension;
class Operator_Ext_Excitation;
class Engine;
class TiXmlElement;
class CSPropMaterial;

//! Basic FDTD-operator
class Operator : public Operator_Base
{
	friend class Engine;
	friend class Engine_Interface_FDTD;
	friend class Operator_Ext_LorentzMaterial;
	friend class Operator_Ext_ConductingSheet;
	friend class Operator_Ext_PML_SF_Plane;
	friend class Operator_Ext_Excitation;
	friend class Operator_Ext_UPML;
	friend class Operator_Ext_Cylinder;
	friend class Operator_Ext_LumpedRLC;
	friend class Operator_Ext_Absorbing_BC;
	friend class Operator_Cache;
	friend struct GPU_SetupModel;

public:
	enum DebugFlags {None=0,debugMaterial=1,debugOperator=2,debugPEC=4};

	enum MatAverageMethods {QuarterCell=0, CentralCell=1};

	//! Create a new operator
	static Operator* New();
	virtual ~Operator();

	virtual Engine* CreateEngine();
	virtual Engine* GetEngine() const {return m_Engine;}

	virtual bool SetGeometryCSX(ContinuousStructure* geo);

	virtual int CalcECOperator( DebugFlags debugFlags = None );

	// the next four functions need to be reimplemented in a derived class
	inline virtual FDTD_FLOAT GetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& vv = *vv_ptr;
		return vv(n, x, y, z);
	}

	inline virtual FDTD_FLOAT GetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& vi = *vi_ptr;
		return vi(n, x, y, z);
	}

	inline virtual FDTD_FLOAT GetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& ii = *ii_ptr;
		return ii(n, x, y, z);
	}
	inline virtual FDTD_FLOAT GetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& iv = *iv_ptr;
		return iv(n, x, y, z);
	}

	// convenient access functions
	inline virtual FDTD_FLOAT GetVV( unsigned int n, unsigned int pos[3] ) const
	{
		return GetVV(n, pos[0], pos[1], pos[2]);
	}

	inline virtual FDTD_FLOAT GetVI( unsigned int n, unsigned int pos[3] ) const
	{
		return GetVI(n, pos[0], pos[1], pos[2]);
	}
	inline virtual FDTD_FLOAT GetII( unsigned int n, unsigned int pos[3] ) const
	{
		return GetII(n, pos[0], pos[1], pos[2]);
	}
	inline virtual FDTD_FLOAT GetIV( unsigned int n, unsigned int pos[3] ) const
	{
		return GetIV(n, pos[0], pos[1], pos[2]);
	}

	// the next four functions need to be reimplemented in a derived class
	inline virtual void SetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& vv = *vv_ptr;
		vv(n, x, y, z) = value;
	}
	inline virtual void SetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& vi = *vi_ptr;
		vi(n, x, y, z) = value;
	}
	inline virtual void SetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& ii = *ii_ptr;
		ii(n, x, y, z) = value;
	}
	inline virtual void SetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& iv = *iv_ptr;
		iv(n, x, y, z) = value;
	}

	virtual void ApplyElectricBC(bool* dirs); //applied by default to all boundaries
	virtual void ApplyMagneticBC(bool* dirs);

	virtual void SetBCSize(int dir, int size) {m_BC_Size[dir]=size;}
	virtual int GetBCSize(int dir) {return m_BC_Size[dir];}

	//! Set a forced timestep to use by the operator
	virtual void SetTimestep(double ts) {dT = ts;}
	virtual void SetTimestepFactor(double factor);
	bool GetTimestepValid() const {return !m_InvaildTimestep;}

	//! Choose a time step method (0=auto, 1=CFL, 3=Rennings)
	void SetTimeStepMethod(int var) {m_TimeStepVar=var;}

	//! Set the material averaging method /sa MatAverageMethods
	void SetMaterialAvgMethod(MatAverageMethods method);

	//! Set material averaging method to the advanced quarter cell material interpolation (default)
	void SetQuarterCellMaterialAvg() {m_MatAverageMethod=QuarterCell;}

	//! Set operator to assume a constant material inside a cell (material probing in the cell center)
	void SetCellConstantMaterial() {m_MatAverageMethod=CentralCell;}

	virtual double GetNumberCells() const;

	virtual unsigned int GetNumberOfNyquistTimesteps() const {return m_Exc->GetNyquistNum();}

	virtual unsigned int GetNumberOfLines(int ny, bool full=false) const {UNUSED(full);return numLines[ny];}

	virtual void ShowStat() const;
	virtual void ShowExtStat() const;

	virtual double GetGridDelta() const {return gridDelta;}

	//! Get the disc line in \a n direction (in drawing units)
	virtual double GetDiscLine(int n, unsigned int pos, bool dualMesh=false) const;

	//! Get the disc line delta in \a n direction (in drawing units)
	virtual double GetDiscDelta(int n, unsigned int pos, bool dualMesh=false) const;

	//! Get the coordinates for a given node index and component, according to the yee-algorithm. Returns true if inside the FDTD domain.
	virtual bool GetYeeCoords(int ny, unsigned int pos[3], double* coords, bool dualMesh) const;

	virtual bool GetNodeCoords(const unsigned int pos[3], double* coords, bool dualMesh=false, CoordinateSystem c_system=UNDEFINED_CS) const;

	//! Get the node width for a given direction \a n and a given mesh position \a pos
	virtual double GetNodeWidth(int ny, const unsigned int pos[3], bool dualMesh = false) const {return GetEdgeLength(ny,pos,!dualMesh);}
	//! Get the node width for a given direction \a n and a given mesh position \a pos
	virtual double GetNodeWidth(int ny, const int pos[3], bool dualMesh = false) const;

	//! Get the node area for a given direction \a n and a given mesh position \a pos
	virtual double GetNodeArea(int ny, const unsigned int pos[3], bool dualMesh = false) const;
	//! Get the node area for a given direction \a n and a given mesh position \a pos
	virtual double GetNodeArea(int ny, const int pos[3], bool dualMesh = false) const;

	//! Get the length of an FDTD edge (unit is meter).
	virtual double GetEdgeLength(int ny, const unsigned int pos[3], bool dualMesh = false) const;

	//! The smallest mesh step (in drawing units)
	virtual double GetMinMeshStep() const;

	//! Get the volume of an FDTD cell
	virtual double GetCellVolume(const unsigned int pos[3], bool dualMesh = false) const;

	//! Get the area around an edge for a given direction \a n and a given mesh position \a pos
	/*!
		This will return the area around an edge with a given direction, measured at the middle of the edge.
		In a cartesian mesh this is equal to the NodeArea, may be different in other coordinate systems.
	*/
	virtual double GetEdgeArea(int ny, const unsigned int pos[3], bool dualMesh = false) const {return GetNodeArea(ny,pos,dualMesh);}

	virtual unsigned int SnapToMeshLine(int ny, double coord, bool &inside, bool dualMesh=false, bool fullMesh=false) const;

	//! Snap the given coordinates to mesh indices
	virtual bool SnapToMesh(const double* coord, unsigned int* uicoord, bool dualMesh=false, bool fullMesh=false, bool* inside=NULL) const;

	//! Snap a given box to the FDTD mesh
	virtual int SnapBox2Mesh(const double* start, const double* stop, unsigned int* uiStart, unsigned int* uiStop, bool dualMesh=false, bool fullMesh=false, int SnapMethod=0, bool* bStartIn=NULL, bool* bStopIn=NULL) const;

	//! Snap a given line to the operator mesh
	/*!
	  \param[in] start coorindate of the line
	  \param[in] stop coorindate of the line
	  \param[out] uiStart the snapped line-start coorindate index
	  \param[out] uiStop the snapped line-stop coorindate index
	  \param[in] dualMesh snap to main or dual mesh (default is main mesh)
	  \return returns a status, 0 = success, 1 = start outside, 2 = stop outside, 3 = both outside
	  */
	virtual int SnapLine2Mesh(const double* start, const double* stop, unsigned int* uiStart, unsigned int* uiStop, bool dualMesh=false, bool fullMesh=false) const;

	virtual void AddExtension(Operator_Extension* op_ext);
	virtual void DeleteExtension(Operator_Extension* op_ext);
	virtual size_t GetNumberOfExtentions() const {return m_Op_exts.size();}
	virtual Operator_Extension* GetExtension(size_t index) const {return m_Op_exts.at(index);}

	virtual void CleanupMaterialStorage();

	virtual double GetDiscMaterial(int type, int ny, const unsigned int pos[3]) const;

	//! Get the cell center coordinate usable for material averaging (Warning, may not be the yee cell center)
	virtual bool GetCellCenterMaterialAvgCoord(const int pos[3], double coord[3]) const;

	virtual void SetExcitationSignal(Excitation* exc);
	virtual Excitation* GetExcitationSignal() const {return m_Exc;}

	Operator_Ext_Excitation* GetExcitationExtension() const;

	virtual double CalcNumericPhaseVelocity(unsigned int start[3], unsigned int stop[3], double propDir[3], float freq) const;

	virtual std::vector<CSPrimitives*> GetPrimitivesBoundBox(
		int posX,
		int posY,
		int posZ,
		CSProperties::PropertyType type=CSProperties::ANY
	) const;

	/*!
	  The per-edge first-hit table of the PEC pass, for the extension builds that run the same query: for every edge
	  (n, x, y, z) the first primitive of the MATERIAL | METAL column list GetPrimitivesBoundBox(x, y, -1, ...) that
	  contains the primal edge midpoint GetYeeCoords(n, pos, coord, false) within CSX->GetDrawingTolerance(), i.e. the
	  primitive CSX->GetPropertyByCoordPriority(coord, vPrims, ...) finds there.

	  The contract (the device setup pass produces the same table): m_FirstHit is an ArrayNIJK<int32_t> of 3 x the
	  cells, N-I-J-K ordered; an entry is -1 for no primitive, else the index of the primitive into m_FirstHitPrims, the
	  sorted full list CSX->GetAllPrimitives(true, MATERIAL | METAL) of the pass (the column lists are subsequences of
	  it in its order, so the first hit of a column list is that primitive of the full list). An x line is valid only
	  when m_FirstHitLine[x] is set (the PEC pass of an operator may leave x lines out, e.g. the inner lines of the
	  cylindrical multigrid operator): for the other lines the consumers query as before. The table exists from
	  CalcPEC() to the end of the extension builds of CalcECOperator() (not on an operator cache hit, which builds no
	  table and runs no consumer), then it is freed.
	  */
	bool HasFirstHit(unsigned int x) const {return (m_FirstHit!=NULL) && (x<m_FirstHitLine.size()) && (m_FirstHitLine[x]!=0);}
	//! The first-hit primitive of edge \a n at \a pos, NULL for none; only valid for an x line with HasFirstHit(pos[0])
	CSPrimitives* GetFirstHit(int n, const unsigned int pos[3]) const
	{
		int32_t idx = (*m_FirstHit)(n, pos[0], pos[1], pos[2]);
		return (idx<0) ? NULL : m_FirstHitPrims[idx];
	}

protected:
	//! use New() for creating a new Operator
	Operator();

	virtual void Init();
	void Delete();
	virtual void Reset();
	virtual void InitOperator();
	virtual void InitDataStorage();

	virtual bool SetupCSXGrid(CSRectGrid* grid);

	virtual Grid_Path FindPath(double start[], double stop[]);

	// debug
	virtual void DumpOperator2File(std::string filename);
	virtual void DumpMaterial2File(std::string filename);
	virtual void DumpPEC2File( std::string filename, unsigned int *range = NULL );

	unsigned int m_Nr_PEC[3]; //count PEC edges
	virtual bool CalcPEC();
	virtual void CalcPEC_Range(unsigned int startX, unsigned int stopX, unsigned int* counter);	//internal to CalcPEC
	virtual void CalcPEC_Curves();	//internal to CalcPEC

	//Calc timestep only internal use
	int m_TimeStepVar;
	double m_TimeStepFactor;
	virtual double CalcTimestep();
	double opt_dT;
	bool m_InvaildTimestep;
	std::string m_Used_TS_Name;

	double CalcTimestep_Var1();
	double CalcTimestep_Var3();

	//! The smallest timestep of a Var3 search and its first position in the serial loop order (n, z, y, x)
	struct Timestep_Min
	{
		double dT;
		unsigned int n;
		unsigned int pos[3];
	};
	//! The Var3 search over the x lines \a xStart to \a xStop (inclusive) in the serial loop order, with the address operator \a adrOp
	void CalcTimestep_Var3_Range(unsigned int xStart, unsigned int xStop, AdrOp* adrOp, Timestep_Min& tmin) const;
	//! The Var3 search over the whole mesh: serial on MainOp here, over x-slabs in the threaded operators
	virtual void CalcTimestep_Var3_Search(Timestep_Min& tmin);
	//! The Var3 search over the x-slabs \a start to \a stop, one thread and one copy of MainOp per slab, merged to the serial result
	void CalcTimestep_Var3_Slabs(const std::vector<unsigned int>& start, const std::vector<unsigned int>& stop, Timestep_Min& tmin);

	//! Calculate the FDTD equivalent circuit parameter for the given position and direction ny. \sa Calc_EffMat_Pos
	virtual bool Calc_ECPos(
		int ny,
		const unsigned int* pos,
		double* EC, const std::vector<CSPrimitives *>& vPrims
	) const;

	//! Get the FDTD raw disc delta, needed by Calc_EffMatPos() \sa Calc_EffMatPos
	/*!
	  Get the raw disc delta for a given position and direction.
	  The result will be positive if a disc delta inside the simulation domain is requested.
	  The result will be the negative value of the first or last disc delta respectively if the position is outside the field domain.
	  */
	virtual double GetRawDiscDelta(int ny, const int pos) const;

	//! Get the material at a given coordinate, direction and type from CSX (internal use only)
	virtual double GetMaterial(
		int ny,
		const double coords[3],
		int MatType,
		const std::vector<CSPrimitives*>& vPrims,
		bool markAsUsed=true
	) const;

	//! The two values of one sample point of the averaging from one search: the types \a MatType and \a MatType+1 (0: permittivity and conductivity, 2: permeability and magnetic loss), as two GetMaterial() calls return them
	virtual void GetMaterialPair(
		int ny,
		const double coords[3],
		int MatType,
		const std::vector<CSPrimitives*>& vPrims,
		double* value,
		bool markAsUsed=true
	) const;

	//! The first primitive of \a vPrims (priority sorted) that contains \a coords within the drawing tolerance, NULL for none: the loop of ContinuousStructure::GetPropertyByCoordPriority
	CSPrimitives* FirstPrimitive(const double* coords, const std::vector<CSPrimitives*>& vPrims, bool markAsUsed) const;
	//! The index into \a vPrims of the primitive FirstPrimitive() returns, -1 for none
	int FirstPrimitiveIndex(const double* coords, const std::vector<CSPrimitives*>& vPrims, bool markAsUsed) const;

	//! The material of the first primitive at \a coords, NULL for none or a non-material property (the background)
	CSPropMaterial* FindMaterial(const double* coords, const std::vector<CSPrimitives*>& vPrims, bool markAsUsed) const;

	//! The value of type \a MatType of the material \a mat at \a coords, the background value for NULL
	double MaterialValue(CSPropMaterial* mat, int ny, const double* coords, int MatType) const;

	MatAverageMethods m_MatAverageMethod;

	//! Calculate the effective/averaged material properties at the given position and direction ny.
	virtual bool Calc_EffMatPos(
		int ny,
		const unsigned int* pos,
		double* EffMat,
		const std::vector<CSPrimitives*>& vPrims
	) const;

	virtual bool AverageMatCellCenter(
		int ny,
		const unsigned int* pos,
		double* EffMat,
		const std::vector<CSPrimitives*>& vPrims
	) const;

	virtual bool AverageMatQuarterCell(
		int ny,
		const unsigned int* pos,
		double* EffMat,
		const std::vector<CSPrimitives*>& vPrims
	) const;

	//! Calc operator at certain \a pos
	virtual void Calc_ECOperatorPos(int n, unsigned int* pos);
	//! Calc operator at \a pos with the array index \a i of \a pos, without touching MainOp
	void Calc_ECOperatorIndex(int n, const unsigned int* pos, unsigned int i);
	//! The coefficient loop of CalcECOperator over all nodes: serial here, over x-slabs in the threaded operators
	virtual void Calc_ECOperator_All();
	//! The coefficient loop over the x-slabs \a start to \a stop (inclusive), one thread per slab, the index from MainOp->GetPos at (0,0,0)
	void Calc_ECOperator_Slabs(const std::vector<unsigned int>& start, const std::vector<unsigned int>& stop);

	//! Calculate and setup lumped elements
	virtual bool Calc_LumpedElements();

	//! Condition to determine if this is a lumped RC, to invoke Calc_LumpedElements
	virtual bool IsLEparRC(const CSPropLumpedElement* const p_prop);

	//! Store the size of the applied boundary conditions
	int m_BC_Size[6];

	//store material properties for post-processing
	ArrayLib::ArrayNIJK<float>* m_epsR_ptr;
	ArrayLib::ArrayNIJK<float>* m_kappa_ptr;
	ArrayLib::ArrayNIJK<float>* m_mueR_ptr;
	ArrayLib::ArrayNIJK<float>* m_sigma_ptr;

	//EC elements, internal only!
	virtual void Init_EC();
	virtual bool Calc_EC();
	//! Number of threads the (threaded) Calc_EC and CalcPEC use, reported in the OPENEMS_SETUP_TIMES lines
	virtual unsigned int SetupThreads() const {return 1;}
	virtual void Calc_EC_Range(unsigned int xStart, unsigned int xStop);
	//! On an operator cache hit Calc_EC and CalcPEC are not called: release any build thread that waits for them
	virtual void SkipBuildThreads() {}

	//! Build the priority-sorted primitive list of \a type once for a pass (Calc_EC, CalcPEC): GetPrimitivesBoundBox() filters it per column until EndPassPrimitives()
	void BeginPassPrimitives(CSProperties::PropertyType type);
	void EndPassPrimitives();
	bool m_PassPrims_On;
	CSProperties::PropertyType m_PassPrims_Type;
	std::vector<CSPrimitives*> m_PassPrims; //!< CSX->GetAllPrimitives(true, m_PassPrims_Type), read-only during the pass
	//! Per entry of m_PassPrims: 1 when the prefilter of the pass applies (a box, polygon or extruded polygon without a transformation, cartesian)
	std::vector<char> m_PassPrimsCut;
	//! Per entry of m_PassPrims, six doubles: lo and hi of each axis such that the primitive's IsInside(p, tol) is false when p[n]<lo or p[n]>hi
	std::vector<double> m_PassPrimsReach;

	//! The box GetPrimitivesBoundBox() filters with: the lines pos-1 to pos+1 of each axis, the full range for a negative pos
	void PrimitivesBoundBox(int posX, int posY, int posZ, double boundBox[6]) const;
	//! The box that holds every point the passes query in the column of PrimitivesBoundBox(): its upper end at the last line is the dual line of the last line
	void PassQueryBox(int posX, int posY, int posZ, double queryBox[6]) const;
	//! True when no point of \a queryBox is inside entry \a j of m_PassPrims (the prefilter of the pass lists)
	bool PassPrimOutside(size_t j, const double queryBox[6]) const;
	//! Allocate the first-hit table for the PEC pass (after BeginPassPrimitives(MATERIAL | METAL)), see HasFirstHit()
	void InitFirstHit();
	void FreeFirstHit();
	//! The column list of the pass list as GetPrimitivesBoundBox(), and the index of each entry into m_FirstHitPrims
	std::vector<CSPrimitives*> GetFirstHitBoundBox(int posX, int posY, int posZ, std::vector<int32_t>& index) const;
	ArrayLib::ArrayNIJK<int32_t>* m_FirstHit;
	std::vector<CSPrimitives*> m_FirstHitPrims; //!< the copy of m_PassPrims of the PEC pass the table indexes into
	std::vector<char> m_FirstHitLine; //!< per x line: set when the PEC pass filled the table there
	FDTD_FLOAT* EC_C[3];
	FDTD_FLOAT* EC_G[3];
	FDTD_FLOAT* EC_L[3];
	FDTD_FLOAT* EC_R[3];

	AdrOp* MainOp;

	std::vector<Operator_Extension*> m_Op_exts;

	Engine* m_Engine;

	// excitation classes
	Excitation* m_Exc; // excitation time signal class
//	Operator_Ext_Excitation* m_Op_Ext_Exc; // excitation extension

	// engine/post-proc needs access
public:
	//EC operator
	ArrayLib::ArrayNIJK<FDTD_FLOAT>* vv_ptr; //calc new voltage from old voltage
	ArrayLib::ArrayNIJK<FDTD_FLOAT>* vi_ptr; //calc new voltage from old current
	ArrayLib::ArrayNIJK<FDTD_FLOAT>* ii_ptr; //calc new current from old current
	ArrayLib::ArrayNIJK<FDTD_FLOAT>* iv_ptr; //calc new current from old voltage
};

inline Operator::DebugFlags operator|( Operator::DebugFlags a, Operator::DebugFlags b ) { return static_cast<Operator::DebugFlags>(static_cast<int>(a) | static_cast<int>(b)); }
inline Operator::DebugFlags& operator|=( Operator::DebugFlags& a, const Operator::DebugFlags& b ) { return a = a | b; }

#endif // OPERATOR_H
