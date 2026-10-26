/*
*	Copyright (C) 2026 Alexandru Iercosan-Lucaci (alex.lucaci@alacrity.ro)
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <typeinfo>

#include "operator_cache.h"
#include "operator.h"
#include "operator_gpu.h"
#include "operator_sse.h"
#include "operator_sse_compressed.h"
#include "operator_multithread.h"
#include "operator_cylinder.h"
#include "extensions/operator_extension.h"
#include "tools/fingerprint.h"
#include "tools/sha256.h"
#include "gpu_build_id.h"

// after the solver headers: CSPrimCylinder.h defines PI as a macro
#include "tinyxml.h"
#include "ContinuousStructure.h"
#include "ParameterObjects.h"
#include "CSTransform.h"
#include "CSBackgroundMaterial.h"
#include "CSPrimBox.h"
#include "CSPrimMultiBox.h"
#include "CSPrimSphere.h"
#include "CSPrimSphericalShell.h"
#include "CSPrimCylinder.h"
#include "CSPrimCylindricalShell.h"
#include "CSPrimPolygon.h"
#include "CSPrimLinPoly.h"
#include "CSPrimRotPoly.h"
#include "CSPrimPolyhedron.h"
#include "CSPrimCurve.h"
#include "CSPrimWire.h"
#include "CSPrimPoint.h"
#include "CSPrimUserDefined.h"
#include "CSPropMaterial.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropConductingSheet.h"
#include "CSPropLumpedElement.h"
#include "CSPropAbsorbingBC.h"
#include "CSPropExcitation.h"

namespace fs = std::filesystem;

using std::cout;
using std::cerr;
using std::endl;

namespace
{
typedef std::chrono::steady_clock Clock;

double Since(Clock::time_point start)
{
	return std::chrono::duration<double>(Clock::now() - start).count();
}

//! A double in text that reads back to the same bits
std::string Num(double val)
{
	char buf[40];
	snprintf(buf, sizeof(buf), "%.17g", val);
	return std::string(buf);
}

std::string Fixed(double val, int digits)
{
	char buf[64];
	snprintf(buf, sizeof(buf), "%.*f", digits, val);
	return std::string(buf);
}

const int SkippedProperties = CSProperties::EXCITATION | CSProperties::PROBEBOX | CSProperties::DUMPBOX;

const char* MaterialNames[4] = {"epsR", "kappa", "mueR", "sigma"};

//! One file of an entry, written (miss), compared (verify) or read (hit) as a sequence of chunks
class EntryFile
{
public:
	enum Mode {Write, Compare, Read};

	EntryFile(const std::string& path, Mode mode) : m_Mode(mode), m_Offset(0), m_Diff(-1)
	{
		if (mode==Write)
			m_File.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
		else
			m_File.open(path.c_str(), std::ios::in | std::ios::binary);
		if (!m_File)
			throw std::runtime_error("operator cache: cannot open " + path);
		m_Path = path;
	}

	Mode GetMode() const {return m_Mode;}

	//! Write: \a data to the file; Compare: \a data against the file; Read: the file into \a data
	void Chunk(void* data, size_t bytes)
	{
		if (m_Mode==Write)
		{
			m_File.write(static_cast<const char*>(data), bytes);
			if (!m_File)
				throw std::runtime_error("operator cache: writing " + m_Path + " failed");
		}
		else if (m_Mode==Read)
		{
			m_File.read(static_cast<char*>(data), bytes);
			if (!m_File || (size_t)m_File.gcount()!=bytes)
				throw std::runtime_error("operator cache: " + m_Path + " is too short");
		}
		else if (m_Diff<0)
		{
			const char* p = static_cast<const char*>(data);
			size_t done = 0;
			while (done<bytes)
			{
				size_t n = std::min(bytes-done, (size_t)(16<<20));
				m_Buffer.resize(n);
				m_File.read(&m_Buffer[0], n);
				size_t got = m_File ? n : (size_t)m_File.gcount();
				for (size_t k=0; k<got; ++k)
					if (m_Buffer[k]!=p[done+k])
					{
						m_Diff = (long long)(m_Offset+done+k);
						return;
					}
				if (got<n)
				{
					m_Diff = (long long)(m_Offset+done+got);
					return;
				}
				done += n;
			}
		}
		m_Offset += bytes;
	}

	//! Compare: the byte offset of the first difference (a file longer than the data counts too), -1 if equal
	long long Finish()
	{
		if (m_Mode==Write)
		{
			m_File.close();
			if (!m_File)
				throw std::runtime_error("operator cache: writing " + m_Path + " failed");
			return -1;
		}
		if ((m_Mode==Compare) && (m_Diff<0) && (m_File.peek()!=EOF))
			m_Diff = (long long)m_Offset;
		if ((m_Mode==Read) && (m_File.peek()!=EOF))
			throw std::runtime_error("operator cache: " + m_Path + " is too long");
		return m_Diff;
	}

	size_t Bytes() const {return m_Offset;}

private:
	Mode m_Mode;
	std::fstream m_File;
	std::string m_Path;
	size_t m_Offset;
	long long m_Diff;
	std::vector<char> m_Buffer;
};

std::string ReadWholeFile(const std::string& path)
{
	std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
	if (!file)
		throw std::runtime_error("operator cache: cannot open " + path);
	std::ostringstream str;
	str << file.rdbuf();
	return str.str();
}

std::string JsonString(const std::string& str)
{
	std::string out = "\"";
	for (size_t i=0; i<str.size(); ++i)
	{
		unsigned char c = str[i];
		if (c=='"' || c=='\\')
		{
			out += '\\';
			out += c;
		}
		else if (c<0x20)
		{
			char buf[8];
			snprintf(buf, sizeof(buf), "\\u%04x", c);
			out += buf;
		}
		else
			out += c;
	}
	return out + "\"";
}

//! The geometry as XML without its excitation, probe and dump properties, without property IDs and colors
bool CanonicalXML(ContinuousStructure* CSX, std::string& xml, std::string& reason)
{
	TiXmlDocument doc;
	if (CSX->Write2XML(&doc, true, true)==false)
	{
		reason = "the geometry could not be written as XML";
		return false;
	}
	TiXmlElement* root = doc.FirstChildElement("ContinuousStructure");
	TiXmlElement* props = root ? root->FirstChildElement("Properties") : NULL;
	if (props==NULL)
	{
		reason = "the XML of the geometry has no properties";
		return false;
	}
	// one element per property, in the order of the property list; the IDs are list indices, which an excitation property shifts
	size_t i = 0;
	TiXmlElement* elem = props->FirstChildElement();
	while (elem)
	{
		TiXmlElement* next = elem->NextSiblingElement();
		CSProperties* prop = CSX->GetProperty(i);
		if (prop==NULL)
		{
			reason = "the XML of the geometry has more properties than the geometry";
			return false;
		}
		if (prop->GetType() & SkippedProperties)
			props->RemoveChild(elem);
		else
			elem->RemoveAttribute("ID");
		elem = next;
		++i;
	}
	if (i!=CSX->GetQtyProperties())
	{
		reason = "the XML of the geometry has fewer properties than the geometry";
		return false;
	}
	TiXmlPrinter printer;
	doc.Accept(&printer);
	xml = printer.CStr();
	return true;
}

//! The exact values of a primitive (the XML writes most of them with 6 or 7 significant digits)
bool ExactPrimitive(CSPrimitives* prim, std::ostream& s, std::string& reason)
{
	s << "primitive " << prim->GetType() << " " << prim->GetPriority() << " " << (int)prim->GetCoordinateSystem();
	if (prim->HasTransform())
	{
		double* m = prim->GetTransform()->GetMatrix();
		s << " transform";
		for (int i=0; i<16; ++i)
			s << " " << Num(m[i]);
	}
	switch (prim->GetType())
	{
	case CSPrimitives::BOX:
		if (CSPrimBox* p = dynamic_cast<CSPrimBox*>(prim))
		{
			for (int i=0; i<6; ++i)
				s << " " << Num(p->GetCoord(i));
			s << "\n";
			return true;
		}
		break;
	case CSPrimitives::MULTIBOX:
		if (CSPrimMultiBox* p = dynamic_cast<CSPrimMultiBox*>(prim))
		{
			s << " boxes " << p->GetQtyBoxes();
			for (unsigned int i=0; i<6*p->GetQtyBoxes(); ++i)
				s << " " << Num(p->GetCoord(i));
			s << "\n";
			return true;
		}
		break;
	case CSPrimitives::SPHERE:
	case CSPrimitives::SPHERICALSHELL:
		if (CSPrimSphere* p = dynamic_cast<CSPrimSphere*>(prim))
		{
			for (int i=0; i<3; ++i)
				s << " " << Num(p->GetCoord(i));
			s << " radius " << Num(p->GetRadius());
			if (CSPrimSphericalShell* shell = dynamic_cast<CSPrimSphericalShell*>(prim))
				s << " width " << Num(shell->GetShellWidth());
			s << "\n";
			return true;
		}
		break;
	case CSPrimitives::CYLINDER:
	case CSPrimitives::CYLINDRICALSHELL:
		if (CSPrimCylinder* p = dynamic_cast<CSPrimCylinder*>(prim))
		{
			for (int i=0; i<6; ++i)
				s << " " << Num(p->GetCoord(i));
			s << " radius " << Num(p->GetRadius());
			if (CSPrimCylindricalShell* shell = dynamic_cast<CSPrimCylindricalShell*>(prim))
				s << " width " << Num(shell->GetShellWidth());
			s << "\n";
			return true;
		}
		break;
	case CSPrimitives::POLYGON:
	case CSPrimitives::LINPOLY:
	case CSPrimitives::ROTPOLY:
		if (CSPrimPolygon* p = dynamic_cast<CSPrimPolygon*>(prim))
		{
			s << " normal " << p->GetNormDir() << " elevation " << Num(p->GetElevation());
			if (CSPrimLinPoly* lin = dynamic_cast<CSPrimLinPoly*>(prim))
				s << " length " << Num(lin->GetLength());
			if (CSPrimRotPoly* rot = dynamic_cast<CSPrimRotPoly*>(prim))
				s << " axis " << rot->GetRotAxisDir() << " angles " << Num(rot->GetAngle(0)) << " " << Num(rot->GetAngle(1));
			s << " coords " << 2*p->GetQtyCoords();
			for (size_t i=0; i<2*p->GetQtyCoords(); ++i)
				s << " " << Num(p->GetCoord((int)i));
			s << "\n";
			return true;
		}
		break;
	case CSPrimitives::POLYHEDRON:
	case CSPrimitives::POLYHEDRONREADER:
		if (CSPrimPolyhedron* p = dynamic_cast<CSPrimPolyhedron*>(prim))
		{
			s << " vertices " << p->GetNumVertices();
			for (unsigned int i=0; i<p->GetNumVertices(); ++i)
			{
				double* v = p->GetVertex(i);
				s << " " << Num(v[0]) << " " << Num(v[1]) << " " << Num(v[2]);
			}
			s << " faces " << p->GetNumFaces();
			for (unsigned int i=0; i<p->GetNumFaces(); ++i)
			{
				unsigned int num = 0;
				int* f = p->GetFace(i, num);
				s << " " << (p->GetFaceValid(i) ? "v" : "i") << num;
				for (unsigned int k=0; k<num; ++k)
					s << " " << f[k];
			}
			s << "\n";
			return true;
		}
		break;
	case CSPrimitives::CURVE:
	case CSPrimitives::WIRE:
		if (CSPrimCurve* p = dynamic_cast<CSPrimCurve*>(prim))
		{
			s << " points " << p->GetNumberOfPoints();
			double pt[3];
			for (size_t i=0; i<p->GetNumberOfPoints(); ++i)
			{
				p->GetPoint(i, pt);
				s << " " << Num(pt[0]) << " " << Num(pt[1]) << " " << Num(pt[2]);
			}
			if (CSPrimWire* wire = dynamic_cast<CSPrimWire*>(prim))
				s << " radius " << Num(wire->GetWireRadius());
			s << "\n";
			return true;
		}
		break;
	case CSPrimitives::POINT:
		if (CSPrimPoint* p = dynamic_cast<CSPrimPoint*>(prim))
		{
			for (int i=0; i<3; ++i)
				s << " " << Num(p->GetCoord(i));
			s << "\n";
			return true;
		}
		break;
	case CSPrimitives::USERDEFINED:
		if (CSPrimUserDefined* p = dynamic_cast<CSPrimUserDefined*>(prim))
		{
			s << " system " << (int)p->GetCoordSystem();
			for (int i=0; i<3; ++i)
				s << " " << Num(p->GetCoordShift(i));
			s << " function " << p->GetFunction() << "\n";
			return true;
		}
		break;
	default:
		break;
	}
	reason = "primitive type " + std::to_string(prim->GetType()) + " (" + prim->GetTypeName() + ") is not covered by the key";
	return false;
}

//! The exact values of the geometry's parameters, properties and primitives, excitation, probe and dump properties left out
bool ExactCSX(ContinuousStructure* CSX, std::string& exact, std::string& reason)
{
	std::ostringstream s;
	ParameterSet* paraSet = CSX->GetParameterSet();
	for (size_t i=0; paraSet && (i<paraSet->GetQtyParameter()); ++i)
	{
		Parameter* para = paraSet->GetParameter(i);
		s << "parameter " << para->GetName() << " " << Num(para->GetValue()) << "\n";
	}
	CSBackgroundMaterial* bg = CSX->GetBackgroundMaterial();
	s << "background " << Num(bg->GetEpsilon()) << " " << Num(bg->GetMue()) << " " << Num(bg->GetKappa()) << " " << Num(bg->GetSigma()) << "\n";
	for (size_t i=0; i<CSX->GetQtyProperties(); ++i)
	{
		CSProperties* prop = CSX->GetProperty(i);
		if (prop->GetType() & SkippedProperties)
			continue;
		if (prop->GetType() & CSProperties::DISCRETE_MATERIAL)
		{
			reason = "discrete material '" + prop->GetName() + "' is read from a file";
			return false;
		}
		s << "property " << prop->GetType() << " " << prop->GetName() << "\n";
		if (CSPropMaterial* mat = prop->ToMaterial())
		{
			s << "material " << mat->GetIsotropy();
			for (int ny=0; ny<3; ++ny)
				s << " " << Num(mat->GetEpsilon(ny)) << " " << Num(mat->GetMue(ny)) << " " << Num(mat->GetKappa(ny)) << " " << Num(mat->GetSigma(ny));
			s << " " << Num(mat->GetDensity()) << "\n";
		}
		if (CSPropLorentzMaterial* lor = dynamic_cast<CSPropLorentzMaterial*>(prop))
		{
			s << "lorentz " << lor->GetDispersionOrder();
			for (int o=0; o<lor->GetDispersionOrder(); ++o)
				for (int ny=0; ny<3; ++ny)
					s << " " << Num(lor->GetEpsPlasmaFreq(o,ny)) << " " << Num(lor->GetEpsLorPoleFreq(o,ny)) << " " << Num(lor->GetEpsRelaxTime(o,ny))
						<< " " << Num(lor->GetMuePlasmaFreq(o,ny)) << " " << Num(lor->GetMueLorPoleFreq(o,ny)) << " " << Num(lor->GetMueRelaxTime(o,ny));
			s << "\n";
		}
		if (CSPropDebyeMaterial* deb = dynamic_cast<CSPropDebyeMaterial*>(prop))
		{
			s << "debye " << deb->GetDispersionOrder();
			for (int o=0; o<deb->GetDispersionOrder(); ++o)
				for (int ny=0; ny<3; ++ny)
					s << " " << Num(deb->GetEpsDelta(o,ny)) << " " << Num(deb->GetEpsRelaxTime(o,ny));
			s << "\n";
		}
		if (CSPropConductingSheet* cs = dynamic_cast<CSPropConductingSheet*>(prop))
			s << "sheet " << Num(cs->GetConductivity()) << " " << Num(cs->GetThickness()) << "\n";
		if (CSPropLumpedElement* le = dynamic_cast<CSPropLumpedElement*>(prop))
			s << "lumped " << Num(le->GetResistance()) << " " << Num(le->GetCapacity()) << " " << Num(le->GetInductance())
				<< " " << le->GetDirection() << " " << le->GetCaps() << " " << (int)le->GetLEtype() << "\n";
		if (CSPropAbsorbingBC* abc = dynamic_cast<CSPropAbsorbingBC*>(prop))
			s << "absorbing " << Num(abc->GetPhaseVelocity()) << " " << abc->GetNormalSignPositive() << "\n";
		for (size_t p=0; p<prop->GetQtyPrimitives(); ++p)
			if (ExactPrimitive(prop->GetPrimitive(p), s, reason)==false)
				return false;
	}
	exact = s.str();
	return true;
}

//! One operator array (0..3: vv, vi, ii, iv) in the stored order of ArrayNIJK (n, i, j, k)
ArrayLib::ArrayNIJK<FDTD_FLOAT>* OperatorArray(Operator* op, int array)
{
	switch (array)
	{
	case 0:
		return op->vv_ptr;
	case 1:
		return op->vi_ptr;
	case 2:
		return op->ii_ptr;
	default:
		return op->iv_ptr;
	}
}

//! Write, compare or read one operator array: the raw ArrayNIJK if the operator holds one, else through the accessors (packed SSE layouts) in the same order
void OperatorArrayIO(Operator* op, int array, EntryFile& file)
{
	ArrayLib::ArrayNIJK<FDTD_FLOAT>* arr = OperatorArray(op, array);
	if (arr!=NULL)
	{
		file.Chunk(arr->data(), arr->size()*sizeof(FDTD_FLOAT));
		return;
	}
	unsigned int numLines[3] = {op->GetNumberOfLines(0,true), op->GetNumberOfLines(1,true), op->GetNumberOfLines(2,true)};
	std::vector<FDTD_FLOAT> row((size_t)numLines[1]*numLines[2]);
	for (unsigned int n=0; n<3; ++n)
		for (unsigned int i=0; i<numLines[0]; ++i)
		{
			size_t idx = 0;
			if (file.GetMode()!=EntryFile::Read)
				for (unsigned int j=0; j<numLines[1]; ++j)
					for (unsigned int k=0; k<numLines[2]; ++k)
						row[idx++] = (array==0) ? op->GetVV(n,i,j,k) : (array==1) ? op->GetVI(n,i,j,k) : (array==2) ? op->GetII(n,i,j,k) : op->GetIV(n,i,j,k);
			file.Chunk(row.data(), row.size()*sizeof(FDTD_FLOAT));
			if (file.GetMode()!=EntryFile::Read)
				continue;
			for (unsigned int j=0; j<numLines[1]; ++j)
				for (unsigned int k=0; k<numLines[2]; ++k)
				{
					FDTD_FLOAT val = row[idx++];
					if (array==0)
						op->SetVV(n,i,j,k,val);
					else if (array==1)
						op->SetVI(n,i,j,k,val);
					else if (array==2)
						op->SetII(n,i,j,k,val);
					else
						op->SetIV(n,i,j,k,val);
				}
		}
}

const char* ArrayFiles[4] = {"vv.f32", "vi.f32", "ii.f32", "iv.f32"};

std::string ExtFile(size_t n, const std::string& name)
{
	std::string file = "ext_" + std::to_string(n) + "_";
	for (size_t i=0; i<name.size(); ++i)
		file += isalnum((unsigned char)name[i]) ? name[i] : '_';
	return file + ".bin";
}
}

Operator_Cache::Operator_Cache(Operator* op, int debugFlags)
{
	m_Op = op;
	m_State = Off_State;
	m_ForcedTS = 0;
	m_Bytes = 0;
	m_Seconds = 0;

	const char* dir = getenv("OPENEMS_OPERATOR_CACHE");
	if ((dir==NULL) || (dir[0]==0))
		return;
	m_Dir = dir;
	m_ForcedTS = op->dT;

	if (debugFlags!=0)
	{
		Bypass("a material, operator or PEC dump is requested");
		return;
	}

	std::string reason;
	if (BuildKey(reason)==false)
	{
		Bypass(reason);
		return;
	}

	m_Entry = (fs::path(m_Dir) / m_Key).string();
	std::error_code ec;
	if (fs::is_directory(m_Entry, ec)==false)
	{
		m_State = Miss_State;
		return;
	}
	if (ReadEntry(reason)==false)
	{
		Bypass(reason);
		return;
	}
	if (FNVHash::EnvOn("OPENEMS_OPERATOR_CACHE_VERIFY"))
	{
		m_State = Verify_State;
		cout << "operator cache: hit " << m_Key << ", verify requested: building" << endl;
	}
	else
		m_State = Hit_State;
}

void Operator_Cache::Bypass(const std::string& reason)
{
	m_State = Bypass_State;
	cout << "operator cache: bypass: " << reason << endl;
}

std::string Operator_Cache::Path(const std::string& name) const
{
	return (fs::path(m_State==Miss_State ? m_Write : m_Entry) / name).string();
}

bool Operator_Cache::BuildKey(std::string& reason)
{
	Operator* op = m_Op;
	const std::type_info& type = typeid(*op);
	if (dynamic_cast<Operator_Cylinder*>(op))
	{
		reason = "cylindrical operator";
		return false;
	}
	if (type==typeid(Operator))
		m_Class = "Operator";
	else if (type==typeid(Operator_GPU))
		m_Class = "Operator_GPU";
	else if (type==typeid(Operator_sse))
		m_Class = "Operator_sse";
	else if (type==typeid(Operator_SSE_Compressed))
		m_Class = "Operator_SSE_Compressed";
	else if (type==typeid(Operator_Multithread))
		m_Class = "Operator_Multithread";
	else
	{
		reason = std::string("operator class ") + type.name() + " is not supported";
		return false;
	}
	uint16_t one = 1;
	if (*reinterpret_cast<unsigned char*>(&one)!=1)
	{
		reason = "big-endian host";
		return false;
	}
	ContinuousStructure* CSX = op->CSX;
	Excitation* exc = op->m_Exc;
	if ((CSX==NULL) || (exc==NULL))
	{
		reason = "no geometry or no excitation";
		return false;
	}

	// a hit swaps the excitation: one that writes the operator arrays (a hard source) or reads them (a plane wave through TF/SF) cannot be swapped
	std::vector<CSProperties*> excProps = CSX->GetPropertyByType(CSProperties::EXCITATION);
	for (size_t i=0; i<excProps.size(); ++i)
	{
		CSPropExcitation* elec = excProps.at(i)->ToExcitation();
		if (elec && (elec->GetExcitType()!=0) && (elec->GetExcitType()!=2))
		{
			reason = "excitation '" + elec->GetName() + "' of type " + std::to_string(elec->GetExcitType()) + " writes or reads the operator";
			return false;
		}
	}

	std::ostringstream key;
	key << "openEMS operator cache 1\n";
	key << "version " << OPENEMS_GPU_BUILD_ID << "\n";
	key << "operator " << m_Class << "\n";
	key << "float_bytes " << sizeof(FDTD_FLOAT) << "\n";
	key << "coordinate_system " << (int)op->m_MeshType << "\n";
	key << "grid_delta " << Num(op->gridDelta) << "\n";
	for (int n=0; n<3; ++n)
	{
		key << "lines_" << "xyz"[n] << " " << op->numLines[n];
		for (unsigned int i=0; i<op->numLines[n]; ++i)
			key << (i ? "," : " ") << Num(op->discLines[n][i]);
		key << "\n";
	}
	key << "boundary_conditions";
	for (int n=0; n<6; ++n)
		key << " " << op->m_BC[n];
	key << "\nboundary_sizes";
	for (int n=0; n<6; ++n)
		key << " " << op->m_BC_Size[n];
	key << "\nbackground " << Num(op->m_BG_epsR) << " " << Num(op->m_BG_mueR) << " " << Num(op->m_BG_kappa) << " " << Num(op->m_BG_sigma) << " " << Num(op->m_BG_density) << "\n";
	key << "drawing_tolerance " << Num(CSX->GetDrawingTolerance()) << "\n";
	key << "material_averaging " << (int)op->m_MatAverageMethod << "\n";
	key << "timestep_method " << op->m_TimeStepVar << "\n";
	key << "forced_timestep " << Num(op->dT) << "\n";
	key << "timestep_factor " << Num(op->m_TimeStepFactor) << "\n";
	key << "excitation_type " << exc->GetExciteType() << "\n";
	key << "excitation_max_freq " << Num(exc->GetMaxFreq()) << "\n";
	key << "excitation_signal_period " << Num(exc->GetSignalPeriod()) << "\n";
	key << "extensions " << op->m_Op_exts.size() << "\n";
	for (size_t n=0; n<op->m_Op_exts.size(); ++n)
	{
		Operator_Extension* ext = op->m_Op_exts.at(n);
		m_ExtNames.push_back(ext->GetExtensionName());
		key << "extension " << n << " " << ext->GetExtensionName();
		std::string settings = ext->CacheKey();
		if (settings.empty()==false)
			key << " | " << settings;
		key << "\n";
	}

	std::string xml, xml2, exact;
	if ((CanonicalXML(CSX, xml, reason)==false) || (CanonicalXML(CSX, xml2, reason)==false))
		return false;
	if (xml!=xml2)
	{
		reason = "the XML writer of the geometry is not deterministic";
		return false;
	}
	if (ExactCSX(CSX, exact, reason)==false)
		return false;
	key << "csx_xml " << xml.size() << "\n" << xml << "\n";
	key << "csx_exact " << exact.size() << "\n" << exact;
	m_CSXInputs = "  \"csx_xml\": {\"bytes\": " + std::to_string(xml.size()) + ", \"sha256\": \"" + SHA256::Hex(xml) + "\"},\n";
	m_CSXInputs += "  \"csx_exact\": {\"bytes\": " + std::to_string(exact.size()) + ", \"sha256\": \"" + SHA256::Hex(exact) + "\"},\n";

	m_KeyInputs = key.str();
	m_Key = SHA256::Hex(m_KeyInputs);
	return true;
}

bool Operator_Cache::ReadEntry(std::string& reason)
{
	try
	{
		if (ReadWholeFile(Path("key_inputs.txt"))!=m_KeyInputs)
		{
			reason = "entry " + m_Key + " was written for other inputs";
			return false;
		}
		std::istringstream table(ReadWholeFile(Path("extensions")));
		std::string line;
		while (std::getline(table, line))
		{
			std::istringstream fields(line);
			size_t index = 0, bytes = 0;
			int cached = 0, active = 0;
			std::string name;
			if (!(fields >> index >> cached >> active >> bytes))
				break;
			std::getline(fields >> std::ws, name);
			if (index!=m_ExtCached.size())
				break;
			m_ExtCached.push_back(cached);
			m_ExtActive.push_back(active);
			m_ExtBytes.push_back(bytes);
			if ((index>=m_ExtNames.size()) || (m_ExtNames.at(index)!=name))
				break;
		}
		if (m_ExtCached.size()!=m_ExtNames.size())
		{
			reason = "the extensions of entry " + m_Key + " differ from this run's";
			return false;
		}
		size_t cells = (size_t)m_Op->numLines[0]*m_Op->numLines[1]*m_Op->numLines[2];
		std::vector<std::pair<std::string, size_t> > sizes;
		for (int a=0; a<4; ++a)
			sizes.push_back(std::make_pair(std::string(ArrayFiles[a]), 3*cells*sizeof(FDTD_FLOAT)));
		sizes.push_back(std::make_pair(std::string("dT"), sizeof(double)));
		sizes.push_back(std::make_pair(std::string("nr_pec"), 3*sizeof(uint32_t)));
		for (size_t n=0; n<m_ExtCached.size(); ++n)
			if (m_ExtCached[n]==1)
				sizes.push_back(std::make_pair(ExtFile(n, m_ExtNames[n]), m_ExtBytes[n]));
		for (int m=0; m<4; ++m)
			if (m_Op->m_StoreMaterial[m])
			{
				std::error_code ec;
				if (fs::exists(Path(std::string("material_") + MaterialNames[m] + ".f32"), ec)==false)
				{
					reason = std::string("material storage ") + MaterialNames[m] + " requested, not in entry " + m_Key;
					return false;
				}
				sizes.push_back(std::make_pair(std::string("material_") + MaterialNames[m] + ".f32", 3*cells*sizeof(float)));
			}
		for (size_t i=0; i<sizes.size(); ++i)
		{
			std::error_code ec;
			uintmax_t size = fs::file_size(Path(sizes[i].first), ec);
			if (ec || (size!=sizes[i].second))
			{
				reason = "entry " + m_Key + " is damaged: " + sizes[i].first + " has not " + std::to_string(sizes[i].second) + " bytes";
				return false;
			}
		}
	}
	catch (const std::exception& e)
	{
		reason = "entry " + m_Key + " is unreadable: " + e.what();
		return false;
	}
	return true;
}

std::string Operator_Cache::TimestepState() const
{
	std::ostringstream state;
	WriteValue(state, m_Op->opt_dT);
	WriteValue(state, (uint8_t)m_Op->m_InvaildTimestep);
	state << m_Op->m_Used_TS_Name;
	return state.str();
}

std::string Operator_Cache::UsedPrimitives() const
{
	std::string used;
	ContinuousStructure* CSX = m_Op->CSX;
	for (size_t i=0; i<CSX->GetQtyProperties(); ++i)
	{
		CSProperties* prop = CSX->GetProperty(i);
		if (prop->GetType() & SkippedProperties)
			continue;
		for (size_t p=0; p<prop->GetQtyPrimitives(); ++p)
			used += prop->GetPrimitive(p)->GetPrimitiveUsed() ? '1' : '0';
	}
	return used;
}

void Operator_Cache::Load()
{
	Clock::time_point start = Clock::now();
	Operator* op = m_Op;

	// the multithreaded operator has started its build threads already
	op->SkipBuildThreads();

	double dT = 0;
	EntryFile dT_file(Path("dT"), EntryFile::Read);
	dT_file.Chunk(&dT, sizeof(dT));
	dT_file.Finish();
	std::istringstream state(ReadWholeFile(Path("timestep")));
	op->opt_dT = ReadValue<double>(state);
	op->m_InvaildTimestep = ReadValue<uint8_t>(state)!=0;
	std::getline(state, op->m_Used_TS_Name, '\0');
	m_Bytes += sizeof(dT) + state.str().size();

	// the warning of the timestep phase, as the build prints it
	if (op->m_InvaildTimestep)
		cerr << "Operator::CalcECOperator: Warning, forced timestep: " << m_ForcedTS << "s is larger than calculated timestep: " << op->opt_dT << "s! It is not recommended using this timestep!! " << endl;
	op->dT = dT;
	op->m_Exc->Reset(dT);

	op->InitOperator();
	for (int a=0; a<4; ++a)
	{
		EntryFile file(Path(ArrayFiles[a]), EntryFile::Read);
		OperatorArrayIO(op, a, file);
		file.Finish();
		m_Bytes += file.Bytes();
	}

	uint32_t pec[3];
	EntryFile pec_file(Path("nr_pec"), EntryFile::Read);
	pec_file.Chunk(pec, sizeof(pec));
	pec_file.Finish();
	for (int n=0; n<3; ++n)
		op->m_Nr_PEC[n] = pec[n];

	op->InitDataStorage();
	ArrayLib::ArrayNIJK<float>* material[4] = {op->m_epsR_ptr, op->m_kappa_ptr, op->m_mueR_ptr, op->m_sigma_ptr};
	for (int m=0; m<4; ++m)
	{
		if (material[m]==NULL)
			continue;
		EntryFile file(Path(std::string("material_") + MaterialNames[m] + ".f32"), EntryFile::Read);
		file.Chunk(material[m]->data(), material[m]->size()*sizeof(float));
		file.Finish();
		m_Bytes += file.Bytes();
	}

	// the timestep search leaves the address operator reflecting at the mesh border
	op->MainOp->SetReflection2Cell();

	std::string used = ReadWholeFile(Path("used.bin"));
	if (used.size()!=UsedPrimitives().size())
		throw std::runtime_error("operator cache: used.bin of entry " + m_Key + " does not fit the geometry");
	ContinuousStructure* CSX = op->CSX;
	size_t idx = 0;
	for (size_t i=0; i<CSX->GetQtyProperties(); ++i)
	{
		CSProperties* prop = CSX->GetProperty(i);
		if (prop->GetType() & SkippedProperties)
			continue;
		for (size_t p=0; p<prop->GetQtyPrimitives(); ++p)
			if (used[idx++]=='1')
				prop->GetPrimitive(p)->SetPrimitiveUsed(true);
	}
	m_Bytes += used.size();
	m_Seconds += Since(start);
}

bool Operator_Cache::LoadExtension(size_t n)
{
	if (m_ExtCached.at(n)!=1)
		return false;
	Operator_Extension* ext = m_Op->m_Op_exts.at(n);
	if (m_ExtActive.at(n)==0)
	{
		ext->SetActive(false); // the build of the entry dropped it
		return true;
	}
	Clock::time_point start = Clock::now();
	std::string path = Path(ExtFile(n, m_ExtNames.at(n)));
	std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
	if (!file || (ext->Deserialize(file)==false))
		throw std::runtime_error("operator cache: extension '" + m_ExtNames.at(n) + "' could not be restored from " + path);
	if (file.peek()!=EOF)
		throw std::runtime_error("operator cache: extension '" + m_ExtNames.at(n) + "' left bytes unread in " + path);
	m_Bytes += m_ExtBytes.at(n);
	m_Seconds += Since(start);
	return true;
}

void Operator_Cache::ExtensionsBuilt()
{
	if ((m_State!=Miss_State) && (m_State!=Verify_State))
		return;
	Clock::time_point start = Clock::now();
	std::vector<Operator_Extension*>& exts = m_Op->m_Op_exts;
	if (m_State==Miss_State)
	{
		try
		{
			std::random_device rd;
			char suffix[32];
			snprintf(suffix, sizeof(suffix), "%08x%08x", (unsigned int)rd(), (unsigned int)rd());
			fs::create_directories(m_Dir);
			m_Write = (fs::path(m_Dir) / (".tmp-" + m_Key.substr(0,16) + "-" + suffix)).string();
			fs::create_directory(m_Write);
			std::ostringstream table;
			for (size_t n=0; n<exts.size(); ++n)
			{
				std::string path = Path(ExtFile(n, m_ExtNames.at(n)));
				std::ofstream file(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
				bool cached = exts.at(n)->Serialize(file);
				file.close();
				if (!file)
					throw std::runtime_error("writing " + path + " failed");
				size_t bytes = 0;
				if (cached)
					bytes = fs::file_size(path);
				else
					fs::remove(path);
				table << n << " " << (int)cached << " " << (int)exts.at(n)->IsActive() << " " << bytes << " " << m_ExtNames.at(n) << "\n";
				m_Bytes += bytes;
			}
			m_ExtTable = table.str();
		}
		catch (const std::exception& e)
		{
			cout << "operator cache: miss " << m_Key << ", not written: " << e.what() << endl;
			std::error_code ec;
			if (m_Write.empty()==false)
				fs::remove_all(m_Write, ec);
			m_State = Bypass_State;
		}
	}
	else
	{
		for (size_t n=0; (n<exts.size()) && m_Failed.empty(); ++n)
		{
			std::ostringstream state;
			bool cached = exts.at(n)->Serialize(state);
			if (((int)cached!=m_ExtCached.at(n)) || ((int)exts.at(n)->IsActive()!=m_ExtActive.at(n)))
			{
				m_Failed = "extension " + std::to_string(n) + " '" + m_ExtNames.at(n) + "' (cached or active differs)";
				break;
			}
			if (!cached)
				continue;
			std::string bytes = state.str();
			EntryFile file(Path(ExtFile(n, m_ExtNames.at(n))), EntryFile::Compare);
			file.Chunk(&bytes[0], bytes.size());
			long long diff = file.Finish();
			if (diff>=0)
				m_Failed = "extension " + std::to_string(n) + " '" + m_ExtNames.at(n) + "' byte " + std::to_string(diff);
			m_Bytes += bytes.size();
		}
	}
	m_Seconds += Since(start);
}

void Operator_Cache::Finish(const std::vector<std::string>& fingerprint)
{
	if (m_State==Hit_State)
	{
		cout << "operator cache: hit " << m_Key << ", loaded " << Fixed(m_Bytes/1e6, 1) << " MB in " << Fixed(m_Seconds, 3) << " s" << endl;
		return;
	}
	if ((m_State!=Miss_State) && (m_State!=Verify_State))
		return;

	Clock::time_point start = Clock::now();
	Operator* op = m_Op;
	EntryFile::Mode mode = (m_State==Miss_State) ? EntryFile::Write : EntryFile::Compare;
	std::vector<std::pair<std::string, size_t> > files;

	// one file of the entry: written (miss) or compared (verify, the first difference is kept)
	auto entry_file = [&](const std::string& name, void* data, size_t bytes, int array)
	{
		if (!m_Failed.empty())
			return;
		EntryFile file(Path(name), mode);
		if (array>=0)
			OperatorArrayIO(op, array, file);
		else
			file.Chunk(data, bytes);
		long long diff = file.Finish();
		if (diff>=0)
			m_Failed = name + " byte " + std::to_string(diff);
		files.push_back(std::make_pair(name, file.Bytes()));
		m_Bytes += file.Bytes();
	};

	std::vector<std::string> material;
	try
	{
		for (int a=0; a<4; ++a)
			entry_file(ArrayFiles[a], NULL, 0, a);
		double dT = op->dT;
		entry_file("dT", &dT, sizeof(dT), -1);
		uint32_t pec[3] = {op->m_Nr_PEC[0], op->m_Nr_PEC[1], op->m_Nr_PEC[2]};
		entry_file("nr_pec", pec, sizeof(pec), -1);
		// the timestep state: the timestep search's result (with a forced timestep), its validity, the method's name
		std::string state = TimestepState();
		entry_file("timestep", &state[0], state.size(), -1);
		ArrayLib::ArrayNIJK<float>* storage[4] = {op->m_epsR_ptr, op->m_kappa_ptr, op->m_mueR_ptr, op->m_sigma_ptr};
		for (int m=0; m<4; ++m)
			if (storage[m])
			{
				entry_file(std::string("material_") + MaterialNames[m] + ".f32", storage[m]->data(), storage[m]->size()*sizeof(float), -1);
				material.push_back(MaterialNames[m]);
			}
		std::string used = UsedPrimitives();
		entry_file("used.bin", &used[0], used.size(), -1);

		if (m_State==Verify_State)
		{
			if (m_Failed.empty())
				cout << "operator cache: verify OK " << m_Key << ", compared " << Fixed(m_Bytes/1e6, 1) << " MB in " << Fixed(m_Seconds+Since(start), 3) << " s" << endl;
			else
			{
				cout << "operator cache: verify FAILED at " << m_Failed << endl;
				throw std::runtime_error("operator cache: verify FAILED at " + m_Failed);
			}
			return;
		}

		entry_file("extensions", &m_ExtTable[0], m_ExtTable.size(), -1);
		entry_file("key_inputs.txt", &m_KeyInputs[0], m_KeyInputs.size(), -1);

		// the manifest, for a reviewer: every input of the key in clear (the geometry as the SHA-256 of its two parts), the files, the fingerprint
		std::ostringstream json;
		std::time_t now = std::time(NULL);
		char when[32];
		strftime(when, sizeof(when), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
		json << "{\n  \"format\": \"openEMS operator cache 1\",\n  \"key\": " << JsonString(m_Key) << ",\n  \"written\": " << JsonString(when) << ",\n";
		json << "  \"key_inputs_file\": \"key_inputs.txt\",\n  \"inputs\": [";
		std::istringstream inputs(m_KeyInputs);
		std::string line;
		bool first = true;
		while (std::getline(inputs, line))
		{
			if (line.compare(0, 8, "csx_xml ")==0)
				break;
			json << (first ? "\n" : ",\n") << "    " << JsonString(line);
			first = false;
		}
		json << "\n  ],\n" << m_CSXInputs;
		json << "  \"files\": [";
		for (size_t i=0; i<files.size(); ++i)
			json << (i ? ",\n" : "\n") << "    {\"name\": " << JsonString(files[i].first) << ", \"bytes\": " << files[i].second << "}";
		json << "\n  ],\n  \"extensions\": [";
		std::istringstream table(m_ExtTable);
		for (size_t n=0; std::getline(table, line); ++n)
		{
			std::istringstream fields(line);
			size_t index = 0, bytes = 0;
			int cached = 0, active = 0;
			fields >> index >> cached >> active >> bytes;
			json << (n ? ",\n" : "\n") << "    {\"index\": " << index << ", \"name\": " << JsonString(m_ExtNames.at(index)) << ", \"cached\": " << (cached ? "true" : "false")
				<< ", \"active\": " << (active ? "true" : "false") << ", \"file\": " << (cached ? JsonString(ExtFile(index, m_ExtNames.at(index))) : std::string("null")) << ", \"bytes\": " << bytes << "}";
		}
		json << "\n  ],\n  \"material_storage\": [";
		for (size_t i=0; i<material.size(); ++i)
			json << (i ? ", " : "") << JsonString(material[i]);
		json << "],\n  \"fingerprint\": [";
		for (size_t i=0; i<fingerprint.size(); ++i)
			json << (i ? ",\n" : "\n") << "    " << JsonString(fingerprint[i]);
		json << (fingerprint.empty() ? "]\n}\n" : "\n  ]\n}\n");
		std::string manifest = json.str();
		entry_file("manifest.json", &manifest[0], manifest.size(), -1);

		std::error_code ec;
		fs::rename(m_Write, m_Entry, ec);
		std::string note;
		if (ec)
		{
			if (fs::is_directory(m_Entry))
				note = " (another run wrote the entry meanwhile, kept that one)";
			else
				throw std::runtime_error("renaming " + m_Write + " to " + m_Entry + " failed: " + ec.message());
			fs::remove_all(m_Write, ec);
		}
		m_Seconds += Since(start);
		cout << "operator cache: miss " << m_Key << ", written " << Fixed(m_Bytes/1e6, 1) << " MB in " << Fixed(m_Seconds, 3) << " s" << note << endl;
	}
	catch (const std::exception& e)
	{
		if (m_State==Verify_State)
			throw;
		cout << "operator cache: miss " << m_Key << ", not written: " << e.what() << endl;
		std::error_code ec;
		fs::remove_all(m_Write, ec);
	}
}
