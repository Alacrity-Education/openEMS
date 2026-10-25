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

#ifndef OPERATOR_CACHE_H
#define OPERATOR_CACHE_H

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

class Operator;

//! Cache of the built operator across runs of one geometry, active with OPENEMS_OPERATOR_CACHE=<dir> (default off)
/*!
	A run that finds no entry for its key builds the operator as usual and writes it to <dir>/<key>/ (a miss);
	a later run with the same key loads it instead of building it (a hit) and builds only the extensions that
	are not cacheable (the excitation, which differs between the ports of a sweep). The key is a SHA-256 over
	everything the build reads: the solver build, the operator class, the mesh, the boundary conditions, the
	timestep settings, the excitation parameters that reach the operator, the extensions and the geometry
	without its excitation, probe and dump properties. Cases the cache cannot reproduce are built as usual
	(a bypass). OPENEMS_OPERATOR_CACHE_VERIFY=1 builds on a hit as well and compares with the entry.
	Without OPENEMS_OPERATOR_CACHE nothing is computed and no file is touched.
*/
class Operator_Cache
{
public:
	Operator_Cache(Operator* op, int debugFlags);

	//! True if the operator is to be loaded from the cache instead of built
	bool Hit() const {return m_State==Hit_State;}

	//! Hit: load the operator arrays, the timestep, the PEC counts, the material storage and the used primitives (throws on a damaged entry)
	void Load();

	//! Hit: restore extension \a n if it is cacheable (returns true; it is deactivated if the build of the entry dropped it), else false: build it
	bool LoadExtension(size_t n);

	//! Miss or verify: called after the extensions are built, before the inactive ones are removed
	void ExtensionsBuilt();

	//! Miss: write the entry; verify: compare with it (throws on a difference). \a fingerprint: the fingerprint lines, if printed
	void Finish(const std::vector<std::string>& fingerprint);

	//! Raw host bytes of \a count values, for Operator_Extension::Serialize
	template <typename T>
	static void Write(std::ostream& out, const T* data, size_t count)
	{
		if (count)
			out.write(reinterpret_cast<const char*>(data), count*sizeof(T));
	}
	template <typename T>
	static void WriteValue(std::ostream& out, const T& value) {Write(out, &value, 1);}

	//! Read what Write wrote, for Operator_Extension::Deserialize; throws on a short stream
	template <typename T>
	static void Read(std::istream& in, T* data, size_t count)
	{
		if (count==0)
			return;
		in.read(reinterpret_cast<char*>(data), count*sizeof(T));
		if (!in || (size_t)in.gcount()!=count*sizeof(T))
			throw std::runtime_error("operator cache: an extension file of the entry is too short");
	}
	template <typename T>
	static T ReadValue(std::istream& in) {T value; Read(in, &value, 1); return value;}

protected:
	enum State {Off_State, Bypass_State, Miss_State, Hit_State, Verify_State};

	Operator* m_Op;
	State m_State;
	std::string m_Dir;		//!< the cache directory
	std::string m_Key;		//!< SHA-256 hex of m_KeyInputs
	std::string m_KeyInputs;	//!< the canonical byte string of the key
	std::string m_CSXInputs;	//!< the manifest lines of the geometry's two parts (size, SHA-256)
	std::string m_Class;		//!< the operator class name
	std::string m_Entry;		//!< <dir>/<key>
	std::string m_Write;		//!< miss: the temporary directory written before the rename
	double m_ForcedTS;		//!< the forced timestep at the start of CalcECOperator (0: none)
	double m_Bytes;			//!< bytes written, loaded or compared
	double m_Seconds;		//!< seconds spent writing, loading or comparing
	std::vector<std::string> m_ExtNames;	//!< the extensions before the build
	std::vector<int> m_ExtCached;		//!< per extension: 1 if cacheable (entry), -1 unknown
	std::vector<int> m_ExtActive;		//!< per extension: 1 if the build kept it (entry)
	std::vector<size_t> m_ExtBytes;		//!< per extension: bytes of its file (entry)
	std::string m_ExtTable;			//!< the extension table (miss: as written)
	std::string m_Failed;			//!< verify: the first difference

	void Bypass(const std::string& reason);
	bool BuildKey(std::string& reason);
	bool ReadEntry(std::string& reason);
	std::string Path(const std::string& name) const;
	std::string TimestepState() const;
	std::string UsedPrimitives() const;
};

#endif // OPERATOR_CACHE_H
