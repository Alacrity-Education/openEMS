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

#ifndef FINGERPRINT_H
#define FINGERPRINT_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

//! FNV-1a 64-bit hash over raw bytes, used for the operator fingerprint (OPENEMS_OPERATOR_FINGERPRINT=1)
/*!
	The fingerprint is a debugging oracle: two operators with equal fingerprints hold bit-identical coefficient
	arrays. It is printed at the end of Operator::CalcECOperator and is never part of a result.
*/
class FNVHash
{
public:
	FNVHash() : m_hash(0xcbf29ce484222325ULL) {}

	//! True if the environment variable OPENEMS_OPERATOR_FINGERPRINT is set to something other than "" or "0" (read once)
	static bool Enabled()
	{
		static const bool enabled = EnvOn("OPENEMS_OPERATOR_FINGERPRINT");
		return enabled;
	}

	static bool EnvOn(const char* name)
	{
		const char* val = getenv(name);
		return (val!=NULL) && (val[0]!='\0') && (strcmp(val,"0")!=0);
	}

	//! Hash \a bytes raw bytes
	void AddBytes(const void* data, size_t bytes)
	{
		const unsigned char* p = static_cast<const unsigned char*>(data);
		for (size_t i=0; i<bytes; ++i)
		{
			m_hash ^= p[i];
			m_hash *= 0x100000001b3ULL;
		}
	}

	//! Hash the bytes of a single value (a count, a flag, a double)
	template <typename T>
	void AddValue(const T& value)
	{
		AddBytes(&value, sizeof(T));
	}

	//! Hash an array of \a count elements: the count first (so that two arrays in a row cannot shift into each other), then the bytes. A NULL array counts as empty.
	template <typename T>
	void AddArray(const T* data, size_t count)
	{
		if (data==NULL)
			count = 0;
		AddValue((uint64_t)count);
		if (count)
			AddBytes(data, count*sizeof(T));
	}

	uint64_t Value() const {return m_hash;}

	//! "0x" and 16 lowercase hex digits
	std::string Hex() const
	{
		char buf[24];
		snprintf(buf, sizeof(buf), "0x%016llx", (unsigned long long)m_hash);
		return std::string(buf);
	}

private:
	uint64_t m_hash;
};

#endif // FINGERPRINT_H
