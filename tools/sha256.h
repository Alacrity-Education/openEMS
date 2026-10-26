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

#ifndef SHA256_H
#define SHA256_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

//! SHA-256 (FIPS 180-4) over raw bytes, used for the key of the operator cache (OPENEMS_OPERATOR_CACHE=<dir>)
class SHA256
{
public:
	SHA256() {Reset();}

	void Reset()
	{
		static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
		for (int i=0; i<8; ++i)
			m_state[i] = init[i];
		m_length = 0;
		m_fill = 0;
	}

	void Add(const void* data, size_t bytes)
	{
		const unsigned char* p = static_cast<const unsigned char*>(data);
		m_length += bytes;
		while (bytes>0)
		{
			size_t n = 64-m_fill;
			if (n>bytes)
				n = bytes;
			for (size_t i=0; i<n; ++i)
				m_block[m_fill+i] = p[i];
			m_fill += n;
			p += n;
			bytes -= n;
			if (m_fill==64)
			{
				Compress(m_block);
				m_fill = 0;
			}
		}
	}

	void Add(const std::string& str) {Add(str.data(), str.size());}

	//! Finish and return the digest as 64 lowercase hex digits; the object is reset afterwards
	std::string Hex()
	{
		uint64_t bits = m_length*8;
		unsigned char pad = 0x80;
		Add(&pad, 1);
		pad = 0;
		while (m_fill!=56)
			Add(&pad, 1);
		unsigned char len[8];
		for (int i=0; i<8; ++i)
			len[i] = (unsigned char)(bits >> (56-8*i));
		Add(len, 8);
		std::string hex;
		char buf[12];
		for (int i=0; i<8; ++i)
		{
			snprintf(buf, sizeof(buf), "%08x", (unsigned int)m_state[i]);
			hex += buf;
		}
		Reset();
		return hex;
	}

	static std::string Hex(const std::string& str)
	{
		SHA256 sha;
		sha.Add(str);
		return sha.Hex();
	}

private:
	static uint32_t Rot(uint32_t x, int n) {return (x >> n) | (x << (32-n));}

	void Compress(const unsigned char* block)
	{
		static const uint32_t k[64] = {
			0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
			0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
			0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
			0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
			0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
			0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
			0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
			0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
		uint32_t w[64];
		for (int i=0; i<16; ++i)
			w[i] = ((uint32_t)block[4*i] << 24) | ((uint32_t)block[4*i+1] << 16) | ((uint32_t)block[4*i+2] << 8) | (uint32_t)block[4*i+3];
		for (int i=16; i<64; ++i)
		{
			uint32_t s0 = Rot(w[i-15], 7) ^ Rot(w[i-15], 18) ^ (w[i-15] >> 3);
			uint32_t s1 = Rot(w[i-2], 17) ^ Rot(w[i-2], 19) ^ (w[i-2] >> 10);
			w[i] = w[i-16] + s0 + w[i-7] + s1;
		}
		uint32_t a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3];
		uint32_t e = m_state[4], f = m_state[5], g = m_state[6], h = m_state[7];
		for (int i=0; i<64; ++i)
		{
			uint32_t S1 = Rot(e, 6) ^ Rot(e, 11) ^ Rot(e, 25);
			uint32_t ch = (e & f) ^ (~e & g);
			uint32_t t1 = h + S1 + ch + k[i] + w[i];
			uint32_t S0 = Rot(a, 2) ^ Rot(a, 13) ^ Rot(a, 22);
			uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
			uint32_t t2 = S0 + maj;
			h = g;
			g = f;
			f = e;
			e = d + t1;
			d = c;
			c = b;
			b = a;
			a = t1 + t2;
		}
		m_state[0] += a; m_state[1] += b; m_state[2] += c; m_state[3] += d;
		m_state[4] += e; m_state[5] += f; m_state[6] += g; m_state[7] += h;
	}

	uint32_t m_state[8];
	uint64_t m_length;
	unsigned char m_block[64];
	size_t m_fill;
};

#endif // SHA256_H
