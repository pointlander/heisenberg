// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package heisenberg

import (
	"fmt"
	"math"
	"math/cmplx"
)

// Sparse64 is an algebriac matrix
type Sparse64 struct {
	R, C   int
	Matrix []map[int]complex64
}

func (a Sparse64) String() string {
	output := ""
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			out := a.Matrix[i]
			var value complex64
			if out != nil {
				value = out[j]
			}
			output += fmt.Sprintf("%f ", value)
		}
		output += fmt.Sprintf("\n")
	}
	return output
}

// Vector64 is a 64 bit vector
type Vector64 []complex64

func (a Vector64) String() string {
	output := ""
	for _, value := range a {
		output += fmt.Sprintf("%f ", value)
	}
	return output
}

// MachineSparse64 is a 64 bit sparse matrix machine
type MachineSparse64 struct {
	Vector64
	Qubits int
}

// Zero adds a zero to the matrix
func (a *MachineSparse64) Zero() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	zero := Vector64{1, 0}
	if qubit == 0 {
		a.Vector64 = zero
		return qubit
	}
	a.Vector64 = a.Tensor(zero)
	return qubit
}

// One adds a one to the matrix
func (a *MachineSparse64) One() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	one := Vector64{0, 1}
	if qubit == 0 {
		a.Vector64 = one
		return qubit
	}
	a.Vector64 = a.Tensor(one)
	return qubit
}

// Tensor product is the tensor product
func (a *Sparse64) Tensor(b *Sparse64) *Sparse64 {
	output := make([]map[int]complex64, a.R*b.R)
	for x, xx := range a.Matrix {
		for y, yy := range b.Matrix {
			for i, ii := range xx {
				for j, jj := range yy {
					values := output[x*b.R+y]
					if values == nil {
						values = make(map[int]complex64)
					}
					value := ii * jj
					if value != 0 {
						values[i*b.C+j] = value
					}
					if len(values) > 0 {
						output[x*b.R+y] = values
					}
				}
			}
		}
	}
	return &Sparse64{
		R:      a.R * b.R,
		C:      a.C * b.C,
		Matrix: output,
	}
}

// Multiply multiplies to matricies
func (a *Sparse64) Multiply(b *Sparse64) *Sparse64 {
	if a.C != b.R {
		panic("invalid dimensions")
	}
	output := make([]map[int]complex64, a.R)
	for j := 0; j < b.C; j++ {
		for x, xx := range a.Matrix {
			var sum complex64
			for y, value := range xx {
				yy := b.Matrix[y]
				var jj complex64
				if yy != nil {
					jj = yy[j]
				}
				sum += jj * value
			}
			values := output[x]
			if values == nil {
				values = make(map[int]complex64)
			}
			if sum != 0 {
				values[j] = sum
			}
			if len(values) > 0 {
				output[x] = values
			}
		}
	}
	return &Sparse64{
		R:      a.R,
		C:      b.C,
		Matrix: output,
	}
}

// Transpose transposes a matrix
func (a *Sparse64) Transpose() {
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			ii := a.Matrix[i]
			var value complex64
			if ii != nil {
				value = ii[j]
			}
			a.Matrix[j][i] = value
		}
	}
	a.R, a.C = a.C, a.R
}

// Copy copies a matrix`
func (a *Sparse64) Copy() *Sparse64 {
	cp := &Sparse64{
		R:      a.R,
		C:      a.C,
		Matrix: make([]map[int]complex64, a.R),
	}
	for a, aa := range a.Matrix {
		value := cp.Matrix[a]
		if value == nil {
			value = make(map[int]complex64)
		}
		for b, bb := range aa {
			value[b] = bb
		}
		cp.Matrix[a] = value
	}
	return cp
}

// Tensor product is the tensor product
func (a Vector64) Tensor(b Vector64) Vector64 {
	output := make(Vector64, 0, len(a)*len(b))
	for _, ii := range a {
		for _, jj := range b {
			output = append(output, ii*jj)
		}
	}

	return output
}

// MultiplyVector multiplies a matrix by a vector
func (a *Sparse64) MultiplyVector(b Vector64) Vector64 {
	if a.C != len(b) {
		panic(fmt.Sprintf("invalid dimensions %d %d", a.C, len(b)))
	}
	output := make(Vector64, 0, a.R)
	for _, xx := range a.Matrix {
		var sum complex64
		for y, value := range xx {
			sum += b[y] * value
		}
		output = append(output, sum)
	}
	return output
}

// ControlledNot controlled not gate
func (a *MachineSparse64) ControlledNot(c []Qubit, t Qubit) *Sparse64 {
	n := a.Qubits
	p := &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
			},
			map[int]complex64{
				1: 1,
			},
		},
	}
	q := p
	for i := 0; i < n-1; i++ {
		q = p.Tensor(q)
	}
	d := q.R

	index := make([]int64, 0)
	for i := 0; i < d; i++ {
		bits := int64(i)

		// Apply X
		apply := true
		for _, j := range c {
			if (bits>>(Qubit(n-1)-j))&1 == 0 {
				apply = false
				break
			}
		}

		if apply {
			if (bits>>(Qubit(n-1)-t))&1 == 0 {
				bits |= 1 << (Qubit(n-1) - t)
			} else {
				bits &= ^(1 << (Qubit(n-1) - t))
			}
		}

		index = append(index, bits)
	}

	g := Sparse64{
		R:      q.R,
		C:      q.C,
		Matrix: make([]map[int]complex64, q.R),
	}
	for i, ii := range index {
		g.Matrix[i] = q.Matrix[int(ii)]
	}

	a.Vector64 = g.MultiplyVector(a.Vector64)

	return &g
}

// Multiply multiplies the machine by a matrix
func (a *MachineSparse64) Multiply(b *Sparse64, qubits ...Qubit) {
	indexes := make(map[int]bool)
	for _, value := range qubits {
		indexes[int(value)] = true
	}

	identity := ISparse64()
	d := ISparse64()
	if indexes[0] {
		d = b.Copy()
	}
	for i := 1; i < a.Qubits; i++ {
		if indexes[i] {
			d = d.Tensor(b)
			continue
		}

		d = d.Tensor(identity)
	}

	a.Vector64 = d.MultiplyVector(a.Vector64)
}

// ISparse64 identity matrix
func ISparse64() *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
			},
			map[int]complex64{
				1: 1,
			},
		},
	}
}

// I multiply by identity
func (a *MachineSparse64) I(qubits ...Qubit) *MachineSparse64 {
	a.Multiply(ISparse64(), qubits...)
	return a
}

// HSparse64 Hadamard matrix
func HSparse64() *Sparse64 {
	v := complex(1/math.Sqrt2, 0)
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: complex64(v),
				1: complex64(v),
			},
			map[int]complex64{
				0: complex64(v),
				1: complex64(-v),
			},
		},
	}
}

// H multiply by Hadamard gate
func (a *MachineSparse64) H(qubits ...Qubit) *MachineSparse64 {
	a.Multiply(HSparse64(), qubits...)
	return a
}

// XSparse64 Pauli X matrix
func XSparse64() *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				1: 1,
			},
			map[int]complex64{
				0: 1,
			},
		},
	}
}

// X multiply by Pauli X matrix
func (a *MachineSparse64) X(qubits ...Qubit) *MachineSparse64 {
	a.Multiply(XSparse64(), qubits...)
	return a
}

// YSparse64 Pauli Y matrix
func YSparse64() *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				1: -1i,
			},
			map[int]complex64{
				0: 1i,
			},
		},
	}
}

// Y multiply by Pauli Y matrix
func (a *MachineSparse64) Y(qubits ...Qubit) *MachineSparse64 {
	a.Multiply(YSparse64(), qubits...)
	return a
}

// ZSparse64 Pauli Z matrix
func ZSparse64() *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
			},
			map[int]complex64{
				1: -1,
			},
		},
	}
}

// Z multiply by Pauli Z matrix
func (a *MachineSparse64) Z(qubits ...Qubit) *MachineSparse64 {
	a.Multiply(ZSparse64(), qubits...)
	return a
}

// SSparse64 phase gate
func SSparse64() *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
			},
			map[int]complex64{
				1: 1i,
			},
		},
	}
}

// S multiply by phase matrix
func (a *MachineSparse64) S(qubits ...Qubit) *MachineSparse64 {
	a.Multiply(SSparse64(), qubits...)
	return a
}

// TSparse64 T gate
func TSparse64() *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
			},
			map[int]complex64{
				1: complex64(cmplx.Exp(1i * math.Pi / 4)),
			},
		},
	}
}

// T multiply by T matrix
func (a *MachineSparse64) T(qubits ...Qubit) *MachineSparse64 {
	a.Multiply(TSparse64(), qubits...)
	return a
}

// USparse64 U gate
func USparse64(theta, phi, lambda float64) *Sparse64 {
	v := complex(theta/2, 0)
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: complex64(cmplx.Cos(v)),
				1: complex64(-1 * cmplx.Exp(complex(0, lambda)) * cmplx.Sin(v)),
			},
			map[int]complex64{
				0: complex64(cmplx.Exp(complex(0, phi)) * cmplx.Sin(v)),
				1: complex64(cmplx.Exp(complex(0, (phi+lambda))) * cmplx.Cos(v)),
			},
		},
	}
}

// U multiply by U matrix
func (a *MachineSparse64) U(theta, phi, lambda float64, qubits ...Qubit) *MachineSparse64 {
	a.Multiply(USparse64(theta, phi, lambda), qubits...)
	return a
}

// RXSparse64 x rotation matrix
func RXSparse64(theta complex128) *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: complex64(cmplx.Cos(complex128(theta))),
				1: -1i * complex64(cmplx.Sin(complex128(theta))),
			},
			map[int]complex64{
				0: -1i * complex64(cmplx.Sin(complex128(theta))),
				1: complex64(cmplx.Cos(complex128(theta))),
			},
		},
	}
}

// RX rotate X gate
func (a *MachineSparse64) RX(theta float64, qubits ...Qubit) *MachineSparse64 {
	a.Multiply(RXSparse64(complex(theta/2, 0)), qubits...)
	return a
}

// RYSparse64 y rotation matrix
func RYSparse64(theta complex128) *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: complex64(cmplx.Cos(complex128(theta))),
				1: -1 * complex64(cmplx.Sin(complex128(theta))),
			},
			map[int]complex64{
				0: complex64(cmplx.Sin(complex128(theta))),
				1: complex64(cmplx.Cos(complex128(theta))),
			},
		},
	}
}

// RY rotate Y gate
func (a *MachineSparse64) RY(theta float64, qubits ...Qubit) *MachineSparse64 {
	a.Multiply(RYSparse64(complex(theta/2, 0)), qubits...)
	return a
}

// RZSparse64 z rotation matrix
func RZSparse64(theta complex128) *Sparse64 {
	return &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: complex64(cmplx.Exp(-1 * complex128(theta))),
			},
			map[int]complex64{
				1: complex64(cmplx.Exp(complex128(theta))),
			},
		},
	}
}

// RZ rotate Z gate
func (a *MachineSparse64) RZ(theta float64, qubits ...Qubit) *MachineSparse64 {
	a.Multiply(RZSparse64(complex(theta/2, 0)), qubits...)
	return a
}

// Swap swaps qubits`
func (a *MachineSparse64) Swap(qubits ...Qubit) *MachineSparse64 {
	length := len(qubits)

	for i := 0; i < length/2; i++ {
		c, t := qubits[i], qubits[(length-1)-i]
		a.ControlledNot([]Qubit{c}, t)
		a.ControlledNot([]Qubit{t}, c)
		a.ControlledNot([]Qubit{c}, t)
	}

	return a
}

// Sparse128 is an algebriac matrix
type Sparse128 struct {
	R, C   int
	Matrix []map[int]complex128
}

func (a Sparse128) String() string {
	output := ""
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			out := a.Matrix[i]
			var value complex128
			if out != nil {
				value = out[j]
			}
			output += fmt.Sprintf("%f ", value)
		}
		output += fmt.Sprintf("\n")
	}
	return output
}

// MachineSparse128 is a 128 bit sparse matrix machine
type MachineSparse128 struct {
	Vector128
	Qubits int
}

// Zero adds a zero to the matrix
func (a *MachineSparse128) Zero() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	zero := Vector128{1, 0}
	if qubit == 0 {
		a.Vector128 = zero
		return qubit
	}
	a.Vector128 = a.Tensor(zero)
	return qubit
}

// One adds a one to the matrix
func (a *MachineSparse128) One() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	one := Vector128{0, 1}
	if qubit == 0 {
		a.Vector128 = one
		return qubit
	}
	a.Vector128 = a.Tensor(one)
	return qubit
}

// Tensor product is the tensor product
func (a *Sparse128) Tensor(b *Sparse128) *Sparse128 {
	output := make([]map[int]complex128, a.R*b.R)
	for x, xx := range a.Matrix {
		for y, yy := range b.Matrix {
			for i, ii := range xx {
				for j, jj := range yy {
					values := output[x*b.R+y]
					if values == nil {
						values = make(map[int]complex128)
					}
					value := ii * jj
					if value != 0 {
						values[i*b.C+j] = value
					}
					if len(values) > 0 {
						output[x*b.R+y] = values
					}
				}
			}
		}
	}
	return &Sparse128{
		R:      a.R * b.R,
		C:      a.C * b.C,
		Matrix: output,
	}
}

// Multiply multiplies to matricies
func (a *Sparse128) Multiply(b *Sparse128) *Sparse128 {
	if a.C != b.R {
		panic("invalid dimensions")
	}
	output := make([]map[int]complex128, a.R)
	for j := 0; j < b.C; j++ {
		for x, xx := range a.Matrix {
			var sum complex128
			for y, value := range xx {
				yy := b.Matrix[y]
				var jj complex128
				if yy != nil {
					jj = yy[j]
				}
				sum += jj * value
			}
			values := output[x]
			if values == nil {
				values = make(map[int]complex128)
			}
			if sum != 0 {
				values[j] = sum
			}
			if len(values) > 0 {
				output[x] = values
			}
		}
	}
	return &Sparse128{
		R:      a.R,
		C:      b.C,
		Matrix: output,
	}
}

// Transpose transposes a matrix
func (a *Sparse128) Transpose() {
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			ii := a.Matrix[i]
			var value complex128
			if ii != nil {
				value = ii[j]
			}
			a.Matrix[j][i] = value
		}
	}
	a.R, a.C = a.C, a.R
}

// Copy copies a matrix`
func (a *Sparse128) Copy() *Sparse128 {
	cp := &Sparse128{
		R:      a.R,
		C:      a.C,
		Matrix: make([]map[int]complex128, a.R),
	}
	for a, aa := range a.Matrix {
		value := cp.Matrix[a]
		if value == nil {
			value = make(map[int]complex128)
		}
		for b, bb := range aa {
			value[b] = bb
		}
		cp.Matrix[a] = value
	}
	return cp
}

// MultiplyVector multiplies a matrix by a vector
func (a *Sparse128) MultiplyVector(b Vector128) Vector128 {
	if a.C != len(b) {
		panic(fmt.Sprintf("invalid dimensions %d %d", a.C, len(b)))
	}
	output := make(Vector128, 0, a.R)
	for _, xx := range a.Matrix {
		var sum complex128
		for y, value := range xx {
			sum += b[y] * value
		}
		output = append(output, sum)
	}
	return output
}

// ControlledNot controlled not gate
func (a *MachineSparse128) ControlledNot(c []Qubit, t Qubit) *Sparse128 {
	n := a.Qubits
	p := &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: 1,
			},
		},
	}
	q := p
	for i := 0; i < n-1; i++ {
		q = p.Tensor(q)
	}
	d := q.R

	index := make([]int64, 0)
	for i := 0; i < d; i++ {
		bits := int64(i)

		// Apply X
		apply := true
		for _, j := range c {
			if (bits>>(Qubit(n-1)-j))&1 == 0 {
				apply = false
				break
			}
		}

		if apply {
			if (bits>>(Qubit(n-1)-t))&1 == 0 {
				bits |= 1 << (Qubit(n-1) - t)
			} else {
				bits &= ^(1 << (Qubit(n-1) - t))
			}
		}

		index = append(index, bits)
	}

	g := Sparse128{
		R:      q.R,
		C:      q.C,
		Matrix: make([]map[int]complex128, q.R),
	}
	for i, ii := range index {
		g.Matrix[i] = q.Matrix[int(ii)]
	}

	a.Vector128 = g.MultiplyVector(a.Vector128)

	return &g
}

// Multiply multiplies the machine by a matrix
func (a *MachineSparse128) Multiply(b *Sparse128, qubits ...Qubit) {
	indexes := make(map[int]bool)
	for _, value := range qubits {
		indexes[int(value)] = true
	}

	identity := ISparse128()
	d := ISparse128()
	if indexes[0] {
		d = b.Copy()
	}
	for i := 1; i < a.Qubits; i++ {
		if indexes[i] {
			d = d.Tensor(b)
			continue
		}

		d = d.Tensor(identity)
	}

	a.Vector128 = d.MultiplyVector(a.Vector128)
}

// ISparse128 identity matrix
func ISparse128() *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: 1,
			},
		},
	}
}

// I multiply by identity
func (a *MachineSparse128) I(qubits ...Qubit) *MachineSparse128 {
	a.Multiply(ISparse128(), qubits...)
	return a
}

// HSparse128 Hadamard matrix
func HSparse128() *Sparse128 {
	v := complex(1/math.Sqrt2, 0)
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: v,
				1: v,
			},
			map[int]complex128{
				0: v,
				1: -v,
			},
		},
	}
}

// H multiply by Hadamard gate
func (a *MachineSparse128) H(qubits ...Qubit) *MachineSparse128 {
	a.Multiply(HSparse128(), qubits...)
	return a
}

// XSparse128 Pauli X matrix
func XSparse128() *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				1: 1,
			},
			map[int]complex128{
				0: 1,
			},
		},
	}
}

// X multiply by Pauli X matrix
func (a *MachineSparse128) X(qubits ...Qubit) *MachineSparse128 {
	a.Multiply(XSparse128(), qubits...)
	return a
}

// YSparse128 Pauli Y matrix
func YSparse128() *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				1: -1i,
			},
			map[int]complex128{
				0: 1i,
			},
		},
	}
}

// Y multiply by Pauli Y matrix
func (a *MachineSparse128) Y(qubits ...Qubit) *MachineSparse128 {
	a.Multiply(YSparse128(), qubits...)
	return a
}

// ZSparse128 Pauli Z matrix
func ZSparse128() *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: -1,
			},
		},
	}
}

// Z multiply by Pauli Z matrix
func (a *MachineSparse128) Z(qubits ...Qubit) *MachineSparse128 {
	a.Multiply(ZSparse128(), qubits...)
	return a
}

// SSparse128 phase gate
func SSparse128() *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: 1i,
			},
		},
	}
}

// S multiply by phase matrix
func (a *MachineSparse128) S(qubits ...Qubit) *MachineSparse128 {
	a.Multiply(SSparse128(), qubits...)
	return a
}

// TSparse128 T gate
func TSparse128() *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: cmplx.Exp(1i * math.Pi / 4),
			},
		},
	}
}

// T multiply by T matrix
func (a *MachineSparse128) T(qubits ...Qubit) *MachineSparse128 {
	a.Multiply(TSparse128(), qubits...)
	return a
}

// USparse128 U gate
func USparse128(theta, phi, lambda float64) *Sparse128 {
	v := complex(theta/2, 0)
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: cmplx.Cos(v),
				1: -1 * cmplx.Exp(complex(0, lambda)) * cmplx.Sin(v),
			},
			map[int]complex128{
				0: cmplx.Exp(complex(0, phi)) * cmplx.Sin(v),
				1: cmplx.Exp(complex(0, (phi+lambda))) * cmplx.Cos(v),
			},
		},
	}
}

// U multiply by U matrix
func (a *MachineSparse128) U(theta, phi, lambda float64, qubits ...Qubit) *MachineSparse128 {
	a.Multiply(USparse128(theta, phi, lambda), qubits...)
	return a
}

// RXSparse128 x rotation matrix
func RXSparse128(theta complex128) *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: cmplx.Cos(theta),
				1: -1i * cmplx.Sin(theta),
			},
			map[int]complex128{
				0: -1i * cmplx.Sin(theta),
				1: cmplx.Cos(theta),
			},
		},
	}
}

// RX rotate X gate
func (a *MachineSparse128) RX(theta float64, qubits ...Qubit) *MachineSparse128 {
	a.Multiply(RXSparse128(complex(theta/2, 0)), qubits...)
	return a
}

// RYSparse128 y rotation matrix
func RYSparse128(theta complex128) *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: cmplx.Cos(theta),
				1: -1 * cmplx.Sin(theta),
			},
			map[int]complex128{
				0: cmplx.Sin(theta),
				1: cmplx.Cos(theta),
			},
		},
	}
}

// RY rotate Y gate
func (a *MachineSparse128) RY(theta float64, qubits ...Qubit) *MachineSparse128 {
	a.Multiply(RYSparse128(complex(theta/2, 0)), qubits...)
	return a
}

// RZSparse128 z rotation matrix
func RZSparse128(theta complex128) *Sparse128 {
	return &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: cmplx.Exp(-1 * theta),
			},
			map[int]complex128{
				1: cmplx.Exp(theta),
			},
		},
	}
}

// RZ rotate Z gate
func (a *MachineSparse128) RZ(theta float64, qubits ...Qubit) *MachineSparse128 {
	a.Multiply(RZSparse128(complex(theta/2, 0)), qubits...)
	return a
}

// Swap swaps qubits`
func (a *MachineSparse128) Swap(qubits ...Qubit) *MachineSparse128 {
	length := len(qubits)

	for i := 0; i < length/2; i++ {
		c, t := qubits[i], qubits[(length-1)-i]
		a.ControlledNot([]Qubit{c}, t)
		a.ControlledNot([]Qubit{t}, c)
		a.ControlledNot([]Qubit{c}, t)
	}

	return a
}
