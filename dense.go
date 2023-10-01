// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package heisenberg

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
)

// Dense64 is an algebriac matrix
type Dense64 struct {
	R, C   int
	Matrix []complex64
}

func (a Dense64) String() string {
	output := ""
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			output += fmt.Sprintf("%f ", a.Matrix[i*a.C+j])
		}
		output += fmt.Sprintf("\n")
	}
	return output
}

// MachineDense64 is a 64 bit dense matrix machine
type MachineDense64 struct {
	Dense64
	Qubits int
}

// Zero adds a zero to the matrix
func (a *MachineDense64) Zero() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	zero := Dense64{
		R: 2,
		C: 1,
		Matrix: []complex64{
			1, 0,
		},
	}
	if qubit == 0 {
		a.Dense64 = zero
		return qubit
	}
	a.Dense64 = *a.Tensor(&zero)
	return qubit
}

// One adds a one to the matrix
func (a *MachineDense64) One() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	one := Dense64{
		R: 2,
		C: 1,
		Matrix: []complex64{
			0, 1,
		},
	}
	if qubit == 0 {
		a.Dense64 = one
		return qubit
	}
	a.Dense64 = *a.Tensor(&one)
	return qubit
}

// Tensor product is the tensor product
func (a *Dense64) Tensor(b *Dense64) *Dense64 {
	output := make([]complex64, 0, len(a.Matrix)*len(b.Matrix))
	for x := 0; x < a.R; x++ {
		for y := 0; y < b.R; y++ {
			for i := 0; i < a.C; i++ {
				for j := 0; j < b.C; j++ {
					output = append(output, a.Matrix[x*a.C+i]*b.Matrix[y*b.C+j])
				}
			}
		}
	}
	return &Dense64{
		R:      a.R * b.R,
		C:      a.C * b.C,
		Matrix: output,
	}
}

// Multiply multiplies to matricies
func (a *Dense64) Multiply(b *Dense64) *Dense64 {
	if a.C != b.R {
		panic("invalid dimensions")
	}
	output := make([]complex64, 0, a.R*b.C)
	for j := 0; j < b.C; j++ {
		for x := 0; x < a.R; x++ {
			var sum complex64
			for y := 0; y < b.R; y++ {
				sum += b.Matrix[y*b.C+j] * a.Matrix[x*a.C+y]
			}
			output = append(output, sum)
		}
	}
	return &Dense64{
		R:      a.R,
		C:      b.C,
		Matrix: output,
	}
}

// Transpose transposes a matrix
func (a *Dense64) Transpose() {
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			a.Matrix[j*a.R+i] = a.Matrix[i*a.C+j]
		}
	}
	a.R, a.C = a.C, a.R
}

// Copy copies a matrix`
func (a *Dense64) Copy() *Dense64 {
	cp := &Dense64{
		R:      a.R,
		C:      a.C,
		Matrix: make([]complex64, len(a.Matrix)),
	}
	copy(cp.Matrix, a.Matrix)
	return cp
}

// ControlledNot controlled not gate
func (a *MachineDense64) ControlledNot(c []Qubit, t Qubit) *Dense64 {
	n := a.Qubits
	p := &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, 1,
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

	g := Dense64{
		R:      q.R,
		C:      q.C,
		Matrix: make([]complex64, q.R*q.C),
	}
	for i, ii := range index {
		copy(g.Matrix[i*g.C:(i+1)*g.C], q.Matrix[int(ii)*g.C:int(ii+1)*g.C])
	}

	a.Dense64 = *g.Multiply(&a.Dense64)

	return &g
}

// Multiply multiplies the machine by a matrix
func (a *MachineDense64) Multiply(b *Dense64, qubits ...Qubit) {
	indexes := make(map[int]bool)
	for _, value := range qubits {
		indexes[int(value)] = true
	}

	identity := IDense64()
	d := IDense64()
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

	a.Dense64 = *d.Multiply(&a.Dense64)
}

// IDense64 identity matrix
func IDense64() *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, 1,
		},
	}
}

// I multiply by identity
func (a *MachineDense64) I(qubits ...Qubit) *MachineDense64 {
	a.Multiply(IDense64(), qubits...)
	return a
}

// HDense64 Hadamard matrix
func HDense64() *Dense64 {
	v := complex(1/math.Sqrt2, 0)
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			complex64(v), complex64(v),
			complex64(v), complex64(-v),
		},
	}
}

// H multiply by Hadamard gate
func (a *MachineDense64) H(qubits ...Qubit) *MachineDense64 {
	a.Multiply(HDense64(), qubits...)
	return a
}

// XDense64 Pauli X matrix
func XDense64() *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			0, 1,
			1, 0,
		},
	}
}

// X multiply by Pauli X matrix
func (a *MachineDense64) X(qubits ...Qubit) *MachineDense64 {
	a.Multiply(XDense64(), qubits...)
	return a
}

// YDense64 Pauli Y matrix
func YDense64() *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			0, -1i,
			1i, 0,
		},
	}
}

// Y multiply by Pauli Y matrix
func (a *MachineDense64) Y(qubits ...Qubit) *MachineDense64 {
	a.Multiply(YDense64(), qubits...)
	return a
}

// ZDense64 Pauli Z matrix
func ZDense64() *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, -1,
		},
	}
}

// Z multiply by Pauli Z matrix
func (a *MachineDense64) Z(qubits ...Qubit) *MachineDense64 {
	a.Multiply(ZDense64(), qubits...)
	return a
}

// SDense64 phase gate
func SDense64() *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, 1i,
		},
	}
}

// S multiply by phase matrix
func (a *MachineDense64) S(qubits ...Qubit) *MachineDense64 {
	a.Multiply(SDense64(), qubits...)
	return a
}

// TDense64 T gate
func TDense64() *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, complex64(cmplx.Exp(1i * math.Pi / 4)),
		},
	}
}

// T multiply by T matrix
func (a *MachineDense64) T(qubits ...Qubit) *MachineDense64 {
	a.Multiply(TDense64(), qubits...)
	return a
}

// UDense64 U gate
func UDense64(theta, phi, lambda float64) *Dense64 {
	v := complex(theta/2, 0)
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			complex64(cmplx.Cos(v)), complex64(-1 * cmplx.Exp(complex(0, lambda)) * cmplx.Sin(v)),
			complex64(cmplx.Exp(complex(0, phi)) * cmplx.Sin(v)), complex64(cmplx.Exp(complex(0, (phi+lambda))) * cmplx.Cos(v)),
		},
	}
}

// U multiply by U matrix
func (a *MachineDense64) U(theta, phi, lambda float64, qubits ...Qubit) *MachineDense64 {
	a.Multiply(UDense64(theta, phi, lambda), qubits...)
	return a
}

// RXDense64 x rotation matrix
func RXDense64(theta complex128) *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			complex64(cmplx.Cos(complex128(theta))), -1i * complex64(cmplx.Sin(complex128(theta))),
			-1i * complex64(cmplx.Sin(complex128(theta))), complex64(cmplx.Cos(complex128(theta))),
		},
	}
}

// RX rotate X gate
func (a *MachineDense64) RX(theta float64, qubits ...Qubit) *MachineDense64 {
	a.Multiply(RXDense64(complex(theta/2, 0)), qubits...)
	return a
}

// RYDense64 y rotation matrix
func RYDense64(theta complex128) *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			complex64(cmplx.Cos(complex128(theta))), -1 * complex64(cmplx.Sin(complex128(theta))),
			complex64(cmplx.Sin(complex128(theta))), complex64(cmplx.Cos(complex128(theta))),
		},
	}
}

// RY rotate Y gate
func (a *MachineDense64) RY(theta float64, qubits ...Qubit) *MachineDense64 {
	a.Multiply(RYDense64(complex(theta/2, 0)), qubits...)
	return a
}

// RZDense64 z rotation matrix
func RZDense64(theta complex128) *Dense64 {
	return &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			complex64(cmplx.Exp(-1 * complex128(theta))), 0,
			0, complex64(cmplx.Exp(complex128(theta))),
		},
	}
}

// RZ rotate Z gate
func (a *MachineDense64) RZ(theta float64, qubits ...Qubit) *MachineDense64 {
	a.Multiply(RZDense64(complex(theta/2, 0)), qubits...)
	return a
}

// Swap swaps qubits`
func (a *MachineDense64) Swap(qubits ...Qubit) *MachineDense64 {
	length := len(qubits)

	for i := 0; i < length/2; i++ {
		c, t := qubits[i], qubits[(length-1)-i]
		a.ControlledNot([]Qubit{c}, t)
		a.ControlledNot([]Qubit{t}, c)
		a.ControlledNot([]Qubit{c}, t)
	}

	return a
}

// Dense128 is an algebriac matrix
type Dense128 struct {
	R, C   int
	Matrix []complex128
}

func (a Dense128) String() string {
	output := ""
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			output += fmt.Sprintf("%f ", a.Matrix[i*a.C+j])
		}
		output += fmt.Sprintf("\n")
	}
	return output
}

// MachineDense128 is a 128 bit dense matrix machine
type MachineDense128 struct {
	Dense128
	Qubits int
}

// Zero adds a zero to the matrix
func (a *MachineDense128) Zero() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	zero := Dense128{
		R: 2,
		C: 1,
		Matrix: []complex128{
			1, 0,
		},
	}
	if qubit == 0 {
		a.Dense128 = zero
		return qubit
	}
	a.Dense128 = *a.Tensor(&zero)
	return qubit
}

// One adds a one to the matrix
func (a *MachineDense128) One() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	one := Dense128{
		R: 2,
		C: 1,
		Matrix: []complex128{
			0, 1,
		},
	}
	if qubit == 0 {
		a.Dense128 = one
		return qubit
	}
	a.Dense128 = *a.Tensor(&one)
	return qubit
}

// Tensor product is the tensor product
func (a *Dense128) Tensor(b *Dense128) *Dense128 {
	output := make([]complex128, 0, len(a.Matrix)*len(b.Matrix))
	for x := 0; x < a.R; x++ {
		for y := 0; y < b.R; y++ {
			for i := 0; i < a.C; i++ {
				for j := 0; j < b.C; j++ {
					output = append(output, a.Matrix[x*a.C+i]*b.Matrix[y*b.C+j])
				}
			}
		}
	}
	return &Dense128{
		R:      a.R * b.R,
		C:      a.C * b.C,
		Matrix: output,
	}
}

// Multiply multiplies to matricies
func (a *Dense128) Multiply(b *Dense128) *Dense128 {
	if a.C != b.R {
		panic("invalid dimensions")
	}
	output := make([]complex128, 0, a.R*b.C)
	for j := 0; j < b.C; j++ {
		for x := 0; x < a.R; x++ {
			var sum complex128
			for y := 0; y < b.R; y++ {
				sum += b.Matrix[y*b.C+j] * a.Matrix[x*a.C+y]
			}
			output = append(output, sum)
		}
	}
	return &Dense128{
		R:      a.R,
		C:      b.C,
		Matrix: output,
	}
}

// Transpose transposes a matrix
func (a *Dense128) Transpose() {
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			a.Matrix[j*a.R+i] = a.Matrix[i*a.C+j]
		}
	}
	a.R, a.C = a.C, a.R
}

// Conjugate computes the complex conjugate
func (a *Dense128) Conjugate() {
	for k, v := range a.Matrix {
		a.Matrix[k] = cmplx.Conj(v)
	}
}

// Sub subtracts two matrices
func (a *Dense128) Sub(b *Dense128) {
	for k := range a.Matrix {
		a.Matrix[k] -= b.Matrix[k]
	}
}

// ComplexMul multiplies two complex matrices
func (a *Dense128) Mul(b *Dense128) {
	cp := a.Copy()
	for i := 0; i < b.R; i++ {
		for j := 0; j < b.C; j++ {
			var sum complex128
			for k := 0; k < a.C; k++ {
				sum += cp.Matrix[i*a.C+k] * b.Matrix[k*b.C+j]
			}
			a.Matrix[i*a.C+j] = sum
		}
	}
}

// Copy copies a matrix`
func (a *Dense128) Copy() *Dense128 {
	cp := &Dense128{
		R:      a.R,
		C:      a.C,
		Matrix: make([]complex128, len(a.Matrix)),
	}
	copy(cp.Matrix, a.Matrix)
	return cp
}

// ControlledNot controlled not gate
func (a *MachineDense128) ControlledNot(c []Qubit, t Qubit) *Dense128 {
	n := a.Qubits
	p := &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			1, 0,
			0, 1,
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

	g := Dense128{
		R:      q.R,
		C:      q.C,
		Matrix: make([]complex128, q.R*q.C),
	}
	for i, ii := range index {
		copy(g.Matrix[i*g.C:(i+1)*g.C], q.Matrix[int(ii)*g.C:int(ii+1)*g.C])
	}

	a.Dense128 = *g.Multiply(&a.Dense128)

	return &g
}

// Multiply multiplies the machine by a matrix
func (a *MachineDense128) Multiply(b *Dense128, c ...Qubit) {
	indexes := make(map[int]bool)
	for _, value := range c {
		indexes[int(value)] = true
	}

	identity := IDense128()
	d := IDense128()
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

	a.Dense128 = *d.Multiply(&a.Dense128)
}

// IDense128 identity matrix
func IDense128() *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			1, 0,
			0, 1,
		},
	}
}

// I multiply by identity
func (a *MachineDense128) I(qubits ...Qubit) *MachineDense128 {
	a.Multiply(IDense128(), qubits...)
	return a
}

// HDense128 Hadamard matrix
func HDense128() *Dense128 {
	v := complex(1/math.Sqrt2, 0)
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			v, v,
			v, -v,
		},
	}
}

// H multiply by Hadamard gate
func (a *MachineDense128) H(qubits ...Qubit) *MachineDense128 {
	a.Multiply(HDense128(), qubits...)
	return a
}

// XDense128 Pauli X matrix
func XDense128() *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			0, 1,
			1, 0,
		},
	}
}

// X multiply by Pauli X matrix
func (a *MachineDense128) X(qubits ...Qubit) *MachineDense128 {
	a.Multiply(XDense128(), qubits...)
	return a
}

// YDense128 Pauli Y matrix
func YDense128() *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			0, -1i,
			1i, 0,
		},
	}
}

// Y multiply by Pauli Y matrix
func (a *MachineDense128) Y(qubits ...Qubit) *MachineDense128 {
	a.Multiply(YDense128(), qubits...)
	return a
}

// ZDense128 Pauli Z matrix
func ZDense128() *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			1, 0,
			0, -1,
		},
	}
}

// Z multiply by Pauli Z matrix
func (a *MachineDense128) Z(qubits ...Qubit) *MachineDense128 {
	a.Multiply(ZDense128(), qubits...)
	return a
}

// SDense128 phase gate
func SDense128() *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			1, 0,
			0, 1i,
		},
	}
}

// S multiply by phase matrix
func (a *MachineDense128) S(qubits ...Qubit) *MachineDense128 {
	a.Multiply(SDense128(), qubits...)
	return a
}

// TDense128 T gate
func TDense128() *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			1, 0,
			0, cmplx.Exp(1i * math.Pi / 4),
		},
	}
}

// T multiply by T matrix
func (a *MachineDense128) T(qubits ...Qubit) *MachineDense128 {
	a.Multiply(TDense128(), qubits...)
	return a
}

// UDense128 U gate
func UDense128(theta, phi, lambda float64) *Dense128 {
	v := complex(theta/2, 0)
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			cmplx.Cos(v), -1 * cmplx.Exp(complex(0, lambda)) * cmplx.Sin(v),
			cmplx.Exp(complex(0, phi)) * cmplx.Sin(v), cmplx.Exp(complex(0, (phi+lambda))) * cmplx.Cos(v),
		},
	}
}

// U multiply by U matrix
func (a *MachineDense128) U(theta, phi, lambda float64, qubits ...Qubit) *MachineDense128 {
	a.Multiply(UDense128(theta, phi, lambda), qubits...)
	return a
}

// RXDense128 x rotation matrix
func RXDense128(theta complex128) *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			cmplx.Cos(theta), -1i * cmplx.Sin(theta),
			-1i * cmplx.Sin(theta), cmplx.Cos(theta),
		},
	}
}

// RX rotate X gate
func (a *MachineDense128) RX(theta float64, qubits ...Qubit) *MachineDense128 {
	a.Multiply(RXDense128(complex(theta/2, 0)), qubits...)
	return a
}

// RYDense128 y rotation matrix
func RYDense128(theta complex128) *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			cmplx.Cos(theta), -1 * cmplx.Sin(theta),
			cmplx.Sin(theta), cmplx.Cos(theta),
		},
	}
}

// RY rotate Y gate
func (a *MachineDense128) RY(theta float64, qubits ...Qubit) *MachineDense128 {
	a.Multiply(RYDense128(complex(theta/2, 0)), qubits...)
	return a
}

// RZDense128 z rotation matrix
func RZDense128(theta complex128) *Dense128 {
	return &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			cmplx.Exp(-1 * theta), 0,
			0, cmplx.Exp(theta),
		},
	}
}

// RZ rotate Z gate
func (a *MachineDense128) RZ(theta float64, qubits ...Qubit) *MachineDense128 {
	a.Multiply(RZDense128(complex(theta/2, 0)), qubits...)
	return a
}

// Swap swaps qubits`
func (a *MachineDense128) Swap(qubits ...Qubit) *MachineDense128 {
	length := len(qubits)

	for i := 0; i < length/2; i++ {
		c, t := qubits[i], qubits[(length-1)-i]
		a.ControlledNot([]Qubit{c}, t)
		a.ControlledNot([]Qubit{t}, c)
		a.ControlledNot([]Qubit{c}, t)
	}

	return a
}

type Point struct {
	X, Y float64
}

// Points returns the point representation of the quantum algorithm.
func (d *MachineDense128) Points() []Point {
	rng := rand.New(rand.NewSource(1))
	type Genome struct {
		Points  []Point
		Fitness float64
	}
	cc := HDense128()
	fitness := func(g Genome) float64 {
		a := make([][]complex128, len(g.Points))
		for i := 0; i < len(g.Points); i++ {
			for j := 0; j < len(g.Points); j++ {
				a[i] = append(a[i], complex(g.Points[j].X-g.Points[i].X, g.Points[j].Y-g.Points[i].Y))
			}
		}
		aa := Dense128{
			R:      len(a),
			C:      len(a[0]),
			Matrix: make([]complex128, 0, len(a)*len(a[0])),
		}
		for i := 0; i < len(a); i++ {
			for j := 0; j < len(a[i]); j++ {
				aa.Matrix = append(aa.Matrix, a[i][j])
			}
		}
		bb := aa.Copy()
		aa.Transpose()
		aa.Conjugate()
		bb.Mul(&aa)
		bb.Sub(cc)
		sum := 0.0
		for _, v := range bb.Matrix {
			sum += cmplx.Abs(v)
		}
		return sum
	}
	pop := make([]Genome, 0, 1024)
	for i := 0; i < 1024; i++ {
		p := make([]Point, 0, cc.R)
		for j := 0; j < cc.R; j++ {
			p = append(p, Point{rng.NormFloat64(), rng.NormFloat64()})
		}
		pop = append(pop, Genome{Points: p})
	}
	generation := 0
	for {
		for i := 0; i < len(pop); i++ {
			pop[i].Fitness = fitness(pop[i])
		}
		sort.Slice(pop, func(i, j int) bool {
			return pop[i].Fitness < pop[j].Fitness
		})
		if pop[0].Fitness < 0.0001 || generation > 8*1024 {
			return pop[0].Points
		}
		fmt.Println(pop[0].Fitness)
		pop = pop[:128]
		length := len(pop)
		for i := 0; i < length/2; i++ {
			a := rng.Intn(len(pop)) / 2
			b := rng.Intn(len(pop)) / 2
			pointsA := make([]Point, len(pop[a].Points))
			copy(pointsA, pop[a].Points)
			pointsB := make([]Point, len(pop[b].Points))
			copy(pointsB, pop[b].Points)
			x := &pointsA[rng.Intn(len(pop[a].Points))]
			y := &pointsB[rng.Intn(len(pop[b].Points))]
			x.X, y.X = y.X, x.X
			pop = append(pop, Genome{Points: pointsA})
			pop = append(pop, Genome{Points: pointsB})
		}
		for i := 0; i < length; i++ {
			a := rng.Intn(len(pop)) / 2
			pointsA := make([]Point, len(pop[a].Points))
			copy(pointsA, pop[a].Points)
			x := &pointsA[rng.Intn(len(pop[a].Points))]
			x.X += rng.NormFloat64()
			x.Y += rng.NormFloat64()
			pop = append(pop, Genome{Points: pointsA})
		}
		generation++
	}
}
