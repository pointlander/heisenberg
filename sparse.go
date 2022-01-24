// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
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

// MachineSparse64 is a 64 bit sparse matrix machine
type MachineSparse64 struct {
	Sparse64
	Qubits int
}

// Zero adds a zero to the matrix
func (a *MachineSparse64) Zero() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	zero := Sparse64{
		R: 2,
		C: 1,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
			},
			map[int]complex64{
				0: 0,
			},
		},
	}
	if qubit == 0 {
		a.Sparse64 = zero
		return qubit
	}
	a.Sparse64 = *a.Tensor(&zero)
	return qubit
}

// One adds a one to the matrix
func (a *MachineSparse64) One() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	one := Sparse64{
		R: 2,
		C: 1,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 0,
			},
			map[int]complex64{
				0: 1,
			},
		},
	}
	if qubit == 0 {
		a.Sparse64 = one
		return qubit
	}
	a.Sparse64 = *a.Tensor(&one)
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

// ControlledNot controlled not gate
func (a *MachineSparse64) ControlledNot(c []Qubit, t Qubit) *Sparse64 {
	n := a.Qubits
	p := &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
				1: 0,
			},
			map[int]complex64{
				0: 0,
				1: 1,
			},
		},
	}
	q := p
	for i := 0; i < n-1; i++ {
		q = p.Tensor(q)
	}
	d := q.R
	f := fmt.Sprintf("%s%s%s", "%0", strconv.Itoa(n), "s")

	index := make([]int64, 0)
	for i := 0; i < d; i++ {
		bits := []rune(fmt.Sprintf(f, strconv.FormatInt(int64(i), 2)))

		// Apply X
		apply := true
		for _, j := range c {
			if bits[j] == '0' {
				apply = false
				break
			}
		}

		if apply {
			if bits[t] == '0' {
				bits[t] = '1'
			} else {
				bits[t] = '0'
			}
		}

		v, err := strconv.ParseInt(string(bits), 2, 0)
		if err != nil {
			panic(fmt.Sprintf("parse int: %v", err))
		}

		index = append(index, v)
	}

	g := Sparse64{
		R:      q.R,
		C:      q.C,
		Matrix: make([]map[int]complex64, q.R),
	}
	for i, ii := range index {
		g.Matrix[i] = q.Matrix[int(ii)]
	}

	a.Sparse64 = *g.Multiply(&a.Sparse64)

	return &g
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
	Sparse128
	Qubits int
}

// Zero adds a zero to the matrix
func (a *MachineSparse128) Zero() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	zero := Sparse128{
		R: 2,
		C: 1,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				0: 0,
			},
		},
	}
	if qubit == 0 {
		a.Sparse128 = zero
		return qubit
	}
	a.Sparse128 = *a.Tensor(&zero)
	return qubit
}

// One adds a one to the matrix
func (a *MachineSparse128) One() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	one := Sparse128{
		R: 2,
		C: 1,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 0,
			},
			map[int]complex128{
				0: 1,
			},
		},
	}
	if qubit == 0 {
		a.Sparse128 = one
		return qubit
	}
	a.Sparse128 = *a.Tensor(&one)
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

// ControlledNot controlled not gate
func (a *MachineSparse128) ControlledNot(c []Qubit, t Qubit) *Sparse128 {
	n := a.Qubits
	p := &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
				1: 0,
			},
			map[int]complex128{
				0: 0,
				1: 1,
			},
		},
	}
	q := p
	for i := 0; i < n-1; i++ {
		q = p.Tensor(q)
	}
	d := q.R
	f := fmt.Sprintf("%s%s%s", "%0", strconv.Itoa(n), "s")

	index := make([]int64, 0)
	for i := 0; i < d; i++ {
		bits := []rune(fmt.Sprintf(f, strconv.FormatInt(int64(i), 2)))

		// Apply X
		apply := true
		for _, j := range c {
			if bits[j] == '0' {
				apply = false
				break
			}
		}

		if apply {
			if bits[t] == '0' {
				bits[t] = '1'
			} else {
				bits[t] = '0'
			}
		}

		v, err := strconv.ParseInt(string(bits), 2, 0)
		if err != nil {
			panic(fmt.Sprintf("parse int: %v", err))
		}

		index = append(index, v)
	}

	g := Sparse128{
		R:      q.R,
		C:      q.C,
		Matrix: make([]map[int]complex128, q.R),
	}
	for i, ii := range index {
		g.Matrix[i] = q.Matrix[int(ii)]
	}

	a.Sparse128 = *g.Multiply(&a.Sparse128)

	return &g
}
