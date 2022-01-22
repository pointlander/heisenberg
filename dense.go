// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
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
func (a *MachineDense64) Zero() {
	a.Qubits++
	zero := Dense64{
		R: 2,
		C: 1,
		Matrix: []complex64{
			1, 0,
		},
	}
	if a.C == 0 {
		a.Dense64 = zero
		return
	}
	a.Dense64 = *a.Tensor(&zero)
}

// One adds a one to the matrix
func (a *MachineDense64) One() {
	a.Qubits++
	one := Dense64{
		R: 2,
		C: 1,
		Matrix: []complex64{
			0, 1,
		},
	}
	if a.C == 0 {
		a.Dense64 = one
		return
	}
	a.Dense64 = *a.Tensor(&one)
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
func (a *MachineDense64) ControlledNot(c []int, t int) *Dense64 {
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
func (a *MachineDense128) Zero() {
	a.Qubits++
	zero := Dense128{
		R: 2,
		C: 1,
		Matrix: []complex128{
			1, 0,
		},
	}
	if a.C == 0 {
		a.Dense128 = zero
		return
	}
	a.Dense128 = *a.Tensor(&zero)
}

// One adds a one to the matrix
func (a *MachineDense128) One() {
	a.Qubits++
	one := Dense128{
		R: 2,
		C: 1,
		Matrix: []complex128{
			0, 1,
		},
	}
	if a.C == 0 {
		a.Dense128 = one
		return
	}
	a.Dense128 = *a.Tensor(&one)
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
func (a *MachineDense128) ControlledNot(c []int, t int) *Dense128 {
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
