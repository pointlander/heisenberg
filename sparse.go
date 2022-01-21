// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/bits"
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

// Zero adds a zero to the matrix
func (a *Sparse64) Zero() {
	zero := Sparse64{
		R: 1,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
				1: 0,
			},
		},
	}
	if a.C == 0 {
		*a = zero
		return
	}
	*a = *a.Tensor(&zero)
}

// One adds a one to the matrix
func (a *Sparse64) One() {
	one := Sparse64{
		R: 1,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 0,
				1: 1,
			},
		},
	}
	if a.C == 0 {
		*a = one
		return
	}
	*a = *a.Tensor(&one)
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
func (a *Sparse64) ControlledNot(c []int, t int) *Sparse64 {
	n := 64 - bits.LeadingZeros64(uint64(a.R*a.C)-1)
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

// Zero adds a zero to the matrix
func (a *Sparse128) Zero() {
	zero := Sparse128{
		R: 1,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
				1: 0,
			},
		},
	}
	if a.C == 0 {
		*a = zero
		return
	}
	*a = *a.Tensor(&zero)
}

// One adds a one to the matrix
func (a *Sparse128) One() {
	one := Sparse128{
		R: 1,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 0,
				1: 1,
			},
		},
	}
	if a.C == 0 {
		*a = one
		return
	}
	*a = *a.Tensor(&one)
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
func (a *Sparse128) ControlledNot(c []int, t int) *Sparse128 {
	n := 64 - bits.LeadingZeros64(uint64(a.R*a.C)-1)
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

	return &g
}
