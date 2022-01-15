// Copyright 2022 The QLFSR Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
)

// Matrix is an algebriac matrix
type Matrix struct {
	R, C   int
	Matrix []complex64
}

func (m Matrix) String() string {
	output := ""
	for i := 0; i < m.R; i++ {
		for j := 0; j < m.C; j++ {
			output += fmt.Sprintf("%f ", m.Matrix[i*m.C+j])
		}
		output += fmt.Sprintf("\n")
	}
	return output
}

// Tensor product is the tensor product
func Tensor(a *Matrix, b *Matrix) *Matrix {
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
	return &Matrix{
		R:      a.R * b.R,
		C:      a.C * b.C,
		Matrix: output,
	}
}

// Multiply multiplies to matricies
func Multiply(a *Matrix, b *Matrix) *Matrix {
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
	return &Matrix{
		R:      a.R,
		C:      b.C,
		Matrix: output,
	}
}

// Transpose transposes a matrix
func Transpose(a *Matrix) {
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			a.Matrix[j*a.C + i] = a.Matrix[i*a.C + j]
		}
	}
	a.R, a.C = a.C, a.R
}

// Copy copies a matrix`
func Copy(a *Matrix) *Matrix {
	cp := &Matrix{
		R: a.R,
		C: a.C,
		Matrix: make([]complex64, len(a.Matrix)),
	}
	copy(cp.Matrix, a.Matrix)
	return cp
}

// ControlledNot controlled not gate
func ControlledNot(n int, c []int, t int) *Matrix {
	m := &Matrix{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, 1,
		},
	}
	for i := 0; i < n-1; i++ {
		m = Tensor(m, m)
	}
	d := m.R
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

	g := Matrix{
		R:      m.R,
		C:      m.C,
		Matrix: make([]complex64, m.R*m.C),
	}
	for i, ii := range index {
		copy(g.Matrix[i*g.C:(i+1)*g.C], m.Matrix[int(ii)*g.C:int(ii + 1)*g.C])
	}

	return &g
}

func main() {
	cnot := Matrix{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, 1,
		},
	}
	unitary := Tensor(&cnot, &cnot)
	fmt.Printf("%s", unitary)

	cnot = Matrix{
		R: 4,
		C: 4,
		Matrix: []complex64{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 1,
			0, 0, 1, 0,
		},
	}
	fmt.Printf("\n")
	unitary = Multiply(&cnot, &cnot)
	fmt.Printf("%s", unitary)

	fmt.Printf("\n")
	a := ControlledNot(3, []int{0}, 1)
	fmt.Printf("%s", a)

	fmt.Printf("\n")
	b := ControlledNot(3, []int{0, 1}, 2)
	fmt.Printf("%s", b)

	fmt.Printf("\n")
	c := Multiply(a, b)
	fmt.Printf("%s", c)

	fmt.Printf("\n")
	d := Copy(c)
	Transpose(d)
	var sum complex64
	for i := range c.Matrix {
		sum += d.Matrix[i] - c.Matrix[i]
	}
	fmt.Println(sum)
}
