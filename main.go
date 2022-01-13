// Copyright 2022 The QLFSR Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

// Matrix is an algebriac matrix
type Matrix struct {
	R, C   int
	Matrix []complex64
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
	for i := 0; i < unitary.R; i++ {
		for j := 0; j < unitary.C; j++ {
			fmt.Printf("%f ", unitary.Matrix[i*unitary.C+j])
		}
		fmt.Printf("\n")
	}

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
	for i := 0; i < unitary.R; i++ {
		for j := 0; j < unitary.C; j++ {
			fmt.Printf("%f ", unitary.Matrix[i*unitary.C+j])
		}
		fmt.Printf("\n")
	}
}
