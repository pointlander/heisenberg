// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/pointlander/heisenberg"
)

func main() {
	heisenberg.Optimize(8, 8, [][2][]float64{
		[2][]float64{[]float64{0, 1}, []float64{1, 0}},
		[2][]float64{[]float64{1, 0}, []float64{0, 1}},
	})

	rnd := rand.New(rand.NewSource(1))
	for i := .1; i <= 1.0; i += .1 {
		start := time.Now()
		for j := 0; j < 256; j++ {
			matrix := heisenberg.Sparse128{
				R:      256,
				C:      256,
				Matrix: make([]map[int]complex128, 256),
			}
			for x := 0; x < matrix.R; x++ {
				for y := 0; y < matrix.C; y++ {
					if rnd.Float64() < i {
						row := matrix.Matrix[x]
						if row == nil {
							row = make(map[int]complex128)
						}
						row[y] = complex(rnd.Float64(), rnd.Float64())
						matrix.Matrix[x] = row
					}
				}
			}
			matrix.Multiply(&matrix)
		}
		fmt.Printf("%f %v\n", i, time.Now().Sub(start))
	}

	start := time.Now()
	for j := 0; j < 256; j++ {
		matrix := heisenberg.Dense128{
			R:      256,
			C:      256,
			Matrix: make([]complex128, 0, 256*256),
		}
		for i := 0; i < 256*256; i++ {
			matrix.Matrix = append(matrix.Matrix, complex(rnd.Float64(), rnd.Float64()))
		}
		matrix.Multiply(&matrix)
	}
	fmt.Printf("%v\n", time.Now().Sub(start))
}
