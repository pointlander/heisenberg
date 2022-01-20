// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

func main() {
	cnot := Matrix64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, 1,
		},
	}
	unitary := cnot.Tensor(&cnot)
	fmt.Printf("%s", unitary)

	cnot = Matrix64{
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
	unitary = cnot.Multiply(&cnot)
	fmt.Printf("%s", unitary)

	fmt.Printf("\n")
	a := Matrix64{}.ControlledNot(3, []int{0}, 2)
	fmt.Printf("%s", a)

	fmt.Printf("\n")
	b := Matrix64{}.ControlledNot(3, []int{0, 1}, 2)
	fmt.Printf("%s", b)

	fmt.Printf("\n")
	c := b.Multiply(a)
	fmt.Printf("%s", c)

	fmt.Printf("\n")
	d := c.Copy()
	d.Transpose()
	var sum complex64
	for i := range c.Matrix {
		sum += d.Matrix[i] - c.Matrix[i]
	}
	fmt.Println(sum)

	state := Matrix64{}
	state.One()
	state.One()
	state.Zero()
	fmt.Println(state)

	fmt.Printf("\n")
	state.Transpose()
	output := c.Multiply(&state)
	fmt.Println(output)
}
