// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/pointlander/heisenberg"
)

func main() {
	machine := heisenberg.MachineDense128{}
	/*q0 := machine.Zero()
	q1 := machine.One()
	machine.Swap(q0, q1)*/
	points := machine.Points()
	fmt.Println(points)
}
