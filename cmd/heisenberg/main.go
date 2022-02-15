// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/pointlander/heisenberg"
)

func main() {
	heisenberg.Optimize(8, 8, [][2][]float64{
		[2][]float64{[]float64{0, 1}, []float64{1, 0}},
		[2][]float64{[]float64{1, 0}, []float64{0, 1}},
	})
}
