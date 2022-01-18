// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"testing"
)

func BenchmarkMatrix64(b *testing.B) {
	for n := 0; n < b.N; n++ {
		a := ControlledNot64(3, []int{0}, 1)
		b := ControlledNot64(3, []int{0, 1}, 2)
		c := a.Multiply(b)
		_ = c
	}
}

func BenchmarkMatrix128(b *testing.B) {
	for n := 0; n < b.N; n++ {
		a := ControlledNot128(3, []int{0}, 1)
		b := ControlledNot128(3, []int{0, 1}, 2)
		c := a.Multiply(b)
		_ = c
	}
}
